"""ROS 2 node: synchronized OV9282 stereo pair of the RB3 Gen 2 vision mezzanine.

Publishes (relative to the node namespace, /stereo with the launch file):
  left/image_raw, right/image_raw      sensor_msgs/Image mono8, identical header.stamp per pair
  left/camera_info, right/camera_info  sensor_msgs/CameraInfo (calibration YAML or nominal model)
  preview/compressed                   sensor_msgs/CompressedImage, side-by-side JPEG for remote viewing
  status                               std_msgs/String, JSON: fps, sync offset, exposure, gain, ...
All camera settings are ROS parameters and can be changed while running (ros2 param set / viewer).
"""
import array
import json
import signal
import threading
import time

import cv2
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from . import camera_info
from .stereo import MODES, AutoExposure, StereoCamera

RESOLUTIONS = {"640x400": (640, 400), "1280x800": (1280, 800)}
RESTART = ("resolution", "left_camera", "flip")      # need a pipeline restart
INFO = ("left_calibration", "right_calibration", "nominal_vfov_deg", "baseline_m")


class StereoCameraNode(Node):
    def __init__(self):
        super().__init__("stereo_camera")
        d = lambda text: ParameterDescriptor(description=text, dynamic_typing=True)  # 3000 or 3000.0
        P = {
            "resolution": ("640x400", "640x400 (up to 120 fps) or 1280x800 (up to 60 fps)"),
            "fps": (30, "frame rate: 30, 60 or 120 (640x400 only)"),
            "left_camera": ("cam0a", "which connector is the left camera: cam0a or cam0b"),
            "flip": (False, "rotate both images 180 deg (sensor readout flip)"),
            "genlock": (True, "keep the two cameras' frames aligned (software genlock)"),
            "auto_exposure": (True, "software auto exposure; exposure_us/gain apply when false"),
            "exposure_us": (2000.0, "manual exposure time (us), limited by the frame time"),
            "gain": (1.0, "manual analogue gain 1.0 .. 15.9"),
            "ae_target": (100.0, "auto exposure: target mean brightness 0-255"),
            "ae_max_exposure_us": (4000.0, "auto exposure: exposure limit (motion blur)"),
            "ae_max_gain": (8.0, "auto exposure: gain limit (noise)"),
            "stamp": ("exposure_center", "header.stamp = exposure_center or frame_end"),
            "frame_id_left": ("stereo_left_optical_frame", "frame_id of the left images"),
            "frame_id_right": ("stereo_right_optical_frame", "frame_id of the right images"),
            "left_calibration": ("", "camera_calibration YAML of the left camera ('' = nominal model)"),
            "right_calibration": ("", "camera_calibration YAML of the right camera ('' = nominal model)"),
            "nominal_vfov_deg": (85.0, "vertical field of view of the nominal model (sim: 85)"),
            "baseline_m": (0.05, "baseline of the nominal model in m (sim: 0.05)"),
            "preview_fps": (10.0, "rate of preview/compressed, 0 = off"),
            "preview_scale": (0.5, "preview image scale"),
            "preview_quality": (75, "preview JPEG quality"),
            "qos_reliable": (False, "image/camera_info QoS reliable instead of best effort (startup only)"),
        }
        for name, (default, text) in P.items():
            self.declare_parameter(name, default, d(text))
        self.p = {name: self.get_parameter(name).value for name in P}

        # best effort is the camera default; reliable for tools that only subscribe reliably
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE) if self.p["qos_reliable"]             else qos_profile_sensor_data
        self.pub = {
            side: (self.create_publisher(Image, f"{side}/image_raw", qos),
                   self.create_publisher(camera_info.CameraInfo, f"{side}/camera_info", qos))
            for side in ("left", "right")}
        self.pub_preview = self.create_publisher(CompressedImage, "preview/compressed", qos_profile_sensor_data)
        self.pub_status = self.create_publisher(String, "status", 10)

        self._pending, self._pending_lock = {}, threading.Lock()
        self.add_on_set_parameters_callback(self._validate)
        self.add_post_set_parameters_callback(self._queue)

        w, h = RESOLUTIONS[self.p["resolution"]]
        self.cam = StereoCamera(w, h, int(self.p["fps"]), left=self.p["left_camera"], flip=bool(self.p["flip"]),
                                genlock=bool(self.p["genlock"]), exposure_us=float(self.p["exposure_us"]),
                                gain=float(self.p["gain"]), log=self.get_logger().info)
        self.ae = AutoExposure(float(self.p["ae_target"]), float(self.p["ae_max_exposure_us"]),
                               float(self.p["ae_max_gain"]))
        self._load_info()
        self.cam.start()
        self.get_logger().info(f"streaming {w}x{h} @ {self.cam.fps} fps, sync offset "
                               f"{(self.cam.offset or 0) * 1e6:+.0f} us")
        self._clock_offset()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ parameters

    def _validate(self, params):
        p = dict(self.p, **{x.name: x.value for x in params})
        if p["resolution"] not in RESOLUTIONS:
            return SetParametersResult(successful=False, reason=f"resolution: {', '.join(RESOLUTIONS)}")
        modes = MODES[RESOLUTIONS[p["resolution"]]]
        if int(p["fps"]) not in modes:
            return SetParametersResult(successful=False,
                                       reason=f"{p['resolution']} supports fps {', '.join(map(str, modes))}")
        if p["left_camera"] not in ("cam0a", "cam0b"):
            return SetParametersResult(successful=False, reason="left_camera: cam0a or cam0b")
        if p["stamp"] not in ("exposure_center", "frame_end"):
            return SetParametersResult(successful=False, reason="stamp: exposure_center or frame_end")
        return SetParametersResult(successful=True)

    def _queue(self, params):
        with self._pending_lock:
            for x in params:
                self._pending[x.name] = x.value

    def _apply_pending(self):
        with self._pending_lock:
            changes, self._pending = self._pending, {}
        if not changes:
            return
        self.p.update(changes)
        log = self.get_logger().info
        if any(k in changes for k in RESTART):
            w, h = RESOLUTIONS[self.p["resolution"]]
            log(f"restarting: {w}x{h} @ {self.p['fps']} fps, left={self.p['left_camera']}, flip={self.p['flip']}")
            self.cam.restart(width=w, height=h, fps=int(self.p["fps"]), left=self.p["left_camera"],
                             flip=bool(self.p["flip"]))
            self._load_info()
        elif "fps" in changes:
            self.cam.set_fps(int(self.p["fps"]))
            log(f"fps -> {self.cam.fps}")
        if "genlock" in changes:
            self.cam.genlock = bool(self.p["genlock"])
            if self.cam.genlock:
                self.cam.align()
        self.ae.target = float(self.p["ae_target"])
        self.ae.max_exposure_us = float(self.p["ae_max_exposure_us"])
        self.ae.max_gain = float(self.p["ae_max_gain"])
        if not self.p["auto_exposure"] and ({"exposure_us", "gain", "auto_exposure"} & changes.keys()):
            exp, gain = self.cam.set_exposure(float(self.p["exposure_us"]), float(self.p["gain"]))
            log(f"manual exposure {exp:.0f} us, gain {gain:.2f}")
        if any(k in changes for k in INFO):
            self._load_info()

    def _load_info(self):
        w, h = self.cam.w, self.cam.h
        self.info = {}
        for side in ("left", "right"):
            path = self.p[f"{side}_calibration"]
            try:
                self.info[side] = camera_info.load(path, w, h) if path else camera_info.nominal(
                    w, h, float(self.p["nominal_vfov_deg"]), float(self.p["baseline_m"]), right=side == "right")
            except Exception as e:
                self.get_logger().error(f"{side} calibration {path}: {e}; using the nominal model")
                self.info[side] = camera_info.nominal(w, h, float(self.p["nominal_vfov_deg"]),
                                                      float(self.p["baseline_m"]), right=side == "right")
            self.info[side].header.frame_id = self.p[f"frame_id_{side}"]

    # ------------------------------------------------------------ capture loop

    def _clock_offset(self):
        """ROS time minus CLOCK_MONOTONIC (frame timestamps are CLOCK_MONOTONIC)."""
        best = None
        for _ in range(5):
            m0 = time.monotonic()
            r = self.get_clock().now().nanoseconds * 1e-9
            m1 = time.monotonic()
            if best is None or m1 - m0 < best[0]:
                best = (m1 - m0, r - (m0 + m1) / 2)
        self._mono_to_ros, self._t_offset = best[1], time.monotonic()

    def _image(self, frame, stamp, frame_id):
        msg = Image(height=frame.shape[0], width=frame.shape[1], encoding="mono8", is_bigendian=0,
                    step=frame.shape[1])
        msg.header.stamp, msg.header.frame_id = stamp, frame_id
        data = array.array("B")
        data.frombytes(np.ascontiguousarray(frame))
        msg.data = data
        return msg

    def _loop(self):
        misses, n, t_stat, t_prev = 0, 0, time.monotonic(), 0.0
        while rclpy.ok() and not self._stop:
            try:
                self._apply_pending()
                pair = self.cam.read_pair(timeout=1.0)
                if pair is None:
                    misses += 1
                    self.get_logger().warning("no stereo pair for 1 s")
                    if misses >= 2:
                        self.get_logger().warning("restarting the camera pipelines")
                        self.cam.restart()
                        misses = 0
                    continue
                misses = 0
                if time.monotonic() - self._t_offset > 1.0:
                    self._clock_offset()
                t = (pair.t_left + pair.t_right) / 2
                if self.p["stamp"] == "exposure_center":
                    t -= self.cam.readout_s + self.cam.exposure_us * 0.5e-6
                stamp = Time(seconds=t + self._mono_to_ros).to_msg()
                for side, frame in (("left", pair.left), ("right", pair.right)):
                    img_pub, info_pub = self.pub[side]
                    img_pub.publish(self._image(frame, stamp, self.p[f"frame_id_{side}"]))
                    self.info[side].header.stamp = stamp
                    info_pub.publish(self.info[side])
                if self.p["auto_exposure"]:
                    self.ae.update(self.cam, pair.left, pair.right)
                now = time.monotonic()
                pf = float(self.p["preview_fps"])
                if pf > 0 and now - t_prev >= 1.0 / pf:
                    t_prev = now
                    self._preview(pair, stamp)
                n += 1
                if now - t_stat >= 5.0:
                    self._status(n / (now - t_stat), pair)
                    n, t_stat = 0, now
            except Exception as e:
                if self._stop or not rclpy.ok():
                    break                                # shutting down
                self.get_logger().error(f"capture loop: {e!r}")
                time.sleep(1.0)

    def _preview(self, pair, stamp):
        img = np.hstack((pair.left, pair.right))
        s = float(self.p["preview_scale"])
        if s != 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(self.p["preview_quality"])])
        if ok:
            msg = CompressedImage(format="jpeg", data=jpg.tobytes())
            msg.header.stamp, msg.header.frame_id = stamp, "stereo_preview"
            self.pub_preview.publish(msg)

    def _status(self, fps, pair):
        st = {"fps": round(fps, 2), "mode": f"{self.cam.w}x{self.cam.h}@{self.cam.fps}",
              "sync_offset_us": None if self.cam.offset is None else round(self.cam.offset * 1e6, 1),
              "pair_offset_us": round((self.cam.pair_offset() or 0) * 1e6, 1),
              "exposure_us": round(self.cam.exposure_us), "gain": round(self.cam.gain, 2),
              "auto_exposure": self.p["auto_exposure"],
              "mean": None if self.ae.mean is None else round(self.ae.mean, 1)}
        self.pub_status.publish(String(data=json.dumps(st)))
        self.get_logger().info(" ".join(f"{k}={v}" for k, v in st.items()))

    def destroy_node(self):
        self._stop = True
        try:
            self._thread.join(timeout=2)
        except KeyboardInterrupt:
            pass
        self.cam.stop()
        super().destroy_node()


def main(args=None):
    # started from a background job SIGINT arrives "ignored" and launch could never stop the node
    signal.signal(signal.SIGINT, signal.default_int_handler)
    rclpy.init(args=args)
    node = StereoCameraNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
