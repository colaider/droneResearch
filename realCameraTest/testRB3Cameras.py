"""Run the swarm estimator on the drone's RB3 Gen 2 stereo cameras (two OV9282 on the vision
mezzanine) and stream a live view over ROS 2, so it can be watched from WSL or any machine on the
same ROS_DOMAIN_ID. The simulation does not use this file.

Same inputs as testRealCameras.py: there is no flight controller, so all drone/IMU inputs are ZERO
(except an optional assumed height, --height, which gives the flow a metric scale), and stereo depth
is off until the cameras are calibrated and stereo-rectified (--stereo-depth turns it on).

On the drone, from the repo root:
    source ~/ros2_ws/install/setup.bash
    python3 realCameraTest/testRB3Cameras.py                 # 640x400 @ 30 fps
    python3 realCameraTest/testRB3Cameras.py --height 1.0    # assume 1 m above the scene
On WSL / another machine:
    bash scripts/view_stereo.sh --ns /vo

Publishes /vo/preview/compressed (left|right with the tracking overlay, JPEG) and /vo/status (JSON).
Ctrl-C stops.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Allow running the file directly from any working directory.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from swarm.compVision.visualAccEst import VisulaAcEst   # the real estimator, used as-is

BIG_CORES, LITTLE_CORES = {4, 5, 6, 7}, {0, 1, 2, 3}     # Cortex-A78 / Cortex-A55 of the QCS6490


def pin(cores):
    """Pin the calling thread; threads and processes it starts afterwards inherit this."""
    try:
        os.sched_setaffinity(0, cores)
    except (AttributeError, OSError):
        pass


def set_drone_inputs(est, height):
    """No flight controller: every drone/IMU input is ZERO, apart from the assumed height."""
    est.drone_pos = np.array([0.0, 0.0, height])
    est.drone_vel = np.zeros(3)
    est.drone_ang_vel = np.zeros(3)
    est.imu_att = np.zeros(3)
    est.previous_cmd_vel = np.zeros(4)
    est.drone_ang = np.zeros(2)
    est.push_sensors()


class LiveView:
    """Preview + status over ROS 2 for view_stereo.sh --ns <namespace>. The node is named like the
    camera node (stereo_camera), so the viewer's keys change exposure, gain, auto exposure and fps."""

    def __init__(self, namespace, preview_fps, scale, settings):
        import threading
        import rclpy
        from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String
        self.rclpy, self.CompressedImage, self.String = rclpy, CompressedImage, String
        rclpy.init()
        self.node = rclpy.create_node("stereo_camera", namespace=namespace)
        self.pub_img = self.node.create_publisher(CompressedImage, "preview/compressed", qos_profile_sensor_data)
        self.pub_status = self.node.create_publisher(String, "status", 10)
        self.period, self.scale, self.t_img = 1.0 / preview_fps if preview_fps > 0 else math.inf, scale, 0.0

        self._pending, self._lock = {}, threading.Lock()
        for name, value in settings.items():
            self.node.declare_parameter(name, value, ParameterDescriptor(dynamic_typing=True))

        def validate(params):
            for p in params:
                if p.name == "resolution" and p.value != settings["resolution"]:
                    return SetParametersResult(successful=False, reason="restart the runner with --resolution")
                if p.name == "fps" and int(p.value) not in (30, 60, 120):
                    return SetParametersResult(successful=False, reason="fps: 30, 60 or 120")
            with self._lock:
                self._pending.update({p.name: p.value for p in params})
            return SetParametersResult(successful=True)

        self.node.add_on_set_parameters_callback(validate)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spinning = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        while self._spinning and self.rclpy.ok():
            self._executor.spin_once(timeout_sec=0.1)

    def changes(self):
        """Settings changed from the viewer since the last call."""
        with self._lock:
            out, self._pending = self._pending, {}
        return out

    def frames(self, frames):
        now = time.monotonic()
        if now - self.t_img < self.period:
            return
        self.t_img = now
        img = cv2.hconcat(list(frames))
        if self.scale != 1.0:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            msg = self.CompressedImage(format="jpeg", data=jpg.tobytes())
            msg.header.stamp = self.node.get_clock().now().to_msg()
            self.pub_img.publish(msg)

    def status(self, status):
        self.pub_status.publish(self.String(data=json.dumps(status)))

    def close(self):
        self._spinning = False
        self._thread.join(timeout=1.0)
        self._executor.shutdown()
        self.node.destroy_node()
        self.rclpy.try_shutdown()


def apply_changes(changes, cam, ae, args, fps):
    """Viewer settings -> camera. Returns the (possibly new) auto exposure and frame rate."""
    if "fps" in changes and int(changes["fps"]) != fps:
        fps = int(changes["fps"])
        cam.set_fps(fps)
    if "auto_exposure" in changes or "exposure_us" in changes or "gain" in changes:
        auto = changes.get("auto_exposure", ae is not None)
        if auto:
            ae = ae or AutoExposureFactory(args)
        else:
            ae = None
            cam.set_exposure(float(changes.get("exposure_us", cam.exposure_us)), float(changes.get("gain", cam.gain)))
    return ae, fps


def AutoExposureFactory(args):
    from rb3_stereo.stereo import AutoExposure
    return AutoExposure(target=args.ae_target, max_exposure_us=args.ae_max_exposure_us)


def main(args):
    from rb3_stereo.stereo import StereoCamera   # only available on the drone
    np.seterr(divide='ignore', invalid='ignore')   # zero drone height -> undefined depth

    w, h = map(int, args.resolution.split('x'))
    pin(LITTLE_CORES)                               # camera capture threads + v4l2-ctl inherit this
    manual = args.exposure_us is not None
    cam = StereoCamera(w, h, args.fps, exposure_us=args.exposure_us if manual else 2000.0,
                       gain=args.gain if manual else 1.0).start()
    ae = None if manual else AutoExposureFactory(args)
    fps = args.fps
    view = None if args.no_ros else LiveView(
        args.ns, args.preview_fps, args.preview_scale,
        dict(resolution=args.resolution, fps=fps, auto_exposure=not manual,
             exposure_us=float(cam.exposure_us), gain=float(cam.gain)))
    pin(BIG_CORES)                                  # estimator + OpenCV's worker threads
    cv2.setNumThreads(len(BIG_CORES))

    est = VisulaAcEst((w, h), args.vfov)
    est.camera_saperation = args.baseline
    est.use_stereo_depth = args.stereo_depth
    print(f"{w}x{h} @ {args.fps} fps | focal {est.foc_l:.0f}px (vfov {args.vfov} deg) baseline {args.baseline} m | "
          f"stereo depth {'on' if est.use_stereo_depth else 'off'} | assumed height {args.height} m"
          + ("" if args.no_ros else f" | live view: view_stereo.sh --ns {args.ns}"))

    idx, prev_t, t_end = 0, None, time.monotonic() + args.seconds if args.seconds else math.inf
    n, busy, t_report = 0, 0.0, time.monotonic()
    try:
        while time.monotonic() < t_end:
            pair = cam.read_pair(timeout=1.0)
            if pair is None:
                print("no stereo pair for 1 s")
                continue
            est.set_dt(max(pair.t_left - prev_t, 1e-3) if prev_t is not None else 1 / args.fps)
            prev_t = pair.t_left
            set_drone_inputs(est, args.height)
            frames = [cv2.cvtColor(pair.left, cv2.COLOR_GRAY2BGR), cv2.cvtColor(pair.right, cv2.COLOR_GRAY2BGR)]
            t0 = time.perf_counter()
            result = est.processing(frames, idx)       # the estimator takes BGR frames
            busy += time.perf_counter() - t0
            if ae:
                ae.update(cam, pair.left, pair.right)
            if view:
                view.frames(result.frames)
                changes = view.changes()
                if changes:
                    ae, fps = apply_changes(changes, cam, ae, args, fps)
                    print(f"viewer: {changes} -> exposure {cam.exposure_us:.0f} us, gain {cam.gain:.2f}, "
                          f"auto exposure {'on' if ae else 'off'}, {fps} fps", flush=True)
            idx, n = idx + 1, n + 1
            now = time.monotonic()
            if now - t_report >= 1.0:
                vx, vy, vz, yaw = (float(x) for x in est.camera_velocity)
                status = {"mode": f"{w}x{h}@{fps}", "fps": round(n / (now - t_report), 1),
                          "estimator_ms": round(busy / n * 1e3, 1),
                          "tracked": int(len(est.tracked_points.get(0, []))),
                          "vx": round(vx, 3), "vy": round(vy, 3), "yaw_rate": round(yaw, 3),
                          "depth_m": None if math.isnan(est.depth_median) else round(float(est.depth_median), 2),
                          "exposure_us": round(cam.exposure_us), "gain": round(cam.gain, 2),
                          "auto_exposure": ae is not None}
                print(" | ".join(f"{k} {v}" for k, v in status.items()), flush=True)
                if view:
                    view.status(status)
                n, busy, t_report = 0, 0.0, now
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        if view:
            view.close()
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--resolution', choices=('640x400', '1280x800'), default='640x400')
    p.add_argument('--fps', type=int, choices=(30, 60, 120), default=30)
    p.add_argument('--exposure-us', type=float, default=None, help='manual exposure (default: auto exposure)')
    p.add_argument('--gain', type=float, default=1.0, help='analogue gain 1.0-15.9 with --exposure-us')
    p.add_argument('--ae-target', type=float, default=100.0, help='auto exposure: mean brightness 0-255')
    p.add_argument('--ae-max-exposure-us', type=float, default=4000.0, help='auto exposure: limit (motion blur)')
    p.add_argument('--vfov', type=float, default=85.0,
                   help='vertical field of view in degrees until the cameras are calibrated')
    p.add_argument('--baseline', type=float, default=0.05, help='stereo baseline in m')
    p.add_argument('--height', type=float, default=0.0, help='assumed height above the scene in m (0 = like testRealCameras)')
    p.add_argument('--stereo-depth', action='store_true', help='stereo depth (needs calibrated, rectified cameras)')
    p.add_argument('--ns', default='/vo', help='ROS namespace of the live view')
    p.add_argument('--preview-fps', type=float, default=10.0)
    p.add_argument('--preview-scale', type=float, default=0.5)
    p.add_argument('--no-ros', action='store_true', help='console output only')
    p.add_argument('--seconds', type=float, default=0, help='stop after this many seconds (0 = run until Ctrl-C)')
    args = p.parse_args(argv)
    if args.resolution == '1280x800' and args.fps == 120:
        p.error('1280x800 supports 30 or 60 fps')
    return args


if __name__ == '__main__':
    raise SystemExit(main(parse_args()))
