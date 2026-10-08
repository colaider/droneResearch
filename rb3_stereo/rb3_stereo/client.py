"""Drop-in stereo source for the swarm vision code: same interface as the simulation's DroneStruct
(get_two_frames(), res, fov, camera_saperation), fed from the ROS topics. The topics come either
from the real camera node (rb3_stereo stereo_camera) or from the simulation (StereoPublisher), so
the vision code does not change between the two.

    from rb3_stereo.client import StereoSubscriber
    cams = StereoSubscriber("/stereo")                 # starts its own ROS node + spin thread
    left, right = cams.get_two_frames()               # BGR uint8, like DroneStruct.get_two_frames()
    proc = VisulaAcEst(cams.res, cams.fov)            # res = (width, height), fov = vertical deg
    proc.camera_saperation = cams.camera_saperation
"""
import os
import threading

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from . import camera_info


def _use_large_shm_profile():
    """Full-rate 1280x800 images (1 MB) need bigger Fast DDS shared-memory segments than the default
    512 KB, in the camera node and in every subscriber on the drone: use the package's profile unless
    the environment already selects one."""
    if os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE") or os.environ.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp") \
            != "rmw_fastrtps_cpp":
        return
    try:
        from ament_index_python.packages import get_package_share_directory
        os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = os.path.join(
            get_package_share_directory("rb3_stereo"), "config", "fastdds.xml")
    except Exception:
        pass


def image_to_array(msg):
    """sensor_msgs/Image (mono8, bgr8, rgb8) -> numpy array (HxW or HxWx3)."""
    ch = {"mono8": 1, "8UC1": 1, "bgr8": 3, "rgb8": 3, "8UC3": 3}[msg.encoding]
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width * ch]
    a = a.reshape(msg.height, msg.width, ch) if ch == 3 else a
    return a[..., ::-1] if msg.encoding == "rgb8" else a


class StereoSubscriber:
    def __init__(self, namespace="/stereo", bgr=True, node=None, domain_id=None, slop=0.002):
        """bgr: return 3-channel BGR like the simulation (mono8 is expanded), else as published.
        slop: max stamp difference (s) for left/right to count as a pair (the camera node publishes
        identical stamps)."""
        ns = namespace.rstrip("/")
        self.bgr, self.slop = bgr, slop
        self._own = node is None
        if self._own:
            if not rclpy.ok():
                _use_large_shm_profile()
                rclpy.init(domain_id=domain_id)
            node = rclpy.create_node("stereo_subscriber")
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(node)
            self._spinning = True
            self._spin_thread = threading.Thread(target=self._spin, daemon=True)
            self._spin_thread.start()
        self.node = node
        self._cv = threading.Condition()
        self._img = {"left": [], "right": []}
        self.info = {}
        self._pair, self._last_returned, self.stamp = None, None, None
        for side in ("left", "right"):
            node.create_subscription(Image, f"{ns}/{side}/image_raw",
                                     lambda m, s=side: self._on_image(s, m), qos_profile_sensor_data)
            node.create_subscription(CameraInfo, f"{ns}/{side}/camera_info",
                                     lambda m, s=side: self._on_info(s, m), qos_profile_sensor_data)

    def _spin(self):
        while self._spinning and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.1)

    def _on_info(self, side, msg):
        with self._cv:
            self.info[side] = msg
            self._cv.notify_all()

    def _on_image(self, side, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self._cv:
            buf = self._img[side]
            buf.append((t, msg))
            del buf[:-4]
            other = self._img["right" if side == "left" else "left"]
            match = min(other, key=lambda x: abs(x[0] - t), default=None)
            if match and abs(match[0] - t) <= self.slop:
                left, right = (msg, match[1]) if side == "left" else (match[1], msg)
                self._pair = (t, left, right)
                self._cv.notify_all()

    def get_two_frames(self, timeout=1.0):
        """Newest stereo pair not returned before: [left, right]. Blocks up to `timeout` s; None on timeout."""
        with self._cv:
            if not self._cv.wait_for(lambda: self._pair is not None and self._pair[0] != self._last_returned,
                                     timeout):
                return None
            t, lmsg, rmsg = self._pair
            self._last_returned = self.stamp = t
        frames = [image_to_array(lmsg), image_to_array(rmsg)]
        if self.bgr:
            frames = [np.repeat(f[..., None], 3, axis=2) if f.ndim == 2 else f for f in frames]
        return frames

    # ---- same attributes the simulation passes to the vision code ----

    def _wait_info(self, timeout=5.0):
        with self._cv:
            self._cv.wait_for(lambda: "left" in self.info and "right" in self.info, timeout)
        if "left" not in self.info:
            raise TimeoutError("no camera_info received")

    @property
    def res(self):
        """(width, height) - Genesis camera convention."""
        self._wait_info()
        return self.info["left"].width, self.info["left"].height

    @property
    def fov(self):
        """Vertical field of view in degrees (Genesis convention), from the left camera_info."""
        self._wait_info()
        return camera_info.vfov_deg(self.info["left"])

    @property
    def camera_saperation(self):
        """Stereo baseline in m (spelling as in the simulation)."""
        self._wait_info()
        return camera_info.baseline_m(self.info["right"])

    camera_separation = camera_saperation

    def close(self):
        if self._own and self._spinning:
            self._spinning = False
            self._spin_thread.join(timeout=1.0)
            self._executor.shutdown()
            self.node.destroy_node()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
