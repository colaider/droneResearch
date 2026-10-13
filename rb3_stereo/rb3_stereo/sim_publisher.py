"""Publish simulated stereo frames on exactly the topics, message types, QoS and stamp convention of
the real camera node, so consumers (StereoSubscriber, rviz, rosbag, ...) cannot tell the difference.

In the Genesis simulation (swarm/droneClasses/droneStruct.py), after the frames are rendered:

    from rb3_stereo.sim_publisher import StereoPublisher
    pub = StereoPublisher(res=(500, 600), fov_deg=85, baseline_m=0.05)    # values of customScene.py
    left, right = drone.get_two_frames()                                  # BGR
    pub.publish(left, right, stamp_s=sim_time)                            # sim_time optional
"""
import array
import threading

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image

from . import camera_info


class StereoPublisher:
    def __init__(self, res, fov_deg, baseline_m, namespace="/stereo", mono=True, node=None, domain_id=None,
                 frame_ids=("stereo_left_optical_frame", "stereo_right_optical_frame")):
        """res = (width, height) and fov_deg = vertical FOV, as in the Genesis camera options.
        mono: publish mono8 like the OV9282 (else bgr8)."""
        ns = namespace.rstrip("/")
        self.mono, self.frame_ids = mono, frame_ids
        if node is None:
            if not rclpy.ok():
                rclpy.init(domain_id=domain_id)
            node = rclpy.create_node("stereo_sim_publisher")
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(node)
            self._spinning = True
            self._spin_thread = threading.Thread(target=self._spin, daemon=True)
            self._spin_thread.start()
        self.node = node
        w, h = res
        self.info = [camera_info.nominal(w, h, fov_deg, baseline_m, right=r) for r in (False, True)]
        self.pub = [(node.create_publisher(Image, f"{ns}/{s}/image_raw", qos_profile_sensor_data),
                     node.create_publisher(CameraInfo, f"{ns}/{s}/camera_info", qos_profile_sensor_data))
                    for s in ("left", "right")]

    def _spin(self):
        while self._spinning and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.1)

    def close(self):
        if getattr(self, "_spinning", False):
            self._spinning = False
            self._spin_thread.join(timeout=1.0)
            self._executor.shutdown()
            self.node.destroy_node()

    def publish(self, left, right, stamp_s=None):
        """left/right: BGR or mono uint8 images. stamp_s: time of the frames (s), default now."""
        stamp = (self.node.get_clock().now() if stamp_s is None else Time(seconds=stamp_s)).to_msg()
        for i, frame in enumerate((left, right)):
            if self.mono and frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.ascontiguousarray(frame)
            ch = 1 if frame.ndim == 2 else 3
            msg = Image(height=frame.shape[0], width=frame.shape[1], encoding="mono8" if ch == 1 else "bgr8",
                        is_bigendian=0, step=frame.shape[1] * ch)
            msg.header.stamp, msg.header.frame_id = stamp, self.frame_ids[i]
            data = array.array("B")
            data.frombytes(frame)
            msg.data = data
            self.info[i].header = msg.header
            self.pub[i][0].publish(msg)
            self.pub[i][1].publish(self.info[i])
