"""Camera topic: the OV9282 stereo pair, published as left/right mono8 images + camera_info.

Reuses rb3_stereo's StereoCamera driver and nominal camera model. A background thread reads
pairs and publishes them on <ns>/left|right/image_raw and <ns>/left|right/camera_info.
"""
import array
import threading

import numpy as np
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from rb3_stereo import camera_info
from rb3_stereo.stereo import StereoCamera


class CameraTopic:
    def __init__(self, node, ns="/stereo", resolution=(640, 400), fps=30,
                 vfov_deg=85.0, baseline_m=0.05):
        self.node = node
        w, h = resolution
        self.pub = {
            side: (node.create_publisher(Image, f"{ns}/{side}/image_raw", qos_profile_sensor_data),
                   node.create_publisher(camera_info.CameraInfo, f"{ns}/{side}/camera_info", qos_profile_sensor_data))
            for side in ("left", "right")}
        self.info = {side: camera_info.nominal(w, h, vfov_deg, baseline_m, right=(side == "right"))
                     for side in ("left", "right")}
        for side in ("left", "right"):
            self.info[side].header.frame_id = f"stereo_{side}_optical_frame"

        self.cam = StereoCamera(w, h, fps, calibration="").start()   # raw (unrectified) images
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="rb3-camera")
        self._thread.start()

    def _image(self, frame, stamp, frame_id):
        msg = Image(height=frame.shape[0], width=frame.shape[1], encoding="mono8",
                    is_bigendian=0, step=frame.shape[1])
        msg.header.stamp, msg.header.frame_id = stamp, frame_id
        data = array.array("B")
        data.frombytes(np.ascontiguousarray(frame))
        msg.data = data
        return msg

    def _loop(self):
        while not self._stop:
            pair = self.cam.read_pair(timeout=1.0)
            if pair is None:
                continue
            stamp = self.node.get_clock().now().to_msg()
            for side, frame in (("left", pair.left), ("right", pair.right)):
                img_pub, info_pub = self.pub[side]
                img_pub.publish(self._image(frame, stamp, f"stereo_{side}_optical_frame"))
                self.info[side].header.stamp = stamp
                info_pub.publish(self.info[side])

    def stop(self):
        self._stop = True
        self._thread.join(timeout=2.0)
        self.cam.stop()
