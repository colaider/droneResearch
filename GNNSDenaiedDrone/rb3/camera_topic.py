"""Camera topic: the OV9282 stereo pair, published as left/right mono8 images + camera_info.

Reuses rb3_stereo's StereoCamera driver and nominal camera model. A background thread reads
pairs and publishes them on <ns>/left|right/image_raw and <ns>/left|right/camera_info.
"""
import array
import threading
import time

import cv2
import numpy as np
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

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

        # Side-by-side JPEG preview, so the drone_gui camera panel can show what the cameras see.
        self.pub_preview = node.create_publisher(CompressedImage, f"{ns}/preview/compressed", qos_profile_sensor_data)
        # Separate topic for the PROCESSED image (your vision output). Left untouched by preview.
        self.pub_processed = node.create_publisher(CompressedImage, f"{ns}/processed/compressed",
                                                   qos_profile_sensor_data)
        self.preview_fps = 10.0
        self.preview_scale = 0.5
        self._t_prev = 0.0

        self.cam = StereoCamera(w, h, fps, calibration="").start()   # raw (unrectified) images
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="rb3-camera")
        self._thread.start()

    def _image(self, frame, stamp, frame_id):
        msg = Image(height=frame.shape[0], width=frame.shape[1], encoding="mono8", is_bigendian=0, step=frame.shape[1])
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

            now = time.monotonic()
            if self.preview_fps > 0 and now - self._t_prev >= 1.0 / self.preview_fps:
                self._t_prev = now
                # plain camera preview (unchanged) ...
                preview = np.hstack((pair.left, pair.right))
                if self.preview_scale != 1.0:
                    preview = cv2.resize(preview, None, fx=self.preview_scale, fy=self.preview_scale,
                                         interpolation=cv2.INTER_AREA)
                self._publish_jpeg(self.pub_preview, preview, stamp, "stereo_preview")
                # ... and your processed image on its own topic
                processed = self.process(pair.left, pair.right)
                if processed is not None:
                    self._publish_jpeg(self.pub_processed, processed, stamp, "stereo_processed")

    def process(self, left, right):
        """Return the image to publish on <ns>/processed/compressed, or None to skip.

        Replace this with your real processing (runs on the full-res raw frames). For now it
        just returns the left frame so the topic has something to show.
        """
        return left

    def _publish_jpeg(self, pub, img, stamp, frame_id):
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            msg = CompressedImage(format="jpeg", data=jpg.tobytes())
            msg.header.stamp, msg.header.frame_id = stamp, frame_id
            pub.publish(msg)

    def stop(self):
        self._stop = True
        self._thread.join(timeout=2.0)
        self.cam.stop()
