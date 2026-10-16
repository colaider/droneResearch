"""Subscribe to the stereo_camera image pair and publish a processed compressed image."""
import threading
import time

import cv2
import numpy as np
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image


class CameraTopic:
    def __init__(self, node, ns="/stereo"):
        """Consume timestamp-matched raw images from the hardware-owning stereo_camera node."""
        self.node = node
        self.pub_processed = node.create_publisher(
            CompressedImage, f"{ns}/processed/compressed", qos_profile_sensor_data
        )
        self.processed_fps = 10.0
        self._t_prev = 0.0
        self._lock = threading.Lock()
        self._left_by_stamp = {}
        self._right_by_stamp = {}
        self._max_pending = 8
        namespace = "/" + ns.strip("/") if ns.strip("/") else ""
        self._subscriptions = (
            node.create_subscription(
                Image, f"{namespace}/left/image_raw", self._on_left, qos_profile_sensor_data
            ),
            node.create_subscription(
                Image, f"{namespace}/right/image_raw", self._on_right, qos_profile_sensor_data
            ),
        )

    @staticmethod
    def _stamp_ns(message):
        stamp = message.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @staticmethod
    def _decode_mono8(message):
        if message.encoding.lower() != "mono8":
            raise ValueError(f"Expected mono8 camera image, received {message.encoding!r}")
        height, width, step = int(message.height), int(message.width), int(message.step)
        if height <= 0 or width <= 0 or step < width:
            raise ValueError("Camera image dimensions or row step are invalid")
        try:
            data = np.frombuffer(message.data, dtype=np.uint8)
        except TypeError:
            data = np.asarray(message.data, dtype=np.uint8).reshape(-1)
        if data.size < height * step:
            raise ValueError("Camera image data is shorter than its declared dimensions")
        return data[:height * step].reshape(height, step)[:, :width].copy()

    def _on_left(self, message):
        self._on_image("left", message)

    def _on_right(self, message):
        self._on_image("right", message)

    def _on_image(self, side, message):
        try:
            image = self._decode_mono8(message)
            stamp_ns = self._stamp_ns(message)
        except (AttributeError, TypeError, ValueError) as error:
            self.node.get_logger().error(f"Discarding invalid {side} camera image: {error}")
            return

        with self._lock:
            pending = self._left_by_stamp if side == "left" else self._right_by_stamp
            other = self._right_by_stamp if side == "left" else self._left_by_stamp
            pending[stamp_ns] = (image, message.header.stamp)
            pair = None
            if stamp_ns in other:
                left = pending.pop(stamp_ns) if side == "left" else other.pop(stamp_ns)
                right = other.pop(stamp_ns) if side == "left" else pending.pop(stamp_ns)
                pair = (left[0], right[0], left[1])
            while len(pending) > self._max_pending:
                del pending[min(pending)]

        if pair is None:
            return
        now = time.monotonic()
        if self.processed_fps <= 0 or now - self._t_prev < 1.0 / self.processed_fps:
            return
        self._t_prev = now
        left, right, stamp = pair
        processed = self.process(left, right)
        if processed is not None:
            self._publish_jpeg(self.pub_processed, processed, stamp, "stereo_processed")

    def process(self, left, right):
        """Return the image for <ns>/processed/compressed, or None to skip publishing."""
        return left

    def _publish_jpeg(self, pub, img, stamp, frame_id):
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            msg = CompressedImage(format="jpeg", data=jpg.tobytes())
            msg.header.stamp, msg.header.frame_id = stamp, frame_id
            pub.publish(msg)

    def stop(self):
        for subscription in self._subscriptions:
            self.node.destroy_subscription(subscription)
