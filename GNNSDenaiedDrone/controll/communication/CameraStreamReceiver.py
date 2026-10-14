"""Receive and synchronize the RB3 stereo camera image topics."""

from threading import Lock

import numpy as np

from sensors.visual import StereoFrame


class CameraStreamReceiver:
    """Pair timestamp-matched mono8 images and expose them as StereoFrame objects.

    The RB3 camera publisher stamps both images from a stereo capture with the
    same ROS timestamp. Only the latest completed pair is retained; callers
    consume it with ``get_frame()``.
    """

    def __init__(self, node, ns="/stereo"):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        self.node = node
        self._lock = Lock()
        self._left_by_stamp = {}
        self._right_by_stamp = {}
        self._latest_frame = None
        self._sequence_id = 0
        self._max_pending = 8
        self._subscriptions = (
            node.create_subscription(
                Image, f"{ns}/left/image_raw", self._on_left, qos_profile_sensor_data
            ),
            node.create_subscription(
                Image, f"{ns}/right/image_raw", self._on_right, qos_profile_sensor_data
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

        data = np.asarray(message.data, dtype=np.uint8).reshape(-1)
        required = height * step
        if data.size < required:
            raise ValueError("Camera image data is shorter than its declared dimensions")
        return data[:required].reshape(height, step)[:, :width].copy()

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
            pending[stamp_ns] = image
            if stamp_ns in other:
                left = image if side == "left" else other.pop(stamp_ns)
                right = image if side == "right" else other.pop(stamp_ns)
                timestamp = stamp_ns / 1_000_000_000
                self._latest_frame = StereoFrame(
                    left, right, timestamp, timestamp, self._sequence_id
                )
                self._sequence_id += 1
                pending.pop(stamp_ns, None)

            while len(pending) > self._max_pending:
                del pending[min(pending)]

    def get_frame(self):
        """Return the newest completed pair once, or None if no pair is ready."""
        with self._lock:
            frame = self._latest_frame
            self._latest_frame = None
        return frame

    def close(self):
        """Unregister the subscriptions without destroying the caller's node."""
        for subscription in self._subscriptions:
            self.node.destroy_subscription(subscription)
