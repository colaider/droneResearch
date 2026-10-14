import unittest
from threading import Lock
from types import SimpleNamespace

import numpy as np

from communication.CameraStreamReceiver import CameraStreamReceiver


def image_message(image, stamp_ns, *, step=None, encoding="mono8"):
    height, width = image.shape
    row_step = width if step is None else step
    rows = np.zeros((height, row_step), dtype=np.uint8)
    rows[:, :width] = image
    return SimpleNamespace(
        encoding=encoding,
        height=height,
        width=width,
        step=row_step,
        data=rows.reshape(-1),
        header=SimpleNamespace(stamp=SimpleNamespace(
            sec=stamp_ns // 1_000_000_000,
            nanosec=stamp_ns % 1_000_000_000,
        )),
    )


class CameraStreamReceiverTests(unittest.TestCase):
    def setUp(self):
        self.receiver = CameraStreamReceiver.__new__(CameraStreamReceiver)
        self.receiver.node = SimpleNamespace(
            get_logger=lambda: SimpleNamespace(error=self.fail)
        )
        self.receiver._lock = Lock()
        self.receiver._left_by_stamp = {}
        self.receiver._right_by_stamp = {}
        self.receiver._latest_frame = None
        self.receiver._sequence_id = 0
        self.receiver._max_pending = 8

    def test_pairs_images_by_timestamp_and_returns_pair_once(self):
        left = np.arange(12, dtype=np.uint8).reshape(3, 4)
        right = left + 20
        stamp = 1_234_000_000

        self.receiver._on_right(image_message(right, stamp))
        self.assertIsNone(self.receiver.get_frame())
        self.receiver._on_left(image_message(left, stamp))

        frame = self.receiver.get_frame()
        self.assertEqual(frame.sequence_id, 0)
        self.assertEqual(frame.left_timestamp, 1.234)
        np.testing.assert_array_equal(frame.left, left)
        np.testing.assert_array_equal(frame.right, right)
        self.assertIsNone(self.receiver.get_frame())

    def test_decodes_padded_rows_and_rejects_non_mono8(self):
        image = np.arange(6, dtype=np.uint8).reshape(2, 3)
        decoded = CameraStreamReceiver._decode_mono8(image_message(image, 0, step=5))
        np.testing.assert_array_equal(decoded, image)

        with self.assertRaisesRegex(ValueError, "Expected mono8"):
            CameraStreamReceiver._decode_mono8(image_message(image, 0, encoding="bgr8"))


if __name__ == "__main__":
    unittest.main()
