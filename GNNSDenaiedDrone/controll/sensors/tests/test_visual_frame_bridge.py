import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from communication.VisualFrameBridge import VisualFrameBridge, render_visual_frame
from sensors.visual import Frame
from sensors.visual.visualDepthEst import SparseStereoDepth


class VisualFrameBridgeTests(unittest.TestCase):
    def test_renders_track_triangles_and_stereo_points_side_by_side(self):
        gray = np.zeros((20, 24), dtype=np.uint8)
        points = np.array([[5, 5], [15, 5], [5, 15]], dtype=np.float32)
        tracks = SimpleNamespace(
            new_points=points,
            triangles=np.array([[0, 1, 2]], dtype=int),
        )
        depth = SimpleNamespace(
            right_points=points + np.array([2, 0], dtype=np.float32),
            valid=np.ones(3, dtype=bool),
        )
        frame = Frame(
            gray_frames=[gray, gray.copy()],
            left_timestamp=1.0,
            right_timestamp=1.0,
        )

        rendered = render_visual_frame(frame, {0: tracks}, [depth])

        self.assertEqual(rendered.shape, (20, 48, 3))
        np.testing.assert_array_equal(rendered[5, 5], [255, 0, 0])
        np.testing.assert_array_equal(rendered[5, 24 + 7], [255, 0, 0])

    def test_renders_right_matches_even_when_depth_is_invalid(self):
        gray = np.zeros((20, 24), dtype=np.uint8)
        points = np.array([[5, 5]], dtype=np.float32)
        tracks = SimpleNamespace(
            new_points=points,
            triangles=np.empty((0, 3), dtype=int),
        )
        depth = SimpleNamespace(
            right_points=np.array([[8, 5]], dtype=np.float32),
            valid=np.array([False]),
        )
        frame = Frame(gray_frames=[gray, gray.copy()])

        rendered = render_visual_frame(frame, {0: tracks}, [depth])

        np.testing.assert_array_equal(rendered[5, 24 + 8], [255, 0, 0])

    def test_stereo_estimator_keeps_matched_coordinates_for_display(self):
        estimator = SparseStereoDepth()
        result = estimator._empty_result(1)

        estimator._store_depths(
            result,
            np.array([[10, 10]], dtype=np.float32),
            np.array([1.0, 0.0]),
            np.array([0]),
            np.array([[10.1, 10]], dtype=np.float32),
            fB=10.0,
            pixel_scale=1.0,
        )

        np.testing.assert_allclose(result.right_points[0], [10.1, 10])
        self.assertFalse(result.valid[0])

    def test_publishes_visualization_with_frame_timestamp(self):
        class CompressedImage:
            def __init__(self):
                self.header = SimpleNamespace(
                    frame_id="",
                    stamp=SimpleNamespace(sec=0, nanosec=0),
                )

        bridge = VisualFrameBridge.__new__(VisualFrameBridge)
        bridge._message_type = CompressedImage
        bridge._publisher = Mock()
        gray = np.zeros((12, 16), dtype=np.uint8)
        frame = Frame(
            gray_frames=[gray, gray.copy()],
            left_timestamp=2.25,
            right_timestamp=2.25,
        )

        rendered = bridge.publish(frame, [], [])

        message = bridge._publisher.publish.call_args.args[0]
        self.assertEqual(message.format, "jpeg")
        self.assertTrue(message.data.startswith(b"\xff\xd8"))
        self.assertEqual(message.header.frame_id, "stereo_visual_return")
        self.assertEqual(message.header.stamp.sec, 2)
        self.assertEqual(message.header.stamp.nanosec, 250_000_000)
        self.assertEqual(rendered.shape, (12, 32, 3))


if __name__ == "__main__":
    unittest.main()
