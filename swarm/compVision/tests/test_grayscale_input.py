"""Native monochrome input bypass and compatibility with the stereo pipeline."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from swarm.compVision.estimatorTypes import StereoFrame
from swarm.compVision.frame import FrameProcessor
from swarm.compVision.tests.estimator_factory import make_estimator


class GrayscaleInputTests(unittest.TestCase):
    def setUp(self):
        self.gray = np.random.default_rng(13).integers(0, 256, (49, 65), dtype=np.uint8)

    def pair(self, **kwargs):
        return StereoFrame(self.gray, self.gray.copy(), 1., 1., 7,
                           pixel_format='GRAY', **kwargs)

    def test_native_gray_skips_color_roundtrip_and_filters(self):
        processor = FrameProcessor(
            image_filters=('clahe', 'gaussian', 'median'), filter_kernel_size=3,
            pyramid_levels=3, convert_to_grayscale=True, filters_enabled=True,
        )
        original = self.gray.copy()
        with patch('swarm.compVision.frame.cv2.cvtColor', wraps=cv2.cvtColor) as convert:
            prepared = processor.process(self.pair(skip_preprocessing=True))
        self.assertEqual([call.args[1] for call in convert.call_args_list],
                         [cv2.COLOR_GRAY2BGR, cv2.COLOR_GRAY2BGR])
        np.testing.assert_array_equal(prepared.gray_frames[0], original)
        expected = original
        for level in prepared.pyramids[0]:
            np.testing.assert_array_equal(level, expected)
            expected = cv2.pyrDown(expected)
        self.assertEqual((prepared.idx, prepared.timestamp), (7, 1.))
        prepared.frames[0][:] = 0
        np.testing.assert_array_equal(prepared.gray_frames[0], original)
        prepared.gray_frames[0][:] = 0
        np.testing.assert_array_equal(self.gray, original)

    def test_optional_filters_operate_directly_on_gray(self):
        processor = FrameProcessor(
            image_filters=('clahe', 'gaussian', 'median'), filter_kernel_size=3,
            pyramid_levels=3, convert_to_grayscale=True, filters_enabled=True,
        )
        expected = cv2.medianBlur(cv2.GaussianBlur(processor.clahe.apply(self.gray), (3, 3), 0), 3)
        prepared = processor.process(self.pair())
        np.testing.assert_array_equal(prepared.gray_frames[0], expected)
        self.assertEqual(prepared.frames[0].shape, (49, 65, 3))

    def test_explicit_format_rejects_mismatched_shapes_and_bit_depth(self):
        color = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
        for left, right, fmt in ((self.gray, self.gray, 'BGR'),
                                 (color, color, 'GRAY'),
                                 (self.gray, color, 'GRAY'),
                                 (self.gray[..., None], self.gray[..., None], 'GRAY'),
                                 (self.gray.astype(np.uint16), self.gray.astype(np.uint16), 'GRAY'),
                                 (self.gray, self.gray, 'RGB')):
            with self.subTest(fmt=fmt, shape=left.shape), self.assertRaises(ValueError):
                StereoFrame(left, right, 1., 1., 0, pixel_format=fmt)
        with self.assertRaises(ValueError):
            self.pair(skip_preprocessing='yes')

    def test_bypass_is_per_frame_and_does_not_change_filter_configuration(self):
        processor = FrameProcessor(
            image_filters=('clahe',), filter_kernel_size=3, pyramid_levels=3,
            convert_to_grayscale=True, filters_enabled=True,
        )
        np.testing.assert_array_equal(processor.process(self.pair(skip_preprocessing=True)).gray_frames[0],
                                      self.gray)
        np.testing.assert_array_equal(processor.process(self.pair()).gray_frames[0],
                                      processor.clahe.apply(self.gray))
        self.assertEqual(processor.image_filters, ('clahe',))

    def test_native_gray_produces_same_tracking_depth_and_velocity_as_bgr(self):
        size = (321, 241)
        texture = np.random.default_rng(4).integers(0, 256, (61, 81), dtype=np.uint8)
        image = cv2.resize(texture, size, interpolation=cv2.INTER_NEAREST)
        mono, color = (
            make_estimator(size, 60, image_filters=filters)
            for filters in (('clahe',), ())
        )
        for estimator in (mono, color):
            estimator.drone_pos[2] = 10
            estimator.disparity_direction = np.array([0., 1.])
            estimator.feature_min_distance = 15
        for index in range(3):
            def shift(x, y):
                return cv2.warpAffine(image, np.float32([[1, 0, x], [0, 1, y]]), size,
                                      borderMode=cv2.BORDER_REFLECT)
            left, right = shift(2 * index, 0), shift(2 * index, 4)
            time = 1. + index * .125
            mono.processing(StereoFrame(left, right, time, time, index,
                                        pixel_format='GRAY', skip_preprocessing=True))
            color.processing(StereoFrame(cv2.cvtColor(left, cv2.COLOR_GRAY2BGR),
                                         cv2.cvtColor(right, cv2.COLOR_GRAY2BGR),
                                         time, time, index))
            np.testing.assert_allclose(mono.camera_velocity, color.camera_velocity)
            np.testing.assert_allclose(mono.depth_points, color.depth_points)
            np.testing.assert_allclose(mono.velocity_flows, color.velocity_flows)
        self.assertGreater(len(mono.depth_points), 5)


if __name__ == '__main__':
    unittest.main()
