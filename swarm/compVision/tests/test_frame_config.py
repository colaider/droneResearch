"""Configuration switches reach the estimator and preserve native mono inputs."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from swarm.compVision.estimatorTypes import StereoFrame
from swarm.compVision.frame import FrameProcessor
from swarm.compVision.tests.estimator_factory import make_estimator
from swarm.config.cameraConfig import StereoCameraConfig


class FrameConfigTests(unittest.TestCase):
    def setUp(self):
        self.gray = np.random.default_rng(19).integers(0, 256, (49, 65), dtype=np.uint8)
        self.color = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)

    def test_estimator_requires_explicit_camera_and_processor_configuration(self):
        from swarm.compVision.visualEst import VisulaEst
        with self.assertRaises(TypeError):
            VisulaEst()

        estimator = make_estimator(
            (65, 49), 60, focal_px=42.4, baseline=.2,
            disparity_direction=(-1., 0.), image_filters=(),
            convert_to_grayscale=True, filters_enabled=True,
            filter_kernel_size=3,
        )
        self.assertEqual(estimator.resolution, (65, 49))
        self.assertEqual(estimator.foc_l, 42.4)
        self.assertEqual(estimator.camera_saperation, .2)
        np.testing.assert_allclose(
            estimator.camera_config.disparity_direction(), (-1., 0.)
        )

    def test_camera_offset_uses_simulation_mount_geometry_and_allows_override(self):
        from swarm.compVision.visualEst import VisulaEst

        options = dict(
            image_filters=(), convert_to_grayscale=True, filters_enabled=True,
            filter_kernel_size=3, max_stereo_skew_s=.005,
        )
        mounted = StereoCameraConfig(
            res=(65, 49), left_focal=42.4, right_focal=42.4,
            baseline=.2, mount_pos=(.1, 0., 0.),
        )
        estimator = VisulaEst(mounted, **options)
        np.testing.assert_allclose(
            estimator.camera_offset_from_yaw_center, [-.1, 0.]
        )

        mounted.camera_offset_from_yaw_center = (.02, -.01)
        overridden = VisulaEst(mounted, **options)
        np.testing.assert_allclose(
            overridden.camera_offset_from_yaw_center, [.02, -.01]
        )

    def test_image_processing_options_are_configured_at_estimator_creation(self):
        estimator = make_estimator(
            res=(65, 49),
            fov=60,
            focal_px=42.4,
            baseline=.2,
            disparity_direction=(-1., 0.),
            image_filters=('median',),
            convert_to_grayscale=False,
            filters_enabled=False,
            filter_kernel_size=5,
            max_stereo_skew_s=.005,
        )
        processor = estimator.frame_processor
        self.assertEqual(processor.image_filters, ('median',))
        self.assertFalse(processor.convert_to_grayscale)
        self.assertFalse(processor.filters_enabled)
        self.assertEqual(processor.filter_kernel_size, 5)

    def test_both_boolean_switches_and_filter_selection_reach_estimator(self):
        for convert in (False, True):
            for filters in (False, True):
                with self.subTest(convert=convert, filters=filters):
                    estimator = make_estimator(
                        (65, 49), 60,
                        focal_px=42.4,
                        baseline=.2,
                        disparity_direction=(-1., 0.),
                        image_filters=('gaussian', 'median'),
                        convert_to_grayscale=convert,
                        filters_enabled=filters,
                        filter_kernel_size=3,
                    )
                    image = self.color if convert else self.gray
                    pair = StereoFrame(image, image.copy(), 1., 1., 0)
                    self.assertEqual(pair.pixel_format, 'BGR' if convert else 'GRAY')
                    estimator.processing(pair)
                    expected = image.copy()
                    if filters:
                        expected = cv2.medianBlur(cv2.GaussianBlur(expected, (3, 3), 0), 3)
                    if convert:
                        expected = cv2.cvtColor(expected, cv2.COLOR_BGR2GRAY)
                    np.testing.assert_array_equal(estimator.buffer[-1].gray_frames[0], expected)
                    self.assertEqual(estimator.frame_processor.filters_enabled, filters)
                    self.assertEqual(estimator.frame_processor.convert_to_grayscale, convert)

    def test_disabled_conversion_rejects_bgr_before_advancing_estimator(self):
        estimator = make_estimator(
            (65, 49), 60, focal_px=42.4, baseline=.2,
            disparity_direction=(-1., 0.), image_filters=('clahe',),
            convert_to_grayscale=False, filters_enabled=True,
            filter_kernel_size=3,
        )
        with self.assertRaisesRegex(ValueError, 'requires native grayscale'):
            estimator.processing(StereoFrame(self.color, self.color, 1., 1., 0))
        self.assertEqual(len(estimator.buffer), 0)
        self.assertIsNone(estimator.current_depth)

    def test_gray_image_lists_work_with_explicit_processor_options(self):
        processor = FrameProcessor(
            image_filters=(), filter_kernel_size=3,
            convert_to_grayscale=False, filters_enabled=False,
        )
        prepared = processor.process([self.gray, self.gray], 3)
        np.testing.assert_array_equal(prepared.gray_frames[0], self.gray)
        self.assertEqual(prepared.idx, 3)

    def test_native_gray_never_gets_color_roundtrip_when_conversion_enabled(self):
        processor = FrameProcessor(
            image_filters=(), filter_kernel_size=3,
            convert_to_grayscale=True, filters_enabled=False,
        )
        with patch('swarm.compVision.frame.cv2.cvtColor', wraps=cv2.cvtColor) as convert:
            processor.process(StereoFrame(self.gray, self.gray, 1., 1., 0))
        self.assertEqual([call.args[1] for call in convert.call_args_list],
                         [cv2.COLOR_GRAY2BGR, cv2.COLOR_GRAY2BGR])

    def test_config_requires_actual_booleans(self):
        with self.assertRaises(ValueError):
            FrameProcessor(
                image_filters=(), filter_kernel_size=3,
                convert_to_grayscale='False', filters_enabled=True,
            )
        with self.assertRaises(ValueError):
            FrameProcessor(
                image_filters=(), filter_kernel_size=3,
                convert_to_grayscale=True, filters_enabled='False',
            )


if __name__ == '__main__':
    unittest.main()
