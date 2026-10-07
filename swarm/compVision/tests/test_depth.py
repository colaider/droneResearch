"""Geometry, failure handling and actual same-frame stereo correspondence."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from swarm.compVision.depth import DepthEstimator


class DepthEstimatorTests(unittest.TestCase):
    def test_signed_horizontal_disparity_and_valid_depth_only_summary(self):
        left = np.array([[50, 40], [90, 50], [80, 40]], np.float32)
        right = left + np.array([[-5, 0], [-10, 0], [5, 0]], np.float32)
        original_left, original_right = left.copy(), right.copy()
        result = DepthEstimator().triangulate(
            left, right, [True, True, True], 400, 0.125, [-2, 0], 20,
            frame_idx=7,
        )
        np.testing.assert_array_equal(result.valid, [True, True, False])
        np.testing.assert_allclose(result.depths, [10, 5, 7.5])
        np.testing.assert_allclose(result.points, [[50, 40, 10], [90, 50, 5]])
        self.assertEqual(result.median, 7.5)
        self.assertAlmostEqual(result.valid_fraction, 2 / 3)
        self.assertEqual(result.frame_idx, 7)
        self.assertEqual(result.left_points.dtype, np.float32)
        self.assertEqual(result.right_points.dtype, np.float32)
        self.assertTrue(np.isnan(result.right_points[2]).all())
        np.testing.assert_array_equal(left, original_left)
        np.testing.assert_array_equal(right, original_right)

    def test_vertical_disparity_and_geometric_rejections(self):
        left = np.tile([40, 30], (6, 1)).astype(np.float32)
        right = left + np.array(
            [[0, 5], [1.1, 5], [0, -5], [0, 0.2], [0, 5], [0, np.nan]],
            np.float32,
        )
        result = DepthEstimator().triangulate(
            left, right, [True, True, True, True, False, True],
            400, 0.125, [0, 3], 99,
        )
        np.testing.assert_array_equal(result.valid, [True, False, False, False, False, False])
        np.testing.assert_allclose(result.depths, 10)
        self.assertEqual(result.median, 10)

    def test_no_valid_matches_use_height_without_claiming_measurement(self):
        for fallback, expected in ((12, 12), (0, np.nan), (-1, np.nan), (np.inf, np.nan)):
            with self.subTest(fallback=fallback):
                result = DepthEstimator().triangulate(
                    [[1, 1]], [[1, 1]], [True], 400, 0.125, [1, 0], fallback
                )
                np.testing.assert_allclose(result.depths, [expected], equal_nan=True)
                self.assertTrue(np.isnan(result.median))
                self.assertEqual(result.valid_fraction, 0)
                self.assertEqual(result.points.shape, (0, 3))

    def test_empty_input_never_calls_lk(self):
        image = np.zeros((50, 50), np.uint8)
        with patch("swarm.compVision.depth.cv2.calcOpticalFlowPyrLK") as lk:
            result = DepthEstimator().estimate(image, image, [], 400, 0.125, [1, 0], 12)
        lk.assert_not_called()
        self.assertEqual(result.left_points.shape, (0, 2))
        self.assertEqual(result.right_points.shape, (0, 2))
        self.assertEqual(result.depths.shape, (0,))
        self.assertEqual(result.valid_fraction, 0)

    def test_failed_lk_returns_fallback_safely(self):
        image = np.zeros((50, 50), np.uint8)
        for side_effect in (
            [(None, None, None)],
            [(np.array([[[24, 20]]], np.float32), np.ones((1, 1), np.uint8), None),
             (None, None, None)],
            cv2.error("LK failed"),
        ):
            with self.subTest(failure=side_effect):
                with patch("swarm.compVision.depth.cv2.calcOpticalFlowPyrLK", side_effect=side_effect):
                    result = DepthEstimator().estimate(
                        image, image, [[20, 20]], 400, 0.125, [1, 0], 12
                    )
                np.testing.assert_allclose(result.depths, [12])
                self.assertFalse(result.valid.any())
                self.assertTrue(np.isnan(result.right_points).all())

    def test_invalid_input_and_match_checks_preserve_original_rows(self):
        image = np.zeros((100, 100), np.uint8)
        points = np.array([[20, 20], [np.nan, 10], [-1, 20], [40, 40], [60, 60]], np.float32)
        forward = np.array([[[24, 20]], [[200, 40]], [[64, 60]]], np.float32)
        backward = np.array([[[20, 20]], [[63, 60]]], np.float32)
        responses = [
            (forward, np.ones((3, 1), np.uint8), None),
            (backward, np.ones((2, 1), np.uint8), None),
        ]
        with patch("swarm.compVision.depth.cv2.calcOpticalFlowPyrLK", side_effect=responses) as lk:
            result = DepthEstimator().estimate(image, image, points, 400, .125, [1, 0], 12)
        self.assertEqual(lk.call_args_list[0].args[2].shape, (3, 1, 2))
        self.assertEqual(lk.call_args_list[1].args[2].shape, (2, 1, 2))
        np.testing.assert_array_equal(result.valid, [True, False, False, False, False])
        np.testing.assert_allclose(result.depths, 12.5)
        self.assertTrue(np.isnan(result.right_points[1:]).all())

    def test_actual_lk_recovers_known_stereo_shift(self):
        rng = np.random.default_rng(20261006)
        left = rng.integers(0, 256, (480, 640), dtype=np.uint8)
        left = cv2.GaussianBlur(left, (5, 5), 0.8)
        right = cv2.warpAffine(left, np.float32([[1, 0, -5], [0, 1, 0]]), (640, 480))
        # Keep pyramid patches clear of the boundary changed by image translation.
        points = np.array([[x, y] for y in (160, 240, 320) for x in (200, 280, 360, 440)], np.float32)
        original = points.copy()
        result = DepthEstimator().estimate(
            left, right, points, 400, 0.125, [-1, 0], 99, frame_idx=8
        )
        self.assertGreaterEqual(result.valid_fraction, 0.9)
        np.testing.assert_allclose(result.depths[result.valid], 10, rtol=0.015)
        np.testing.assert_allclose(
            result.right_points[result.valid], points[result.valid] + [-5, 0], atol=0.08
        )
        np.testing.assert_array_equal(points, original)

    def test_bad_calibration_and_images_are_explicit_errors(self):
        estimator = DepthEstimator()
        for focal, baseline, direction in ((0, .1, [1, 0]), (400, -.1, [1, 0]),
                                            (400, .1, [0, 0]), (400, .1, [np.nan, 0])):
            with self.subTest(calibration=(focal, baseline, direction)):
                with self.assertRaises(ValueError):
                    estimator.triangulate([], [], [], focal, baseline, direction, 12)
        with self.assertRaises(ValueError):
            estimator.estimate(np.zeros((20, 20)), np.zeros((20, 20)), [], 400, .1, [1, 0], 12)


if __name__ == "__main__":
    unittest.main()
