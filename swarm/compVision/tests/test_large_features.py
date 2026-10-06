"""Contract and image-geometry tests for coarse structural proposals."""

import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from swarm.compVision.largeFeatures import LargeFeatureDetector, LargeFeatures


def make_pyramid(gray, levels=2):
    pyramid = [gray]
    for _ in range(levels):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


class LargeFeatureTests(unittest.TestCase):
    def test_defaults_are_independent(self):
        first, second = LargeFeatures(), LargeFeatures()
        first.regions.append((1, 2, 3, 4))
        self.assertEqual(second.regions, [])
        self.assertEqual(second.lines.shape, (0, 4))
        self.assertEqual(second.lines.dtype, np.float32)

    def test_rectangle_produces_structural_region(self):
        gray = np.full((240, 320), 30, np.uint8)
        cv2.rectangle(gray, (80, 60), (240, 180), 220, -1)
        result = LargeFeatureDetector().detect(make_pyramid(gray))
        self.assertGreaterEqual(len(result.lines), 4)
        # The four edges should form a connected, padded proposal containing
        # the rectangle centre; exact LSD endpoints may vary across versions.
        self.assertTrue(any(x0 <= 160 < x1 and y0 <= 120 < y1
                            for x0, y0, x1, y1 in result.regions))
        self.assertEqual(result.lines.dtype, np.float32)
        self.assertTrue(np.isfinite(result.lines).all())

    def test_blank_and_tiny_inputs_are_empty(self):
        detector = LargeFeatureDetector()
        for pyramid in ([], [np.empty((0, 0), np.uint8)],
                        make_pyramid(np.zeros((240, 320), np.uint8)),
                        make_pyramid(np.arange(64, dtype=np.uint8).reshape(8, 8))):
            with self.subTest(shapes=[image.shape for image in pyramid]):
                result = detector.detect(pyramid)
                self.assertEqual(result.regions, [])
                self.assertEqual(result.lines.shape, (0, 4))

    def test_odd_sizes_use_power_of_two_mapping_and_clip_outputs(self):
        gray = np.zeros((241, 321), np.uint8)
        cv2.rectangle(gray, (20, 20), (300, 220), 255, -1)
        detector = LargeFeatureDetector(min_length_px=1, pad_px=100)
        detector._detector = Mock()
        # Controlled detector output isolates the coordinate contract from
        # implementation-dependent subpixel LSD positions.
        detector._detector.detect.return_value = (
            np.array([[[1.25, 2.5, 30.5, 2.5]],
                      [[-1, 30, 90, 30]],
                      [[np.nan, 1, 20, 1]]], dtype=np.float32),
            None, None, None,
        )
        result = detector.detect(make_pyramid(gray))
        np.testing.assert_allclose(result.lines[0], [5, 10, 122, 10])
        np.testing.assert_allclose(result.lines[1], [0, 120, 320, 120])
        self.assertEqual(len(result.lines), 2)
        for x0, y0, x1, y1 in result.regions:
            self.assertTrue(0 <= x0 < x1 <= 321)
            self.assertTrue(0 <= y0 < y1 <= 241)

    def test_shorter_pyramid_uses_available_level(self):
        gray = np.zeros((80, 100), np.uint8)
        cv2.rectangle(gray, (20, 20), (80, 60), 255, -1)
        detector = LargeFeatureDetector()
        detector._detector = Mock()
        detector._detector.detect.return_value = (
            np.array([[[5, 10, 25, 10]]], dtype=np.float32),
            None, None, None,
        )
        result = detector.detect(make_pyramid(gray, levels=1))
        np.testing.assert_allclose(result.lines[0], [10, 20, 50, 20])

    def test_region_count_can_be_limited(self):
        gray = np.zeros((320, 480), np.uint8)
        for start, end in [((30, 30), (140, 110)),
                           ((300, 200), (430, 280))]:
            cv2.rectangle(gray, start, end, 255, -1)
        result = LargeFeatureDetector(max_regions=1).detect(make_pyramid(gray))
        self.assertEqual(len(result.regions), 1)
        lines_only = LargeFeatureDetector(max_regions=0).detect(make_pyramid(gray))
        self.assertEqual(lines_only.regions, [])
        self.assertGreater(len(lines_only.lines), 0)

    def test_invalid_pyramid_rejected(self):
        detector = LargeFeatureDetector()
        with self.assertRaises(ValueError):
            detector.detect([np.zeros((40, 40, 3), np.uint8)])
        with self.assertRaises(ValueError):
            detector.detect([np.zeros((40, 40), np.float32)])
        with self.assertRaises(ValueError):
            detector.detect([np.zeros((40, 40), np.uint8),
                             np.zeros((13, 13), np.uint8)])


if __name__ == "__main__":
    unittest.main()
