"""Contract and image-geometry tests for coarse structural proposals."""

import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from swarm.compVision.frame import FrameProcessor
from swarm.compVision.largeFeatures import (LargeFeatureDetector, LargeFeatures,
                                            draw_large_features, structure_points)


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
        # Coordinate mapping only: the 40x50 layer is smaller than the texture
        # window, so the smooth-side line filter is not meaningful here.
        detector = LargeFeatureDetector(filter_lines=False)
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



def grass(shape=(480, 640), seed=7):
    """Coarse random texture: high local contrast everywhere, like grass."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (shape[0] // 2, shape[1] // 2), dtype=np.uint8)
    return cv2.resize(cv2.GaussianBlur(small, (3, 3), 0.8), shape[::-1],
                      interpolation=cv2.INTER_LINEAR)


def grass_with_box(box=(200, 120, 440, 330)):
    gray = grass()
    x0, y0, x1, y1 = box
    gray[y0:y1, x0:x1] = 180 + gray[y0:y1, x0:x1] // 32   # nearly flat box top
    return gray


def frame_pyramid(gray, levels=4):
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    processor = FrameProcessor(image_filters=(), pyramid_levels=levels)
    return processor.process([bgr, bgr], 0).pyramids[0]


class StructureOnCoarseLayersTests(unittest.TestCase):
    BOX = (200, 120, 440, 330)

    def test_box_in_grass_gives_quad_lines_and_corners_per_level(self):
        pyramid = frame_pyramid(grass_with_box(self.BOX))
        result = LargeFeatureDetector().detect(pyramid, levels=(1, 2))
        self.assertEqual(sorted(result), [1, 2])
        x0, y0, x1, y1 = self.BOX
        box_corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
        for level, features in result.items():
            with self.subTest(level=level):
                tol = 4 * 2 ** level
                self.assertEqual(features.level, level)
                self.assertEqual(features.smooth_mask.shape, pyramid[level].shape)
                self.assertEqual(len(features.quads), 1)
                quad = features.quads[0]
                self.assertTrue(4 <= len(quad) <= 6)
                np.testing.assert_allclose(quad.min(axis=0), [x0, y0], atol=tol)
                np.testing.assert_allclose(quad.max(axis=0), [x1 - 1, y1 - 1], atol=tol)
                # Every structural line lies on the box outline, not in the grass.
                self.assertGreaterEqual(len(features.lines), 2)
                ends = features.lines.reshape(-1, 2)
                outline = np.minimum.reduce([np.abs(ends[:, 0] - x0), np.abs(ends[:, 0] - x1),
                                             np.abs(ends[:, 1] - y0), np.abs(ends[:, 1] - y1)])
                self.assertLess(outline.max(), tol)
                for corner in features.corners:
                    self.assertLess(np.linalg.norm(box_corners - corner, axis=1).min(), tol)

    def test_grass_only_has_no_structure(self):
        result = LargeFeatureDetector().detect(frame_pyramid(grass()), levels=(1, 2, 3))
        for level, features in result.items():
            with self.subTest(level=level):
                self.assertEqual(features.quads, [])
                self.assertLessEqual(len(features.lines), 1)
                self.assertEqual(len(features.corners), 0)

    def test_levels_missing_from_pyramid_are_skipped(self):
        pyramid = frame_pyramid(grass_with_box(), levels=2)
        self.assertEqual(sorted(LargeFeatureDetector().detect(pyramid, levels=(1, 2, 5))), [1])

    def test_detect_layer_matches_detect(self):
        pyramid = frame_pyramid(grass_with_box())
        detector = LargeFeatureDetector()
        a = detector.detect(pyramid, levels=(2,))[2]
        b = detector.detect_layer(pyramid[2], 2, pyramid[0].shape)
        np.testing.assert_array_equal(a.lines, b.lines)
        self.assertEqual(len(a.quads), len(b.quads))

    def test_structure_points_and_drawing(self):
        features = LargeFeatures(
            lines=np.array([[0, 0, 100, 0]], np.float32),
            corners=np.array([[50, 50]], np.float32),
            quads=[np.array([[10, 10], [90, 10], [90, 90], [10, 90]], np.float32)])
        pts = structure_points(features, spacing_px=10)
        self.assertEqual(pts.dtype, np.float32)
        np.testing.assert_array_equal(pts[0], [50, 50])
        on_line = pts[(pts[:, 1] == 0)]
        self.assertEqual(len(on_line), 11)
        interior = pts[(pts[:, 1] > 0) & ~np.all(pts == [50, 50], axis=1)]
        self.assertTrue(((interior > 10) & (interior < 90)).all())
        self.assertEqual(len(structure_points(features, 10, quad_interior=False)), 12)
        image = np.zeros((120, 120, 3), np.uint8)
        draw_large_features(image, features, (0, 255, 255))
        self.assertTrue(image.any())


if __name__ == "__main__":
    unittest.main()
