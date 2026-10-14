"""Dense stereo depth, feature depth and map points on synthetic stereo pairs."""

import unittest

import cv2
import numpy as np

from swarm.compVision.denseStereo import (DenseStereo, DepthMap, StereoRectifier,
                                          estimate_disparity_direction, fill_quad_plane,
                                          map_points, sample_depth)
from swarm.compVision.largeFeatures import LargeFeatureDetector, LargeFeatures
from swarm.mapping.terrainMap import TerrainMap

F, B = 500.0, 0.12
Z_GROUND, Z_BOX = 10.0, 6.0
BOX = (slice(150, 330), slice(260, 460))      # rows, cols of the raised box top


def stereo_pair(shape=(480, 640), seed=3, flat_box=False):
    """Left/right images with right-minus-left disparity along -x (SGBM layout)."""
    rng = np.random.default_rng(seed)
    ground = cv2.GaussianBlur(rng.integers(0, 256, shape, dtype=np.uint8), (5, 5), 1.2)
    top = np.full_like(ground[BOX], 180) if flat_box else 120 + ground[BOX] // 4
    left = ground.copy()
    left[BOX] = top
    right = np.roll(ground, -int(round(F * B / Z_GROUND)), axis=1)
    db = int(round(F * B / Z_BOX))
    right[BOX[0], BOX[1].start - db:BOX[1].stop - db] = top
    return left, right


# Transform a canonical (-1, 0) pair into one with the given baseline direction,
# and the inverse for mapping depth back to the canonical layout.
ORIENTATIONS = {
    (-1, 0): (lambda a: a, lambda a: a),
    (1, 0): (lambda a: a[:, ::-1], lambda a: a[:, ::-1]),
    (0, -1): (lambda a: a.T, lambda a: a.T),
    (0, 1): (lambda a: a.T[::-1], lambda a: a[::-1].T),
}


class DenseStereoTests(unittest.TestCase):
    def test_depth_of_ground_and_box_for_every_baseline_orientation(self):
        left, right = stereo_pair()
        for direction, (forward, back) in ORIENTATIONS.items():
            with self.subTest(direction=direction):
                stereo = DenseStereo(F, B, direction, min_depth=4.0)
                result = stereo.compute(np.ascontiguousarray(forward(left)),
                                        np.ascontiguousarray(forward(right)), 0)
                depth = back(result.depth)
                ground, box = depth[20:120, 100:600], depth[BOX]
                self.assertAlmostEqual(np.nanmedian(ground), Z_GROUND, delta=0.1)
                self.assertAlmostEqual(np.nanmedian(box), Z_BOX, delta=0.1)
                self.assertGreater(np.isfinite(box).mean(), 0.9)
                np.testing.assert_array_equal(result.valid, np.isfinite(result.depth))

    def test_pyramid_level_uses_scaled_focal_length(self):
        left, right = stereo_pair()
        stereo = DenseStereo(F, B, (-1, 0), min_depth=4.0)
        result = stereo.compute(cv2.pyrDown(left), cv2.pyrDown(right), 1)
        self.assertEqual(result.level, 1)
        self.assertEqual(result.depth.shape, (240, 320))
        self.assertAlmostEqual(np.nanmedian(result.depth[80:160, 135:225]), Z_BOX, delta=0.25)
        self.assertEqual(stereo.num_disparities(1), 16)
        self.assertEqual(stereo.num_disparities(0), 16)
        self.assertEqual(DenseStereo(F, B, (-1, 0), min_depth=1.0).num_disparities(0), 64)

    def test_off_axis_direction_rejected(self):
        with self.assertRaises(ValueError):
            DenseStereo(F, B, (1, 1))
        with self.assertRaises(ValueError):
            DenseStereo(F, B, (0, 0))

    def test_measured_direction(self):
        left, right = stereo_pair()
        direction, shift = estimate_disparity_direction(left, right)
        np.testing.assert_allclose(direction, [-1, 0], atol=0.05)
        self.assertAlmostEqual(shift, F * B / Z_GROUND, delta=1.0)


class FeatureDepthTests(unittest.TestCase):
    def depth_map(self, level=0):
        depth = np.full((480 // 2 ** level, 640 // 2 ** level), Z_GROUND, np.float32)
        r, c = BOX
        s = 2 ** level
        depth[r.start // s:r.stop // s, c.start // s:c.stop // s] = Z_BOX
        return DepthMap(depth=depth, valid=np.isfinite(depth), level=level)

    def test_sample_depth_snaps_edges_to_the_nearer_surface(self):
        dm = self.depth_map(level=1)
        edge = [[BOX[1].start, 200], [BOX[1].stop - 1, 240]]
        np.testing.assert_allclose(sample_depth(dm, edge), Z_BOX)
        np.testing.assert_allclose(sample_depth(dm, [[50, 50]]), Z_GROUND)
        dm.depth[:] = np.nan
        self.assertTrue(np.isnan(sample_depth(dm, [[50, 50]])).all())
        self.assertEqual(sample_depth(dm, np.empty((0, 2))).shape, (0,))

    def test_plane_fill_covers_textureless_top_from_its_outline(self):
        dm = self.depth_map()
        r, c = BOX
        # SGBM leaves holes on smooth tops: only a 6 px rim has depth.
        dm.depth[r.start + 6:r.stop - 6, c.start + 6:c.stop - 6] = np.nan
        dm.valid = np.isfinite(dm.depth)
        quad = np.array([[c.start, r.start], [c.stop - 1, r.start],
                         [c.stop - 1, r.stop - 1], [c.start, r.stop - 1]], np.float32)
        filled = fill_quad_plane(dm, quad, spacing_px=16)
        self.assertGreater(len(filled), 50)
        np.testing.assert_allclose(filled[:, 2], Z_BOX, rtol=0.01)
        self.assertTrue(((filled[:, 0] >= c.start) & (filled[:, 0] < c.stop)).all())

    def test_plane_fill_follows_a_tilted_roof(self):
        h, w = 240, 320
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        depth = (1.0 / (0.1 + 0.0002 * u)).astype(np.float32)   # plane: 1/Z affine in u
        dm = DepthMap(depth=depth, valid=np.ones_like(depth, bool), level=0)
        quad = np.array([[40, 40], [280, 40], [280, 200], [40, 200]], np.float32)
        filled = fill_quad_plane(dm, quad)
        np.testing.assert_allclose(filled[:, 2], 1.0 / (0.1 + 0.0002 * filled[:, 0]), rtol=1e-3)

    def test_map_points_weights_and_terrain_heights(self):
        left, right = stereo_pair(flat_box=True)
        pyramid = [left]
        for _ in range(3):
            pyramid.append(cv2.pyrDown(pyramid[-1]))
        features = LargeFeatureDetector().detect(pyramid, levels=(1, 2))
        self.assertTrue(all(len(f.quads) == 1 for f in features.values()))
        depth = DenseStereo(F, B, (-1, 0), min_depth=4.0).compute(
            pyramid[1], cv2.pyrDown(right), 1)
        points = map_points(features, depth)
        self.assertEqual(points.shape[1], 4)
        self.assertEqual(set(np.unique(points[:, 3])), {1.0, 4.0})
        structure = points[points[:, 3] == 4.0]
        inside = ((structure[:, 0] > BOX[1].start + 8) & (structure[:, 0] < BOX[1].stop - 8) &
                  (structure[:, 1] > BOX[0].start + 8) & (structure[:, 1] < BOX[0].stop - 8))
        # The textureless top gets no SGBM depth; the silhouette match fills it
        # at the box disparity (5 px at level 1), to within +-0.5 layer px.
        self.assertLess(depth.valid[BOX[0].start // 2 + 6:BOX[0].stop // 2 - 6,
                                    BOX[1].start // 2 + 6:BOX[1].stop // 2 - 6].mean(), 0.1)
        self.assertGreater(inside.sum(), 20)
        disparity = depth.focal_px * B / structure[inside, 2]
        np.testing.assert_allclose(disparity, F * B / Z_BOX / 2, atol=0.6)
        self.assertEqual(map_points(features, None).shape, (0, 4))
        self.assertEqual(map_points({}, depth)[:, 3].max(), 1.0)
        self.assertEqual(len(map_points(LargeFeatures(level=1), depth)),
                         len(map_points({}, depth)))

    def test_terrain_weight_scale_moves_the_cell_average(self):
        class Cam:
            res, focal_px, baseline = (640, 480), F, B

            @staticmethod
            def backproject(pts, depth):
                return np.column_stack([np.zeros(len(depth)), np.zeros(len(depth)), -depth])

        pts = np.array([[320, 240, 10.0], [320, 240, 6.0]])
        plain, weighted = TerrainMap(Cam, min_hits=1), TerrainMap(Cam, min_hits=1)
        plain.add(pts, np.zeros(3), (1, 0, 0, 0))
        weighted.add(pts, np.zeros(3), (1, 0, 0, 0), weight_scale=[1000.0, 1.0])
        h_plain, h_weighted = np.nanmax(plain.height()), np.nanmax(weighted.height())
        # Inverse-variance weights ~ 1/Z^4 already favour the near point (-6.46)...
        self.assertAlmostEqual(h_plain, -(10 * 6 ** 4 + 6 * 10 ** 4) / (6 ** 4 + 10 ** 4), delta=0.01)
        # ...and weight_scale can override that.
        self.assertAlmostEqual(h_weighted, -10.0, delta=0.05)


class RectifierTests(unittest.TestCase):
    def test_identity_passes_images_through(self):
        rect = StereoRectifier.identity()
        left, right = np.zeros((4, 4), np.uint8), np.ones((4, 4), np.uint8)
        self.assertTrue(rect.is_identity)
        out = rect.rectify(left, right)
        self.assertIs(out[0], left)
        self.assertIs(out[1], right)

    def test_calibration_gives_axis_aligned_direction_and_baseline(self):
        K1 = np.array([[1700.0, 0, 640], [0, 1700, 400], [0, 0, 1]])
        K2 = np.array([[1200.0, 0, 640], [0, 1200, 400], [0, 0, 1]])
        D = np.zeros(5)
        R = cv2.Rodrigues(np.array([0.0, 0.01, 0.0]))[0]
        for T, expected_axis in (([-0.125, 0, 0], 0), ([0, -0.125, 0], 1)):
            with self.subTest(T=T):
                rect = StereoRectifier.from_calibration(K1, D, K2, D, R, T, (1280, 800))
                self.assertFalse(rect.is_identity)
                self.assertAlmostEqual(rect.baseline, 0.125, delta=1e-3)
                self.assertAlmostEqual(abs(rect.disparity_direction[expected_axis]), 1.0, places=6)
                self.assertGreater(rect.focal_px, 0)
                image = np.full((800, 1280), 128, np.uint8)
                l, r = rect.rectify(image, image)
                self.assertEqual(l.shape, image.shape)
                with self.assertRaises(ValueError):
                    rect.rectify(image[:400], image[:400])
                DenseStereo(rect.focal_px, rect.baseline, rect.disparity_direction)


if __name__ == '__main__':
    unittest.main()
