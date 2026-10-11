"""StructureMapper: coarse-layer structures + dense depth into the TerrainMap height grid."""

import unittest
from contextlib import redirect_stdout
from io import StringIO

import cv2
import numpy as np

from swarm.compVision.frame import FrameProcessor
from swarm.compVision.structureMap import StructureMapper, camera_for
from swarm.compVision.denseStereo import StereoRectifier
from swarm.compVision.terrainMap import TerrainMap
from swarm.config.cameraConfig import StereoCameraConfig

F, B, Z_GROUND, Z_BOX = 500.0, 0.12, 10.0, 6.0
BOX = (slice(150, 330), slice(260, 460))


def grass_pair(shape=(480, 640), seed=5):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (shape[0] // 2, shape[1] // 2), dtype=np.uint8)
    ground = cv2.resize(cv2.GaussianBlur(small, (3, 3), 0.8), shape[::-1])
    top = 170 + ground[BOX] // 32                       # almost textureless box top
    left = ground.copy()
    left[BOX] = top
    right = np.roll(ground, -int(round(F * B / Z_GROUND)), axis=1)
    db = int(round(F * B / Z_BOX))
    right[BOX[0], BOX[1].start - db:BOX[1].stop - db] = top
    bgr = lambda g: cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return FrameProcessor(image_filters=(), pyramid_levels=4).process([bgr(left), bgr(right)], 7)


class StructureMapperTests(unittest.TestCase):
    def test_box_rises_above_ground_in_the_height_grid(self):
        # Deliberately configure a rotated mount to exercise direction-mismatch detection.
        cfg = StereoCameraConfig(res=(640, 480), left_focal=F, right_focal=F, baseline=B,
                                 left_up=(1, 0, 0), right_up=(1, 0, 0))
        self.assertGreater(abs(cfg.disparity_direction()[1]), 0.9)
        # One frame only, so every observed cell counts (default needs 2 hits).
        mapper = StructureMapper(cfg, levels=(1, 2), depth_level=0, min_depth=4.0,
                                 terrain=TerrainMap(cfg, min_hits=1))
        frame = grass_pair()
        with redirect_stdout(StringIO()) as log:
            points = mapper.step(frame, drone_pos=(0, 0, Z_GROUND), drone_quat=(1, 0, 0, 0))
        self.assertIn('using [-1.  0.]', log.getvalue())
        np.testing.assert_allclose(mapper.dense.direction, [-1, 0])

        self.assertEqual(points.shape[1], 4)
        stats = mapper.stats()
        self.assertGreaterEqual(stats['quads'], 2)          # one per level
        self.assertAlmostEqual(stats['structure_depth_m'], Z_BOX, delta=0.3)

        height = mapper.terrain.height()
        observed = height[np.isfinite(height)]
        self.assertGreater(len(observed), 100)
        self.assertAlmostEqual(np.median(observed), 0.0, delta=0.2)          # ground
        self.assertAlmostEqual(np.percentile(observed, 99), Z_GROUND - Z_BOX, delta=0.4)

        image = frame.frames[0].copy()
        mapper.draw(image)
        self.assertFalse(np.array_equal(image, frame.frames[0]))
        # The same frame index is only added to the map once.
        before = mapper.terrain.hits.sum()
        mapper.step(frame, drone_pos=(0, 0, Z_GROUND), drone_quat=(1, 0, 0, 0))
        self.assertEqual(mapper.terrain.hits.sum(), before)

    def test_depth_disabled_detects_only(self):
        mapper = StructureMapper(StereoCameraConfig(res=(640, 480)), levels=(2,), depth_level=None)
        points = mapper.step(grass_pair())
        self.assertEqual(points.shape, (0, 4))
        self.assertIsNone(mapper.depth_map)
        self.assertEqual(sorted(mapper.features), [2])
        self.assertIsNone(mapper.stats()['structure_depth_m'])

    def test_camera_for_rectified_pair(self):
        self.assertEqual(camera_for(None, (640, 480)).res, (640, 480))
        K = np.array([[300.0, 0, 160], [0, 300, 120], [0, 0, 1]])
        rect = StereoRectifier.from_calibration(K, np.zeros(5), K, np.zeros(5), np.eye(3),
                                                [0, 0.1, 0], (320, 240))
        cfg = camera_for(rect, (320, 240))
        self.assertAlmostEqual(cfg.focal_px, rect.focal_px)
        self.assertAlmostEqual(cfg.baseline, 0.1, places=6)
        np.testing.assert_allclose(cfg.disparity_direction(), [0, 1])


if __name__ == '__main__':
    unittest.main()
