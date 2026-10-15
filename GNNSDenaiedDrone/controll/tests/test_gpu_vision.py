"""Camera-free tests; UMat can use CPU fallback on machines without OpenCL."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensors.visual.gpuVision import VisionBackend
from sensors.visual.smallFeatures import replenish
from sensors.visual.visualDepthEst import SparseStereoDepth
from sensors.visual.visualEst import VisulaEst
from sensors.visual.config.cameraConfig import StereoCameraConfig
from sensors.visual.estimatorTypes import StereoFrame


def frame(left, right=None):
    return SimpleNamespace(pyramids=[[left], [left if right is None else right]])


class GPUVisionTests(unittest.TestCase):
    def setUp(self):
        self.image = cv2.GaussianBlur(np.random.default_rng(4).integers(
            0, 256, (241, 321), dtype=np.uint8), (3, 3), 0)
        self.points = np.float32([(x, y) for y in range(40, 190, 30)
                                for x in range(40, 220, 30)]).reshape(-1, 1, 2)

    def shifted(self, dx, dy=0):
        return cv2.warpAffine(self.image, np.float32([[1, 0, dx], [0, 1, dy]]),
                             (321, 241), borderMode=cv2.BORDER_REFLECT)

    def test_temporal_lk(self):
        shifted = self.shifted(3, 2)
        for mode in ('cpu', 'opencl'):
            with self.subTest(mode=mode):
                backend = VisionBackend(mode)
                backend.begin_frame(frame(self.image))
                backend.begin_frame(frame(shifted))
                result, status, _ = backend.calcOpticalFlowPyrLK(
                    self.image, shifted, self.points, winSize=(21, 21), maxLevel=1)
                valid = status.ravel() == 1
                self.assertGreater(valid.sum(), 20)
                np.testing.assert_allclose(np.median((result-self.points)[valid, 0], axis=0),
                                           [3, 2], atol=.1)

    def test_seeded_lk_large_disparity_and_reverse(self):
        # At maxLevel=0 a 28px displacement needs the initial stereo predictions.
        shifted = self.shifted(28)
        expected = self.points + np.float32([28, 0])
        for mode in ('cpu', 'opencl'):
            with self.subTest(mode=mode):
                backend = VisionBackend(mode)
                backend.begin_frame(frame(self.image, shifted))
                parameters = dict(winSize=(15, 15), maxLevel=0,
                                  flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
                matched, status, _ = backend.calcOpticalFlowPyrLK(
                    self.image, shifted, self.points, expected.copy(), **parameters)
                valid = status.ravel() == 1
                self.assertGreater(valid.sum(), 20)
                np.testing.assert_allclose(matched[valid], expected[valid], atol=.15)
                back, status, _ = backend.calcOpticalFlowPyrLK(
                    shifted, self.image, matched[valid], self.points[valid].copy(), **parameters)
                np.testing.assert_allclose(back[status.ravel() == 1],
                                           self.points[valid][status.ravel() == 1], atol=.15)

    def test_gftt_mask_canny_and_dilation(self):
        mask = np.zeros_like(self.image)
        mask[20:220, 20:150] = 255
        expected_edges = cv2.dilate(cv2.Canny(self.image, 50, 150), np.ones((5, 5), np.uint8))
        for mode in ('cpu', 'opencl'):
            with self.subTest(mode=mode):
                backend = VisionBackend(mode)
                backend.begin_frame(frame(self.image))
                corners = backend.goodFeaturesToTrack(self.image, maxCorners=50,
                    qualityLevel=.005, minDistance=10, blockSize=11, mask=mask)
                self.assertGreater(len(corners), 10)
                xy = corners[:, 0].astype(int)
                self.assertTrue((mask[xy[:, 1], xy[:, 0]] > 0).all())
                edges = backend.Canny(self.image, 50, 150)
                result = backend.dilate(edges, np.ones((5, 5), np.uint8))
                self.assertIsInstance(result, np.ndarray)
                self.assertLess(np.mean(result != expected_edges), .01)

    def test_empty_replenishment(self):
        blank = np.zeros_like(self.image)
        backend = VisionBackend('opencl')
        backend.begin_frame(frame(blank))
        result = replenish(blank, [], 0, backend=backend)
        self.assertEqual(result.shape, (0, 2))

    def test_cache_reuses_both_cameras_and_evicts_old_frames(self):
        backend = VisionBackend('opencl')
        pairs = [frame(self.image.copy(), self.image.copy()) for _ in range(3)]
        for pair in pairs[:2]:
            backend.begin_frame(pair)
            for camera in pair.pyramids:
                backend.Canny(camera[0], 50, 150)
        self.assertEqual(len(backend._images), 4)
        key = id(pairs[1].pyramids[1][0])
        cached = backend._images[key]
        backend.begin_frame(pairs[2])
        self.assertEqual(len(backend._images), 2)
        self.assertIs(backend._images[key], cached)
        for camera in pairs[0].pyramids:
            self.assertNotIn(id(camera[0]), backend._images.sources)
        for camera in pairs[2].pyramids:
            backend.Canny(camera[0], 50, 150)
        self.assertEqual(len(backend._images), 4)

    def test_stereo_depth_uses_shared_backend(self):
        right = self.shifted(12)
        for mode in ('cpu', 'opencl'):
            with self.subTest(mode=mode):
                backend = VisionBackend(mode)
                backend.begin_frame(frame(self.image, right))
                stereo = SparseStereoDepth(backend=backend)
                result = stereo.estimate(self.image, right, self.points[:, 0],
                                         focal_px=160, baseline=.06, direction=(1., 0.))
                self.assertGreater(result.valid.sum(), 20)
                np.testing.assert_allclose(result.depths[result.valid], .8, atol=.03)

    def test_estimator_routes_all_operations_through_shared_backend(self):
        cam = StereoCameraConfig(res=(321, 241), left_focal=160, right_focal=160,
                                 epipolar_direction=(1., 0.))
        for layers in (1, 2):
            est = VisulaEst(cam, image_filters=(), convert_to_grayscale=True,
                filters_enabled=False, filter_kernel_size=3, pyramid_levels=layers,
                max_stereo_skew_s=.005, vision_backend='opencl')
            est.max_feature_points, est.feature_min_distance = 120, 12
            calls = dict.fromkeys(('calcOpticalFlowPyrLK', 'goodFeaturesToTrack', 'Canny', 'dilate'), 0)
            for name in calls:
                operation = getattr(est.vision_backend, name)
                def counted(*args, _name=name, _operation=operation, **kwargs):
                    calls[_name] += 1
                    return _operation(*args, **kwargs)
                setattr(est.vision_backend, name, counted)
            for i in range(3):
                pair = StereoFrame(self.shifted(2*i), self.shifted(2*i+12),
                                   i*.033, i*.033, i)
                est.processing(pair, drone_pos=np.float64([0, 0, 2]),
                    drone_vel=np.zeros(3), drone_ang_vel=np.zeros(3), imu_att=np.zeros(3))
            self.assertIs(est.stereo.backend, est.vision_backend)
            self.assertTrue(all(count > 0 for count in calls.values()))
            self.assertGreater(len(est.depth_points), 10)
            self.assertTrue(np.isfinite(est.camera_velocity).all())
            self.assertLessEqual(len(est.vision_backend._images), 4 * layers)


if __name__ == '__main__':
    unittest.main()
