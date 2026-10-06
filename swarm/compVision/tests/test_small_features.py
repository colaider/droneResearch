"""Feature-stage regression tests against the original detector and known motion.

The frozen legacy function below is extracted from visualAccEst.before.py,
VisulaAcEst._replenish. It provides an independent pre-refactor baseline without
requiring the original monolithic class or its camera configuration at runtime.
"""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from swarm.compVision.smallFeatures import FeatureTracks, SmallFeatureTracker


def legacy_replenish(self, gray, pts, cam, max_points=300):
    h, w = gray.shape
    mask = np.full((h, w), 255, np.uint8)
    for x, y in pts.astype(int):
        cv2.circle(mask, (x, y), 7, 0, -1)
    cand = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.005, minDistance=100, blockSize=11, mask=mask)
    if cand is None:
        return pts
    cand = cand.reshape(-1, 2)
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((5, 5), np.uint8))
    xs = np.clip(cand[:, 0].astype(int), 0, w - 1)
    ys = np.clip(cand[:, 1].astype(int), 0, h - 1)
    keep = edges[ys, xs] > 0
    cand = cand[keep]
    if len(cand) == 0:
        return pts

    def quad(p):
        return (p[:, 0] >= w / 2).astype(int) + 2 * (p[:, 1] >= h / 2).astype(int)
    pt_q = quad(pts) if len(pts) else np.array([], dtype=int)
    cand_q = quad(cand)
    need = max_points - len(pts)
    if need <= 0:
        return pts
    existing = np.bincount(pt_q, minlength=4)
    target = (len(pts) + need) // 4
    quota = np.maximum(target - existing, 0)
    take = []
    for q in range(4):
        in_q = np.flatnonzero(cand_q == q)
        take.extend(in_q[:quota[q]])
    remaining_need = need - len(take)
    if remaining_need > 0:
        remaining = np.setdiff1d(np.arange(len(cand)), take)
        take.extend(remaining[:remaining_need])
    if not take:
        return pts
    return np.vstack([pts, cand[take]]).astype(np.float32)


def textured_image(seed=31, shape=(480, 640)):
    image = np.random.default_rng(seed).integers(0, 256, shape, np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0)


class SmallFeatureTests(unittest.TestCase):
    def test_no_region_detection_matches_original_for_ordinary_frames(self):
        tracker = SmallFeatureTracker()
        for seed in (5, 31, 72):
            gray = textured_image(seed)
            for existing in (np.empty((0, 2), np.float32),
                             np.array([[80, 80], [460, 80],
                                       [80, 380], [460, 380]], np.float32)):
                with self.subTest(seed=seed, existing=len(existing)):
                    expected = legacy_replenish(None, gray, existing, 0, 32)
                    actual = tracker.replenish(gray, existing, 0, 32)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(actual.dtype, np.float32)

    def test_uneven_existing_tracks_cannot_overallocate_new_slots(self):
        gray = textured_image(shape=(600, 800))
        existing = np.array([[30 + 25 * (i % 4), 30 + 25 * (i // 4)]
                             for i in range(12)], np.float32)
        result = SmallFeatureTracker().replenish(gray, existing, max_points=16)
        self.assertEqual(result.shape, (16, 2))
        np.testing.assert_array_equal(result[:len(existing)], existing)
        # This specifically exercises the old quadrant oversubscription bug.
        original = legacy_replenish(None, gray, existing, 0, 16)
        self.assertGreater(len(original), 16)

    def test_roi_priority_is_bounded_and_global_coverage_remains(self):
        gray = textured_image(shape=(600, 800))
        tracker = SmallFeatureTracker(max_points=20, region_fraction=0.25)
        original_gftt = cv2.goodFeaturesToTrack
        with patch('swarm.compVision.smallFeatures.cv2.goodFeaturesToTrack',
                   wraps=original_gftt) as detector:
            result = tracker.replenish(gray, [], max_points=20,
                                       regions=[(0, 0, 800, 300)])
        self.assertEqual(len(detector.call_args_list), 2)
        roi_request, global_request = detector.call_args_list
        self.assertEqual(roi_request.kwargs['maxCorners'], 5)
        self.assertEqual(global_request.kwargs['maxCorners'], 500)
        self.assertTrue((roi_request.kwargs['mask'][300:] == 0).all())
        self.assertEqual(result.shape, (20, 2))
        self.assertTrue((result[:5, 1] < 300).all())
        self.assertTrue((result[5:, 1] >= 300).any())
        # The fraction reserves first-choice slots, not a cap on the final
        # number of ROI points: global selection may also choose ROI points.
        distances = np.linalg.norm(result[:, None] - result[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        self.assertGreaterEqual(float(distances.min()), tracker.min_distance)

    def test_invalid_existing_points_removed_and_valid_order_retained(self):
        gray = textured_image()
        points = np.array([[42.5, 39.5], [np.nan, 20], [-1, 20],
                           [641, 20], [100, 120]], np.float64)
        result = SmallFeatureTracker().replenish(gray, points, max_points=2)
        np.testing.assert_array_equal(result, points[[0, 4]].astype(np.float32))
        self.assertEqual(result.dtype, np.float32)

    def test_blank_images_and_zero_budget_have_empty_outputs(self):
        gray = np.zeros((240, 320), np.uint8)
        for tracker in (SmallFeatureTracker(), SmallFeatureTracker(max_points=0)):
            result = tracker.track(gray, gray)
            self.assertEqual(result.old_points.shape, (0, 2))
            self.assertEqual(result.new_points.dtype, np.float32)
            self.assertEqual(result.triangles.shape, (0, 3))
            self.assertEqual(result.links.shape, (0, 2))
        tracker = SmallFeatureTracker()
        self.assertEqual(tracker.replenish(textured_image(), [], max_points=0).shape, (0, 2))

    def test_degenerate_mesh_inputs_are_empty(self):
        for points in ([], [[10, 10], [20, 20]],
                       [[10, 10], [20, 20], [30, 30], [40, 40]],
                       [[10, 10], [10, 10], [10, 10]],
                       [[10, 10], [20, 20], [np.nan, 30]]):
            with self.subTest(points=points):
                triangles, vertices, links, adjacency = SmallFeatureTracker.build_triangles(points)
                self.assertEqual(triangles.shape, (0, 3))
                self.assertEqual(vertices.dtype, np.float32)
                self.assertEqual(links.shape, (0, 2))
                self.assertEqual(adjacency.shape, (len(points), len(points)))

    def test_failed_forward_lk_clears_tracks_without_backward_call(self):
        gray = np.zeros((120, 160), np.uint8)
        points = np.array([[20, 20], [80, 20], [20, 80], [80, 80]], np.float32)
        statuses = np.ones((4, 1), np.uint8)
        cases = [(None, None, None),
                 (points.reshape(-1, 1, 2), np.zeros_like(statuses), None),
                 (np.full((4, 1, 2), np.nan, np.float32), statuses, None)]
        for result in cases:
            tracker = SmallFeatureTracker()
            tracker.tracked_points[0] = points.copy()
            with patch.object(tracker, 'replenish', return_value=points), \
                 patch('swarm.compVision.smallFeatures.cv2.calcOpticalFlowPyrLK',
                       return_value=result) as lk:
                output = tracker.track(gray, gray)
            self.assertEqual(lk.call_count, 1)
            self.assertEqual(output.new_points.shape, (0, 2))
            self.assertEqual(tracker.tracked_points[0].shape, (0, 2))

    def test_failed_backward_lk_clears_tracks(self):
        gray = np.zeros((120, 160), np.uint8)
        points = np.array([[20, 20], [80, 20], [20, 80], [80, 80]], np.float32)
        tracker = SmallFeatureTracker()
        forward = (points.reshape(-1, 1, 2), np.ones((4, 1), np.uint8), None)
        with patch.object(tracker, 'replenish', return_value=points), \
             patch('swarm.compVision.smallFeatures.cv2.calcOpticalFlowPyrLK',
                   side_effect=[forward, (None, None, None)]) as lk:
            output = tracker.track(gray, gray)
        self.assertEqual(lk.call_count, 2)
        self.assertEqual(output.new_points.shape, (0, 2))
        self.assertEqual(tracker.tracked_points[0].shape, (0, 2))

    def test_known_translation_survives_tracking_and_mesh_filters(self):
        first = textured_image(shape=(480, 640))
        motion = np.array([3.0, -2.0], np.float32)
        affine = np.array([[1, 0, motion[0]], [0, 1, motion[1]]], np.float32)
        second = cv2.warpAffine(first, affine, (640, 480), borderMode=cv2.BORDER_REFLECT)
        tracker = SmallFeatureTracker()
        result = tracker.track(first, second)
        self.assertGreaterEqual(len(result.new_points), 8)
        self.assertGreater(len(result.triangles), 0)
        self.assertGreater(len(result.links), 0)
        self.assertEqual(result.new_points.dtype, np.float32)
        flow = result.new_points - result.old_points
        np.testing.assert_allclose(np.median(flow, axis=0), motion, atol=0.05)
        self.assertLess(float(np.percentile(np.linalg.norm(flow - motion, axis=1), 90)), 0.1)
        np.testing.assert_array_equal(tracker.tracked_points[0], result.new_points)
        result.new_points[:] = 0
        self.assertTrue(np.any(tracker.tracked_points[0]))


if __name__ == '__main__':
    unittest.main()
