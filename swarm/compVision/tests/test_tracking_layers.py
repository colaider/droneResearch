"""Temporal tracking: pixel units, budget, state and drawing."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from swarm.compVision.smallFeatures import TRACK_COLOR, build_triangles, replenish
from swarm.compVision.tests.estimator_factory import make_estimator
from swarm.visualization.display import draw_tracks


class TrackingTests(unittest.TestCase):
    def setUp(self):
        # Odd dimensions catch incorrect width/height conversions.
        self.width, self.height = 641, 481
        texture = np.random.default_rng(4).integers(0, 256, (61, 81), dtype=np.uint8)
        self.image = cv2.cvtColor(cv2.resize(texture, (self.width, self.height),
                                           interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        self.est = make_estimator(
            (self.width, self.height), 60,
            disparity_direction=(0., 1.), image_filters=(),
        )
        self.est.drone_pos[2] = 10

    def stereo(self, index):
        def shifted(dx, dy):
            return cv2.warpAffine(self.image, np.float32([[1, 0, dx], [0, 1, dy]]),
                                  (self.width, self.height), borderMode=cv2.BORDER_REFLECT)
        return [shifted(4 * index, 0), shifted(4 * index, 4)]

    def run_frames(self):
        self.est.max_feature_points = 181
        self.est.feature_min_distance = 15
        self.est.processing(self.stereo(0), 0)
        self.est.processing(self.stereo(1), 1)

    def test_default_tracks_and_outputs_every_point_with_its_color(self):
        # Zero height excludes velocity observations, but must not hide image tracks.
        self.est.drone_pos[2] = 0
        self.est.processing(self.stereo(0), 0)
        self.est.processing(self.stereo(1), 1)
        points = self.est.tracks.new_points
        self.assertFalse(self.est.velocity_point_mask.any())
        self.assertLessEqual(len(points), self.est.max_feature_points)
        self.assertGreater(len(points), 3)
        np.testing.assert_array_equal(
            self.est.depth_colors, np.tile(TRACK_COLOR, (len(self.est.depth_colors), 1)))
        self.assertEqual(self.est.depth_colors.dtype, np.uint8)

    def test_output_clears_on_lost_tracks_and_recovers(self):
        self.run_frames()
        with patch('swarm.compVision.visualVelEst.cv2.calcOpticalFlowPyrLK',
                   return_value=(None, None, None)):
            self.est.processing(self.stereo(2), 2)
        self.assertEqual(self.est.tracks.new_points.shape, (0, 2))
        self.assertEqual(self.est.tracked_points.shape, (0, 2))
        self.assertEqual(self.est.velocity_points.shape, (0, 3))
        self.est.processing(self.stereo(3), 3)
        self.assertGreater(len(self.est.tracks.new_points), 0)

    def test_real_lk_tracks_same_motion_in_full_resolution(self):
        self.run_frames()
        tracks = self.est.tracks
        self.assertGreater(len(tracks.new_points), 20)
        np.testing.assert_allclose(np.median(tracks.new_points - tracks.old_points, axis=0),
                                   [4, 0], atol=.05)
        np.testing.assert_allclose(np.median(tracks.flow, axis=0), [4, 0], atol=.1)
        np.testing.assert_allclose(self.est.tracked_points, tracks.new_points, atol=1e-5)
        self.assertGreater(len(tracks.triangles), 0)
        self.assertLess(tracks.triangles.max(), len(tracks.new_points))
        self.assertLessEqual(len(tracks.new_points), 181)
        self.assertEqual(len(tracks.new_points), len(self.est.velocity_point_mask))
        self.assertTrue(np.isfinite(self.est.camera_velocity).all())
        self.assertGreater(len(self.est.depth_points), 20)

    def test_overlays_use_track_color_and_leave_tracking_images_clean(self):
        self.run_frames()
        original = self.stereo(1)
        for camera in (0, 1):
            np.testing.assert_array_equal(self.est.buffer[-1].frames[camera], original[camera])
            np.testing.assert_array_equal(self.est.buffer[-1].gray_frames[camera],
                                          cv2.cvtColor(original[camera], cv2.COLOR_BGR2GRAY))
            display = self.est.current_frame.frames[camera][90:]
            if camera == 0 or self.est.point_depth.valid.any():
                self.assertTrue(np.any(np.all(display == TRACK_COLOR, axis=2)))
        # Explicitly check edge pixels, independent of legends/point markers.
        canvas = np.zeros((40, 40, 3), np.uint8)
        points = np.float32([[5, 5], [30, 5], [5, 30]])
        left, right = draw_tracks(canvas.copy(), canvas.copy(), points, points,
                                  np.array([[0, 1], [1, 2], [0, 2]]), color=TRACK_COLOR)
        np.testing.assert_array_equal(left[5, 15], TRACK_COLOR)
        np.testing.assert_array_equal(right[5, 15], TRACK_COLOR)

    def test_skipped_processing_does_not_reuse_seeds_from_wrong_frame(self):
        self.run_frames()
        self.est.update_frame(self.stereo(2), 2)
        self.est.update_frame(self.stereo(3), 3)
        received_seed_lengths = []
        def capture(gray, points, camera, **kwargs):
            received_seed_lengths.append(len(points))
            return replenish(gray, points, camera, **kwargs)
        with patch('swarm.compVision.visualVelEst.replenish', side_effect=capture):
            self.est.aply_flow()
        self.assertEqual(received_seed_lengths, [0])
        self.assertEqual(self.est.tracks.frame_idx, 3)

    def test_feature_spacing_uses_configured_distance(self):
        self.est.processing(self.stereo(0), 0)
        with patch('swarm.compVision.visualVelEst.replenish', wraps=replenish) as detect:
            self.est.processing(self.stereo(1), 1)
        self.assertEqual([call.kwargs['min_distance'] for call in detect.call_args_list], [30])

    def test_tracking_view_uses_pixel_coordinates_and_track_color(self):
        self.est.processing(self.stereo(0), 0)
        self.assertEqual(self.est.tracks.new_points.shape, (0, 2))
        self.est.processing(self.stereo(1), 1)
        view = self.est.tracking_view
        self.assertEqual(view.shape[:2], (self.height + 28, max(330, self.width)))
        points = np.rint(self.est.tracks.new_points).astype(int)
        points = points[(points[:, 0] < self.width) & (points[:, 1] < self.height)]
        self.assertGreater(len(points), 3)
        np.testing.assert_array_equal(view[points[:, 1] + 28, points[:, 0]],
                                      np.tile(TRACK_COLOR, (len(points), 1)))

    def test_simulation_opens_tracking_window_and_reuses_it(self):
        import importlib.util
        from pathlib import Path
        from types import SimpleNamespace
        # Load only the display wrapper; camera/Genesis startup is not needed here.
        source = Path(__file__).resolve().parents[2] / 'droneClasses' / 'droneStruct.py'
        spec = importlib.util.spec_from_file_location('_tracking_window_test', source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', {'genesis.utils.misc': SimpleNamespace(tensor_to_array=lambda x: x)}):
            spec.loader.exec_module(module)
        drone = module.DroneStruct.__new__(module.DroneStruct)
        drone.frame_processor = self.est
        drone._camera_display = None
        drone._camera_window_ready = False
        drone._tracking_window_ready = False
        drone._depth_window_ready = False
        self.run_frames()
        with patch.object(cv2, 'namedWindow') as create, patch.object(cv2, 'resizeWindow'), \
             patch.object(cv2, 'imshow') as show, patch.object(cv2, 'waitKey') as pump:
            drone.camera_show()
            self.assertEqual([c.args[0] for c in create.call_args_list],
                             ['Tracking', 'Stereo depth'])
            self.assertIs(show.call_args_list[0].args[1], self.est.tracking_view)
            drone.camera_show()
            self.assertEqual(create.call_count, 2)
            self.assertEqual(pump.call_count, 2)

    def test_unbalanced_existing_points_cannot_exceed_shared_budget(self):
        # All retained points are in one quadrant; quotas in other quadrants
        # must still respect the total number of available slots.
        gray = np.full((100, 100), 255, np.uint8)
        existing = np.float32([[10, 10], [11, 11], [12, 12]])
        candidates = np.float32([[60, 10], [70, 10], [10, 60], [10, 70], [60, 60], [70, 70]])
        with patch('swarm.compVision.smallFeatures.cv2.goodFeaturesToTrack',
                   return_value=candidates[:, None, :]), patch('swarm.compVision.smallFeatures.cv2.Canny',
                                                               return_value=gray):
            result = replenish(gray, existing, 0, max_points=4)
        self.assertEqual(len(result), 4)
        np.testing.assert_array_equal(result[:3], existing)

    def test_degenerate_mesh_does_not_crash_a_sparse_layer(self):
        for points in ([[1, 1], [2, 2], [3, 3]], [[1, 1], [1, 1], [1, 1]]):
            triangles, _, links, _ = build_triangles(points)
            self.assertEqual(triangles.shape, (0, 3))
            self.assertEqual(links.shape, (0, 2))


if __name__ == '__main__':
    unittest.main()
