"""All-layer pyramid tracking: pixel units, budgets, state and drawing."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from swarm.compVision.smallFeatures import LAYER_COLORS, build_triangles, replenish, unique_point_mask
from swarm.compVision.visualAccEst import VisulaAcEst


class TrackingLayerTests(unittest.TestCase):
    def setUp(self):
        # Odd dimensions catch the incorrect full-size / layer-size conversion.
        self.width, self.height = 641, 481
        texture = np.random.default_rng(4).integers(0, 256, (61, 81), dtype=np.uint8)
        self.image = cv2.cvtColor(cv2.resize(texture, (self.width, self.height),
                                           interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        self.est = VisulaAcEst((self.width, self.height), 60)
        self.est.frame_processor.pyramid_levels = 3  # fixed fixture; production may use more layers
        self.est.disparity_direction = np.array([0., 1.])  # synthetic right image shifts down
        self.est.image_filters = ()
        self.est.drone_pos[2] = 10

    def stereo(self, index):
        def shifted(dx, dy):
            return cv2.warpAffine(self.image, np.float32([[1, 0, dx], [0, 1, dy]]),
                                  (self.width, self.height), borderMode=cv2.BORDER_REFLECT)
        return [shifted(4 * index, 0), shifted(4 * index, 4)]

    def run_layers(self):
        self.est.max_feature_points = 181
        self.est.feature_min_distance = 15
        self.est.processing(self.stereo(0), 0)
        self.est.processing(self.stereo(1), 1)

    def test_default_tracks_all_layers_and_outputs_every_point_with_its_color(self):
        # Zero height excludes velocity observations, but must not hide image tracks.
        self.est.drone_pos[2] = 0
        self.est.processing(self.stereo(0), 0)
        self.est.processing(self.stereo(1), 1)
        self.assertEqual(set(self.est.point_layers), {0, 1, 2})
        self.assertFalse(self.est.velocity_point_mask.any())
        self.assertLessEqual(sum(map(len, self.est.all_points)), self.est.max_feature_points)
        for layer in self.est.layer_tracks:
            points = self.est.all_points[layer]
            self.assertGreater(len(points), 3)
            np.testing.assert_array_equal(points, self.est.layer_tracks[layer].new_points)
            np.testing.assert_array_equal(self.est.point_colors[layer],
                np.tile(LAYER_COLORS[layer], (len(points), 1)))
            self.assertEqual(self.est.point_colors[layer].dtype, np.uint8)

    def test_combined_output_clears_on_lost_tracks_and_recovers(self):
        self.run_layers()
        with patch('swarm.compVision.visualAccEst.cv2.calcOpticalFlowPyrLK',
                   return_value=(None, None, None)):
            self.est.processing(self.stereo(2), 2)
        self.assertEqual([p.shape for p in self.est.all_points], [(0, 2)] * 3)
        self.assertEqual([c.shape for c in self.est.point_colors], [(0, 3)] * 3)
        self.assertEqual(self.est.point_layers.shape, (0,))
        self.est.processing(self.stereo(3), 3)
        self.assertGreater(sum(map(len, self.est.all_points)), 0)
        self.assertEqual(set(self.est.point_layers), {0, 1, 2})

    def test_real_lk_tracks_same_motion_in_full_resolution_at_every_layer(self):
        self.run_layers()
        for layer, tracks in self.est.layer_tracks.items():
            with self.subTest(layer=layer):
                self.assertGreater(len(tracks.new_points), 20)
                np.testing.assert_allclose(np.median(tracks.new_points - tracks.old_points, axis=0),
                                           [4, 0], atol=.05)
                np.testing.assert_allclose(np.median(tracks.flow, axis=0), [4, 0], atol=.1)
                np.testing.assert_allclose(self.est.tracked_points_by_layer[layer][0] * 2**layer,
                                           tracks.new_points, atol=1e-5)
                self.assertGreater(len(tracks.triangles), 0)
                self.assertLess(tracks.triangles.max(), len(tracks.new_points))
        # Shared budget, not 181 new points independently at every scale.
        self.assertLessEqual(sum(len(t.new_points) for t in self.est.layer_tracks.values()), 181)
        self.assertLessEqual(len(self.est.layer_tracks[0].new_points), 61)
        for layer in (1, 2):
            self.assertLessEqual(len(self.est.layer_tracks[layer].new_points), 60)
        self.assertEqual(len(self.est.point_layers), len(self.est.velocity_point_mask))
        self.assertTrue(np.isfinite(self.est.camera_velocity).all())
        expected_vx = -4 * 10 / (self.est.foc_l * self.est.dt)
        self.assertAlmostEqual(self.est.camera_velocity[0], expected_vx, delta=.05)

    def test_overlays_use_layer_colors_and_leave_tracking_images_clean(self):
        self.run_layers()
        original = self.stereo(1)
        for camera in (0, 1):
            np.testing.assert_array_equal(self.est.buffer[-1].frames[camera], original[camera])
            gray = cv2.cvtColor(original[camera], cv2.COLOR_BGR2GRAY)
            for level in self.est.buffer[-1].pyramids[camera]:
                np.testing.assert_array_equal(level, gray)
                gray = cv2.pyrDown(gray)
            display = self.est.current_frame.frames[camera][90:]
            for layer, color in enumerate(LAYER_COLORS[:3]):
                if camera == 0 or self.est.depth_by_layer[layer].valid.any():
                    self.assertTrue(np.any(np.all(display == color, axis=2)))
        # Explicitly check edge pixels, independent of legends/point markers.
        for color in LAYER_COLORS[:3]:
            canvas = np.zeros((40, 40, 3), np.uint8)
            points = np.float32([[5, 5], [30, 5], [5, 30]])
            left, right = self.est.drawing(canvas.copy(), canvas.copy(), points, points,
                                           np.array([[0, 1], [1, 2], [0, 2]]), color=color)
            np.testing.assert_array_equal(left[5, 15], color)
            np.testing.assert_array_equal(right[5, 15], color)

    def test_loop_visits_every_layer_present_in_the_frame_pyramid(self):
        self.est.frame_processor.pyramid_levels = 4
        self.est.processing(self.stereo(0), 0)
        with patch.object(self.est, 'lucas_kanade_flow', wraps=self.est.lucas_kanade_flow) as track:
            self.est.processing(self.stereo(1), 1)
        self.assertEqual([call.kwargs['layer'] for call in track.call_args_list], [0, 1, 2, 3])
        self.assertEqual([call.kwargs['tracking_gray'][0].shape for call in track.call_args_list],
                         [(481, 641), (241, 321), (121, 161), (61, 81)])
        self.assertEqual(set(self.est.layer_tracks), {0, 1, 2, 3})

    def test_one_failed_layer_does_not_erase_other_layers(self):
        self.run_layers()
        actual_lk = cv2.calcOpticalFlowPyrLK
        def fail_half_resolution(previous, current, points, next_points, **options):
            if previous.shape == (241, 321):
                return None, None, None
            return actual_lk(previous, current, points, next_points, **options)
        with patch('swarm.compVision.visualAccEst.cv2.calcOpticalFlowPyrLK',
                   side_effect=fail_half_resolution):
            self.est.processing(self.stereo(2), 2)
        self.assertEqual(len(self.est.all_points), 3)
        self.assertEqual(self.est.all_points[1].shape, (0, 2))
        self.assertEqual(self.est.point_colors[1].shape, (0, 3))
        self.assertGreater(len(self.est.all_points[2]), 0)
        self.assertEqual(len(self.est.layer_tracks[1].new_points), 0)
        self.assertEqual(len(self.est.tracked_points_by_layer[1][0]), 0)
        self.assertEqual(self.est.layer_tracks[1].frame_idx, 2)
        for layer in (0, 2):
            self.assertGreater(len(self.est.layer_tracks[layer].new_points), 20)
            self.assertEqual(self.est.layer_tracks[layer].frame_idx, 2)
        self.assertNotIn(1, self.est.point_layers)

    def test_skipped_processing_does_not_reuse_seeds_from_wrong_frame(self):
        self.run_layers()
        self.est.update_frame(self.stereo(2), 2)
        self.est.update_frame(self.stereo(3), 3)
        original = self.est._replenish
        received_seed_lengths = []
        def capture(gray, points, camera, max_points, **kwargs):
            received_seed_lengths.append(len(points))
            return original(gray, points, camera, max_points, **kwargs)
        with patch.object(self.est, '_replenish', side_effect=capture):
            self.est.aply_flow()
        self.assertEqual(received_seed_lengths, [0, 0, 0])
        self.assertTrue(all(t.frame_idx == 3 for t in self.est.layer_tracks.values()))

    def test_duplicate_votes_suppressed_in_priority_order(self):
        points = np.float32([[10, 10], [50, 50], [11, 11], [90, 90], [np.nan, 0]])
        np.testing.assert_array_equal(unique_point_mask(points, radius=5),
                                      [True, True, False, True, False])
        self.run_layers()
        points = np.vstack([self.est.layer_tracks[l].new_points for l in self.est.layer_tracks])
        accepted = points[self.est.velocity_point_mask]
        distance = np.linalg.norm(accepted[:, None] - accepted[None, :], axis=2)
        np.fill_diagonal(distance, np.inf)
        self.assertGreaterEqual(distance.min(), self.est.cross_layer_min_distance)

    def test_feature_spacing_scales_with_native_resolution(self):
        self.est.processing(self.stereo(0), 0)
        from swarm.compVision.smallFeatures import replenish as actual_replenish
        with patch('swarm.compVision.visualAccEst.replenish', wraps=actual_replenish) as detect:
            self.est.processing(self.stereo(1), 1)
        self.assertEqual([call.kwargs['min_distance'] for call in detect.call_args_list],
                         [30, 15, 7.5])

    def test_layer_views_use_native_coordinates_and_isolated_colors(self):
        self.est.processing(self.stereo(0), 0)
        self.assertEqual([p.shape for p in self.est.all_points], [(0, 2)] * 3)
        self.assertEqual(len(self.est.layer_views), 3)
        self.est.processing(self.stereo(1), 1)
        for layer, view in enumerate(self.est.layer_views):
            gray = self.est.current_frame.pyramids[0][layer]
            self.assertEqual(view.shape[:2], (gray.shape[0] + 28, max(330, gray.shape[1])))
            points = np.rint(self.est.all_points[layer] / (2 ** layer)).astype(int)
            points = points[(points[:, 0] < gray.shape[1]) & (points[:, 1] < gray.shape[0])]
            self.assertGreater(len(points), 3)
            np.testing.assert_array_equal(view[points[:, 1] + 28, points[:, 0]],
                np.tile(LAYER_COLORS[layer], (len(points), 1)))
            # Other layer colors must not appear in this window.
            for other in set(range(3)) - {layer}:
                self.assertFalse(np.any(np.all(view == LAYER_COLORS[other], axis=2)))

    def test_simulation_opens_one_window_per_layer_and_reuses_them(self):
        import importlib.util
        from pathlib import Path
        from types import SimpleNamespace
        # Load only the display wrapper; camera/Genesis startup is not needed here.
        source = Path(__file__).resolve().parents[2] / 'droneClasses' / 'droneStruct.py'
        spec = importlib.util.spec_from_file_location('_layer_window_test', source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', {'genesis.utils.misc': SimpleNamespace(tensor_to_array=lambda x: x)}):
            spec.loader.exec_module(module)
        drone = module.DroneStruct.__new__(module.DroneStruct)
        drone.frame_processor = self.est
        drone._camera_display = None
        drone._layer_windows = set()
        drone._depth_window_ready = False
        self.run_layers()
        with patch.object(cv2, 'namedWindow') as create, patch.object(cv2, 'resizeWindow'), \
             patch.object(cv2, 'imshow') as show, patch.object(cv2, 'waitKey') as pump, \
             patch.object(cv2, 'destroyWindow') as close:
            drone.camera_show()
            self.assertEqual([c.args[0] for c in create.call_args_list],
                             ['Tracking layer 0', 'Tracking layer 1', 'Tracking layer 2', 'Stereo depth'])
            for layer, call in enumerate(show.call_args_list[:3]):
                self.assertIs(call.args[1], self.est.layer_views[layer])
            drone.camera_show()
            self.assertEqual(create.call_count, 4)
            self.assertEqual(pump.call_count, 2)
            self.est.layer_views = self.est.layer_views[:2]
            drone.camera_show()
            close.assert_called_once_with('Tracking layer 2')

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
