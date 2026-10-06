"""Pipeline contracts and stereo-time alignment without cameras or Genesis."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from realCameraTest import testRealCameras as runner
from swarm.compVision.frame import Frame
from swarm.compVision.largeFeatures import LargeFeatures
from swarm.compVision.smallFeatures import FeatureTracks
from swarm.compVision.visualAccEst import EnumFrame, VisulaAcEst


def texture(seed=31, shape=(480, 640)):
    gray = np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.5)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def translate(image, dx, dy):
    matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]),
                          borderMode=cv2.BORDER_REFLECT)


def estimator():
    est = VisulaAcEst((640, 480), 60)
    est.image_filters = ()
    est.foc_l = 500.0
    est.camera_saperation = 0.12
    est.disparity_direction = np.array([-1.0, 0.0])
    est.drone_pos = np.array([0.0, 0.0, 15.0])
    est.drone_vel = np.array([0.0, 0.0, 0.75])
    est.set_dt(0.1)
    return est


def known_tracks(dx=2.0, dy=1.0):
    points = np.array([[150, 140], [270, 140], [150, 260], [270, 260],
                       [390, 140], [390, 260]], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [1, 2, 3], [1, 3, 4], [3, 4, 5]])
    links = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3],
                      [1, 4], [3, 4], [3, 5], [4, 5]])
    return FeatureTracks(points, points + np.array([dx, dy], np.float32), triangles, links)


class PipelineTests(unittest.TestCase):
    def test_routes_coarse_regions_from_previous_frame_and_uses_real_current_stereo(self):
        est = estimator()
        base = texture()
        moved = translate(base, 2, 1)
        old_regions = LargeFeatures(regions=[(100, 100, 450, 300)])
        new_regions = LargeFeatures(regions=[(102, 101, 452, 301)])
        tracks = known_tracks()
        # Actual stereo disparity changes from 6 px to 10 px. Copying the old
        # right-image points plus temporal flow would incorrectly retain 6 px.
        with patch.object(est.frame_processor, 'process', wraps=est.frame_processor.process) as frame_stage, \
             patch.object(est.large_feature_detector, 'detect', side_effect=[old_regions, new_regions]) as coarse_stage, \
             patch.object(est.small_features, 'track', return_value=tracks) as fine_stage, \
             patch.object(est.depth_estimator, 'estimate', wraps=est.depth_estimator.estimate) as depth_stage:
            first = est.processing([base, translate(base, -6, 0)], 10)
            self.assertEqual(first.idx, 10)
            fine_stage.assert_not_called()
            depth_stage.assert_not_called()
            out = est.processing([moved, translate(moved, -10, 0)], 11)

        self.assertEqual(frame_stage.call_count, 2)
        self.assertEqual(coarse_stage.call_count, 2)
        self.assertIs(coarse_stage.call_args.args[0], est.buffer[-1].pyramids[0])
        self.assertIs(fine_stage.call_args.args[0], est.buffer[-2].gray_frames[0])
        self.assertIs(fine_stage.call_args.args[1], est.buffer[-1].gray_frames[0])
        self.assertEqual(fine_stage.call_args.args[2], old_regions.regions)
        self.assertEqual(depth_stage.call_count, 2)
        self.assertIs(depth_stage.call_args_list[0].args[1], est.buffer[-2].gray_frames[1])
        self.assertIs(depth_stage.call_args_list[1].args[1], est.buffer[-1].gray_frames[1])
        self.assertEqual(est.previous_depth.frame_idx, 10)
        self.assertEqual(est.current_depth.frame_idx, 11)
        self.assertEqual(est.depth_frame_idx, out.idx)
        self.assertGreater(est.depth_valid_frac, 0.8)
        self.assertAlmostEqual(est.previous_depth.median, 10.0, delta=0.12)
        self.assertAlmostEqual(est.depth_median, 6.0, delta=0.12)
        np.testing.assert_allclose(est.depth_points[:, :2], tracks.new_points[est.current_depth.valid])
        valid_both = est.previous_depth.valid & est.current_depth.valid
        np.testing.assert_allclose(
            est.current_depth.right_points[valid_both],
            tracks.new_points[valid_both] - np.array([10.0, 0.0]), atol=0.1)
        copied_old_right = est.previous_depth.right_points[valid_both] + np.array([2.0, 1.0])
        self.assertGreater(np.min(np.linalg.norm(
            est.current_depth.right_points[valid_both] - copied_old_right, axis=1)), 3.5)
        self.assertIsInstance(out, Frame)
        self.assertEqual(len(out.frames), 2)
        self.assertEqual(out.gray_frames[0].shape, (480, 640))
        # Display overlays must not get into subsequent tracking inputs.
        np.testing.assert_array_equal(est.buffer[-1].frames[0], moved)
        self.assertFalse(np.array_equal(out.frames[0], est.buffer[-1].frames[0]))

    def test_actual_fine_tracker_depth_and_velocity_run_together(self):
        est = estimator()
        base = texture(45)
        for idx in range(3):
            left = translate(base, 2 * idx, idx)
            est.push_sensors()
            out = est.processing([left, translate(left, -8, 0)], idx)
        self.assertGreater(len(est.tracked_points[0]), 3)
        self.assertGreater(len(est.depth_points), 3)
        self.assertEqual(est.depth_frame_idx, 2)
        self.assertAlmostEqual(est.depth_median, 7.5, delta=0.2)
        self.assertEqual(est.camera_velocity.shape, (4,))
        self.assertTrue(np.isfinite(est.camera_velocity).all())
        self.assertEqual(est.camera_velocity[2], est.drone_vel[2])
        self.assertEqual(out.idx, 2)

    def test_velocity_layout_keeps_depth_separate_from_sensor_vz(self):
        est = estimator()
        est.frame_size = (480, 640, 3)
        points = np.column_stack((known_tracks().old_points, np.full(6, 10.0)))
        flow = np.tile([2.0, -1.0], (6, 1))
        est.estimate_velocities(flow, points)
        # f=500 px, Z=10m, dt=0.1s: visual velocity is [-0.4,+0.2].
        np.testing.assert_allclose(est.camera_velocity, [-0.4, 0.2, 0.75, 0.0], atol=1e-9)
        self.assertEqual(est.estimated_height, 10.0)
        self.assertNotEqual(est.camera_velocity[2], est.estimated_height)

    def test_disabled_stereo_does_not_promote_height_to_a_stereo_measurement(self):
        est = estimator()
        est.use_stereo_depth = False
        image = texture()
        with patch.object(est.small_features, 'track', return_value=known_tracks()), \
             patch.object(est.depth_estimator, 'estimate', side_effect=AssertionError('Unexpected stereo matching')):
            est.processing([image, image], 0)
            est.processing([image, image], 1)
        self.assertEqual(est.depth_valid_frac, 0.0)
        self.assertTrue(np.isnan(est.depth_median))
        self.assertEqual(est.depth_points.shape, (0, 3))
        np.testing.assert_array_equal(est.current_depth.depths, np.full(6, 15.0))
        self.assertTrue(np.isnan(est.current_depth.right_points).all())
        self.assertEqual(est.camera_velocity[2], 0.75)

    def test_blank_frames_clear_depth_and_return_sensor_fallback(self):
        est = estimator()
        est.drone_vel = np.array([1.0, -2.0, 0.75])
        est.drone_ang_vel = np.array([0.0, 0.0, 0.125])
        blank = np.zeros((480, 640, 3), np.uint8)
        for idx in range(3):
            est.vertical_divergence = 0.4
            out = est.processing([blank, blank], idx)
            np.testing.assert_array_equal(est.camera_velocity, [1.0, -2.0, 0.75, 0.125])
            self.assertEqual(est.depth_frame_idx, idx)
            self.assertEqual(est.depth_points.shape, (0, 3))
            self.assertEqual(est.depth_valid_frac, 0.0)
            self.assertTrue(np.isnan(est.depth_median))
            self.assertEqual(est.vertical_divergence, 0.0)
            self.assertEqual(out.idx, idx)
        self.assertEqual(len(est.tracked_points[0]), 0)

    def test_tracking_loss_clears_previously_measured_depth(self):
        est = estimator()
        image = texture()
        with patch.object(est.small_features, 'track', side_effect=[known_tracks(), FeatureTracks.empty()]):
            est.processing([image, translate(image, -8, 0)], 0)
            est.processing([translate(image, 2, 1), translate(image, -6, 1)], 1)
            self.assertGreater(len(est.depth_points), 0)
            self.assertIsNotNone(est.current_depth)
            blank = np.zeros_like(image)
            est.processing([blank, blank], 2)
        self.assertIsNone(est.current_depth)
        self.assertIsNone(est.previous_depth)
        self.assertEqual(est.depth_points.shape, (0, 3))
        self.assertEqual(est.depth_valid_frac, 0.0)
        self.assertTrue(np.isnan(est.depth_median))
        self.assertEqual(est.depth_frame_idx, 2)

    def test_resize_requires_new_calibration_without_replacing_valid_state(self):
        est = estimator()
        image = texture()
        with patch.object(est.small_features, 'track', return_value=known_tracks()):
            est.processing([image, translate(image, -8, 0)], 0)
            est.processing([translate(image, 2, 1), translate(image, -6, 1)], 1)
        self.assertGreater(len(est.depth_points), 0)
        est.tracked_points[0] = known_tracks().new_points.copy()
        current = est.current_frame
        depth = est.current_depth
        points = est.depth_points.copy()
        small = texture(shape=(240, 320))
        with self.assertRaisesRegex(ValueError, '[Cc]alibrat'):
            est.processing([small, small], 2)
        self.assertIs(est.current_frame, current)
        self.assertIs(est.current_depth, depth)
        self.assertEqual(len(est.buffer), 2)
        self.assertEqual(est.frame_size, image.shape)
        np.testing.assert_array_equal(est.depth_points, points)
        self.assertEqual(est.depth_frame_idx, 1)

    def test_invalid_input_and_dt_do_not_replace_good_frame(self):
        est = estimator()
        image = texture()
        previous = est.processing([image, image], 4)
        for pair in ([image], [image, image[:200]], [image, image.astype(np.float32)]):
            with self.assertRaises(ValueError):
                est.processing(pair, 5)
            self.assertIs(est.current_frame, previous)
            self.assertEqual(len(est.buffer), 1)
        for dt in (0, -0.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                est.set_dt(dt)
        self.assertEqual(est.dt, 0.1)

    def test_real_camera_public_adapter_and_filter_settings(self):
        self.assertIs(EnumFrame, Frame)
        self.assertIs(runner.VisulaAcEst, VisulaAcEst)
        est = runner.make_estimator(500.0, (640, 480), filters=('median',), kernel_size=5)
        self.assertEqual(est.foc_l, 500.0)
        self.assertEqual(est.camera_saperation, runner.BASELINE)
        self.assertFalse(est.use_stereo_depth)
        self.assertEqual(est.frame_processor.image_filters, ('median',))
        self.assertEqual(est.frame_processor.filter_kernel_size, 5)
        runner.zero_drone_inputs(est)
        self.assertEqual(len(est.buf_ang_vel), 1)
        image = texture()
        out = est.processing([image, image], 0)
        np.testing.assert_array_equal(out.frames[0], cv2.medianBlur(image, 5))
        self.assertEqual(out.gray_frames[0].shape, (480, 640))
        self.assertEqual(out.idx, 0)


if __name__ == '__main__':
    unittest.main()
