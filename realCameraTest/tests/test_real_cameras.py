"""Headless USB runner and shared-estimator checks; no cameras or Genesis needed."""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from realCameraTest.visualAccEst import VisulaAcEst
from realCameraTest.display import compose_camera_display
from realCameraTest import testRealCameras as runner
from swarm.compVision.visualAccEst import VisulaAcEst as SwarmEstimator


class EstimatorTests(unittest.TestCase):
    def test_tracking_and_filters_are_shared(self):
        for name in ['preprocess_frame', '_apply_clahe', 'lucas_kanade_flow', '_replenish',
                     '_build_triangles', 'filter_by_neighbors', 'drawing', 'estimate_velocities']:
            self.assertIs(getattr(VisulaAcEst, name), getattr(SwarmEstimator, name))
        self.assertNotIn('genesis', sys.modules)

    def test_filter_preview_and_clean_history(self):
        frame = np.random.default_rng(4).integers(40, 150, (120, 160, 3), dtype=np.uint8)
        est = VisulaAcEst(500, 0.125, res=(160, 120))
        est.image_filters = ('gaussian', 'clahe')
        expected = est.preprocess_frame(frame)
        output = est.processing([frame, frame], 0)
        gray = cv2.cvtColor(expected, cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(output.gray_frames[0], gray)
        output.frames[0][:] = (0, 255, 0)
        np.testing.assert_array_equal(est.buffer[-1].frames[0], expected)
        grid = compose_camera_display(output.frames, output.gray_frames)
        np.testing.assert_array_equal(grid[176:296, :160], cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    def test_blank_frames_and_resolution_change(self):
        est = VisulaAcEst(500, 0.125)
        for i in range(3):
            result = est.processing([np.zeros((120, 160, 3), np.uint8)] * 2, i)
            self.assertEqual(result.gray_frames[0].shape, (120, 160))
            self.assertTrue(np.isfinite(est.camera_velocity).all())
        est.processing([np.zeros((100, 140, 3), np.uint8), np.zeros((80, 90, 3), np.uint8)], 3)
        self.assertEqual(len(est.buffer), 1)
        self.assertEqual(est.current_frame.gray_frames[1].shape, (100, 140))

    def test_usb_depth_and_no_command_filter(self):
        est = VisulaAcEst(500, 0.1, default_depth=2, disparity_dir=(2, 0))
        left = np.array([[10., 10., 2.], [20., 20., 2.], [30., 30., 2.]])
        right = left + [-5, 0, 0]
        right[2, 1] += 10  # Reject an epipolar mismatch.
        olds = [left, right]
        news = [left.copy(), right.copy()]
        est.trinagulate_altitude(news, None, olds, None, None)
        self.assertAlmostEqual(est.depth_valid_frac, 2/3)
        self.assertAlmostEqual(est.depth_median, 10)
        np.testing.assert_allclose(news[0][:, 2], 10)
        self.assertEqual(est.depth_points.shape, (2, 3))
        self.assertTrue(est.point_prediction_filtering(left, left + 50).all())
        est._take_sensor_window()
        np.testing.assert_array_equal(est.avg_ang_vel, np.zeros(3))
        self.assertEqual(est._current_height(), 2)

    def test_synthetic_camera_sequence(self):
        frame = cv2.GaussianBlur(np.random.default_rng(42).integers(60, 150, (480, 640, 3), dtype=np.uint8), (3, 3), 0)
        shift = lambda dx, dy: cv2.warpAffine(frame, np.float32([[1, 0, dx], [0, 1, dy]]), (640, 480))
        est = VisulaAcEst(500, 0.125, res=(640, 480))
        est.processing([frame, shift(-4, 0)], 0)
        out = est.processing([shift(2, 1), shift(-2, 1)], 1)
        self.assertGreater(len(est.tracked_points[0]), 10)
        self.assertGreater(est.depth_valid_frac, 0.5)
        self.assertTrue(np.isfinite(est.camera_velocity).all())
        self.assertEqual(out.gray_frames[0].shape, (480, 640))

    def test_invalid_calibration_and_timing(self):
        for kwargs in [{'focal_px': 0}, {'baseline': -1}, {'dt': 0}, {'disparity_dir': (0, 0)},
                       {'default_depth': np.nan}, {'epipolar_tol': 0}]:
            args = dict(focal_px=500, baseline=0.1)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                VisulaAcEst(**args)


class FakeCapture:
    def __init__(self, value, opened=True, grab_ok=True):
        self.frame = np.full((120, 160, 3), value, np.uint8)
        self.opened, self.grab_ok, self.released = opened, grab_ok, False

    def isOpened(self): return self.opened
    def grab(self): return self.grab_ok
    def retrieve(self): return True, self.frame.copy()
    def release(self): self.released = True


class RunnerTests(unittest.TestCase):
    def test_cli_filter_selection(self):
        self.assertIsNone(runner.parse_args([]).filters)
        self.assertEqual(runner.parse_args(['--filters']).filters, [])
        args = runner.parse_args(['2', '3', '--filters', 'gaussian', 'clahe', '--kernel-size', '5', '--no-rotate'])
        self.assertEqual((args.left, args.right), (2, 3))
        self.assertEqual(args.filters, ['gaussian', 'clahe'])
        self.assertEqual(args.kernel_size, 5)
        self.assertTrue(args.no_rotate)

    def test_capture_failure_releases_both_cameras(self):
        captures = [FakeCapture(20, opened=False), FakeCapture(40)]
        with patch.object(runner, 'open_camera', side_effect=captures), patch.object(cv2, 'destroyAllWindows'):
            self.assertEqual(runner.main(), 1)
        self.assertTrue(all(c.released for c in captures))

    def test_failed_grabs_can_quit(self):
        captures = [FakeCapture(20, grab_ok=False), FakeCapture(40)]
        with patch.object(runner, 'open_camera', side_effect=captures), patch.object(
                cv2, 'waitKey', return_value=ord('q')), patch.object(cv2, 'destroyAllWindows'):
            self.assertEqual(runner.main(), 0)
        self.assertTrue(all(c.released for c in captures))

    def test_single_window_swap_resets_calibration_and_tracking(self):
        captures = [FakeCapture(20), FakeCapture(40)]
        created = []

        class FakeEstimator:
            def __init__(self, **kwargs):
                self.config = kwargs
                self.image_filters = ('clahe',)
                self.camera_velocity = np.zeros(3)
                self.depth_median, self.depth_valid_frac = np.nan, 0
                self.indices = []
                created.append(self)
            def set_dt(self, dt): assert dt > 0
            def processing(self, frames, idx):
                self.indices.append(idx)
                return SimpleNamespace(frames=frames, gray_frames=[cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])

        with patch.object(runner, 'open_camera', side_effect=captures), patch.object(
                runner, 'VisulaAcEst', FakeEstimator), patch.object(runner, 'RES', (160, 120)), patch.object(
                runner, 'prepare_frame', wraps=runner.prepare_frame) as prepare, patch.object(
                cv2, 'namedWindow') as window, patch.object(cv2, 'resizeWindow'), patch.object(
                cv2, 'imshow') as show, patch.object(cv2, 'waitKey', side_effect=[ord('s'), ord('q')]), patch.object(
                cv2, 'destroyAllWindows'):
            self.assertEqual(runner.main(filters=('median',), kernel_size=5), 0)
        self.assertEqual(len(created), 2)
        self.assertEqual(created[0].config['focal_px'], runner.LEFT_FOCAL)
        self.assertEqual(created[1].config['focal_px'], runner.RIGHT_FOCAL)
        self.assertEqual(created[0].indices, [0])
        self.assertEqual(created[1].indices, [0])
        self.assertEqual(created[1].image_filters, ('median',))
        self.assertEqual(created[1].filter_kernel_size, 5)
        rotations = [c.args[1] for c in prepare.call_args_list]
        self.assertEqual(rotations, [cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE,
                                     cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE])
        window.assert_called_once()
        self.assertEqual(show.call_count, 2)
        self.assertTrue(all(c.released for c in captures))


if __name__ == '__main__':
    unittest.main()
