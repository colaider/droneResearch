"""Headless checks: the runner drives the REAL swarm estimator with zeroed drone inputs.

No physical cameras and no Genesis are needed.
"""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from realCameraTest import testRealCameras as runner
from realCameraTest.display import compose_camera_display
from swarm.compVision.visualAccEst import VisulaAcEst as SwarmEstimator


def textured(seed=0):
    rng = np.random.default_rng(seed)
    base = cv2.resize(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8), (640, 480),
                      interpolation=cv2.INTER_NEAREST)
    return base


class DirectConnectionTests(unittest.TestCase):
    def test_runner_uses_the_real_estimator_not_a_copy(self):
        # No duplicate estimator in the folder: the runner references the swarm class itself.
        self.assertIs(runner.VisulaAcEst, SwarmEstimator)

    def test_make_estimator_sets_calibration(self):
        est = runner.make_estimator(runner.LEFT_FOCAL, (640, 480), filters=('median',), kernel_size=5)
        self.assertIsInstance(est, SwarmEstimator)
        self.assertAlmostEqual(est.foc_l, runner.LEFT_FOCAL)
        self.assertAlmostEqual(est.camera_saperation, runner.BASELINE)
        self.assertEqual(est.image_filters, ('median',))
        self.assertEqual(est.filter_kernel_size, 5)

    def test_zero_drone_inputs_are_all_zero(self):
        est = runner.make_estimator(runner.LEFT_FOCAL, (640, 480))
        runner.zero_drone_inputs(est)
        for name, size in [('drone_pos', 3), ('drone_vel', 3), ('drone_ang_vel', 3),
                           ('imu_att', 3), ('previous_cmd_vel', 4), ('drone_ang', 2)]:
            vec = getattr(est, name)
            self.assertEqual(vec.shape, (size,))
            np.testing.assert_array_equal(vec, np.zeros(size))
        # push_sensors was called, so the sensor window has one sample queued.
        self.assertEqual(len(est.buf_ang_vel), 1)

    def test_drives_real_estimator_without_crash(self):
        # Translating textured stereo pair -> the real pipeline should run and stay finite.
        base = textured(1)
        canvas = cv2.copyMakeBorder(base, 60, 60, 60, 60, cv2.BORDER_REFLECT)
        est = runner.make_estimator(runner.LEFT_FOCAL, (640, 480))
        out = None
        for t in range(5):
            x, y = 40 + t * 4, 40 + t * 2
            left = canvas[y:y + 480, x:x + 640].copy()
            right = canvas[y:y + 480, x + 8:x + 8 + 640].copy()
            runner.zero_drone_inputs(est)
            out = est.processing([left, right], t)
        self.assertTrue(np.isfinite(est.camera_velocity).all())
        self.assertEqual(out.gray_frames[0].shape, (480, 640))
        self.assertEqual(len(out.frames), 2)
        self.assertNotIn('genesis', sys.modules)

    def test_display_composes_four_panels(self):
        est = runner.make_estimator(runner.LEFT_FOCAL, (160, 120))
        frame = textured(2)[:120, :160]
        out = est.processing([frame, frame], 0)
        grid = compose_camera_display(out.frames, out.gray_frames)
        self.assertIsNotNone(grid)
        self.assertEqual(grid.ndim, 3)


class FakeCapture:
    def __init__(self, value, opened=True, grab_ok=True):
        self.frame = np.full((120, 160, 3), value, np.uint8)
        self.opened, self.grab_ok, self.released = opened, grab_ok, False

    def isOpened(self): return self.opened
    def grab(self): return self.grab_ok
    def retrieve(self): return True, self.frame.copy()
    def release(self): self.released = True


class FakeEstimator:
    """Stand-in matching how the runner uses the estimator (make_estimator sets attrs after init)."""
    def __init__(self, res, fov):
        self.res, self.fov = res, fov
        self.image_filters = ('clahe',)
        self.filter_kernel_size = 3
        self.camera_velocity = np.zeros(3)
        self.depth_median, self.depth_valid_frac = np.nan, 0.0
        self.indices, self.pushes = [], 0

    def push_sensors(self): self.pushes += 1
    def set_dt(self, dt): assert dt > 0

    def processing(self, frames, idx):
        self.indices.append(idx)
        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        return SimpleNamespace(frames=frames, gray_frames=gray)


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

    def test_swap_resets_estimator_and_rotations(self):
        captures = [FakeCapture(20), FakeCapture(40)]
        created = []
        orig_make = runner.make_estimator

        def tracking_make(focal, res, filters=None, kernel_size=None):
            est = orig_make(focal, res, filters, kernel_size)  # exercises real make_estimator path
            est.focal_used = focal
            created.append(est)
            return est

        with patch.object(runner, 'open_camera', side_effect=captures), patch.object(
                runner, 'VisulaAcEst', FakeEstimator), patch.object(runner, 'make_estimator', tracking_make), \
                patch.object(runner, 'RES', (160, 120)), patch.object(
                runner, 'prepare_frame', wraps=runner.prepare_frame) as prepare, patch.object(
                cv2, 'namedWindow') as window, patch.object(cv2, 'resizeWindow'), patch.object(
                cv2, 'imshow') as show, patch.object(cv2, 'waitKey', side_effect=[ord('s'), ord('q')]), \
                patch.object(cv2, 'destroyAllWindows'):
            self.assertEqual(runner.main(filters=('median',), kernel_size=5), 0)

        self.assertEqual(len(created), 2)                       # swap builds a fresh estimator
        self.assertEqual(created[0].focal_used, runner.LEFT_FOCAL)
        self.assertEqual(created[1].focal_used, runner.RIGHT_FOCAL)
        self.assertEqual(created[0].indices, [0])               # tracking restarts at idx 0
        self.assertEqual(created[1].indices, [0])
        self.assertGreaterEqual(created[0].pushes, 1)           # drone inputs pushed each frame
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
