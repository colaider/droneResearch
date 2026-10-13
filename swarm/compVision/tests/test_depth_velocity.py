"""Tests for vertical velocity derived from consecutive stereo-depth frames."""

import unittest
from unittest.mock import patch

import numpy as np

from swarm.compVision.frame import Frame
from swarm.compVision.visualDepthEst import FrameDepth, estimate_vertical_velocity
from swarm.compVision.visualEst import VisulaEst


def frame_depth(depths):
    points = np.column_stack((
        np.arange(len(depths), dtype=float),
        np.zeros(len(depths)),
        np.asarray(depths, dtype=float),
    ))
    return FrameDepth([], points, np.empty((len(points), 3), np.uint8),
                      np.empty(len(points)), float(np.median(depths)), 1.0,
                      np.empty((0, 0), np.uint8))


class DepthVelocityTests(unittest.TestCase):
    def test_positive_up_velocity_from_increasing_ground_depth(self):
        previous = frame_depth([10.0, 10.1, 9.9])
        current = frame_depth([10.4, 10.5, 10.3])

        self.assertAlmostEqual(estimate_vertical_velocity(previous, current, 0.2), 2.0)

    def test_insufficient_depth_uses_nan_to_allow_pipeline_fallback(self):
        previous = frame_depth([10.0, np.nan])
        current = frame_depth([10.4, 10.5, 10.3])

        self.assertTrue(np.isnan(estimate_vertical_velocity(previous, current, 0.2)))
        self.assertTrue(np.isnan(estimate_vertical_velocity(None, current, 0.2)))

    def test_invalid_frame_interval_is_rejected(self):
        depth = frame_depth([10.0, 10.1, 9.9])

        for dt in (0.0, -0.1, np.nan, np.inf):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                estimate_vertical_velocity(depth, depth, dt)

    def test_velocity_fit_returns_complete_vector_with_depth_vertical_speed(self):
        estimator = VisulaEst((640, 480), 60)
        estimator.drone_vel[2] = 0.75
        estimator.drone_ang_vel = np.zeros(3)
        estimator.avg_ang_vel = np.zeros(3)
        estimator.avg_att = np.zeros(3)
        estimator.frame_size = (480, 640, 3)
        estimator.foc_l = 500.0
        estimator.dt = 0.1
        estimator.camera_saperation = 0.12
        points_xy = np.array([
            [150, 140], [270, 140], [150, 260],
            [270, 260], [390, 140], [390, 260],
        ], dtype=float)
        points = np.column_stack((points_xy, np.full(len(points_xy), 10.0)))
        flow = np.tile([2.0, -1.0], (len(points), 1))

        velocity = estimator.estimate_velocities(flow, points, vertical_velocity=2.0)

        self.assertEqual(velocity.shape, (4,))
        self.assertAlmostEqual(velocity[2], 2.0)
        np.testing.assert_array_equal(velocity, estimator.camera_velocity)

    def test_processing_publishes_depth_rate_and_falls_back_on_first_frame(self):
        estimator = VisulaEst((640, 480), 60)
        estimator.set_dt(0.2)
        estimator.drone_vel[2] = 0.75

        def update_frame(frame, idx):
            self.assertIsNone(frame)
            estimator.current_frame = Frame(idx=idx, pyramids=[[]])

        def estimate_velocities(flow, points, vertical_velocity=np.nan):
            self.assertEqual(flow.shape, (0, 2))
            self.assertEqual(points.shape, (0, 3))
            estimator.camera_velocity = np.array(
                [0.0, 0.0,
                 vertical_velocity if np.isfinite(vertical_velocity)
                 else estimator.drone_vel[2],
                 0.0]
            )
            return estimator.camera_velocity.copy()

        with (
            patch.object(estimator, "update_frame", side_effect=update_frame),
            patch.object(estimator, "aply_flow"),
            patch.object(estimator, "estimate_velocities", side_effect=estimate_velocities),
            patch(
                "swarm.compVision.visualEst.estimate_frame_depth",
                side_effect=[frame_depth([10, 10.1, 9.9]),
                             frame_depth([10.4, 10.5, 10.3])],
            ),
        ):
            estimator.processing(None, 0)
            self.assertEqual(estimator.camera_velocity[2], 0.75)
            estimator.processing(None, 1)

        self.assertAlmostEqual(estimator.camera_velocity[2], 2.0)


if __name__ == "__main__":
    unittest.main()
