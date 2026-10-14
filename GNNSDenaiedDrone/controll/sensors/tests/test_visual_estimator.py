import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sensors.sensors import Sensors
from sensors.visual import StereoCameraConfig, StereoFrame
from sensors.visual.config.cameraConfig import STEREO_CAM


class SensorsVisualEstimatorTests(unittest.TestCase):
    def make_camera(self, camera_offset_from_yaw_center=(0.0, 0.0)):
        return StereoCameraConfig(
            res=(64, 48),
            left_focal=50.0,
            right_focal=50.0,
            baseline=0.1,
            camera_offset_from_yaw_center=camera_offset_from_yaw_center,
            epipolar_direction=(-1.0, 0.0),
        )

    def make_sensor(self, **kwargs):
        camera_offset_from_yaw_center = kwargs.pop(
            "camera_offset_from_yaw_center", (0.0, 0.0)
        )
        return Sensors(
            camera_config=self.make_camera(camera_offset_from_yaw_center),
            image_filters=(),
            convert_to_grayscale=True,
            filters_enabled=True,
            filter_kernel_size=3,
            pyramid_levels=1,
            max_stereo_skew_s=0.005,
            **kwargs,
        )

    def test_sensor_buffers_imu_and_processes_stereo_frame(self):
        sensors = self.make_sensor()
        estimator = sensors.visual_estimator
        sensors.update(np.array([0.0, 0.0, 9.81]), np.zeros(3), 0.0, 0.01)
        sensors.update(np.array([0.0, 0.0, 9.81]), np.zeros(3), 0.0, 0.01)

        image = np.zeros((48, 64, 3), dtype=np.uint8)
        frame = StereoFrame(image, image.copy(), 1.0, 1.0, 0)
        processed = sensors.process_visual_frame(frame)

        self.assertIs(processed, estimator.current_frame)
        np.testing.assert_array_equal(estimator.avg_ang_vel, np.zeros(3))
        np.testing.assert_array_equal(estimator.avg_att, np.zeros(3))
        self.assertEqual(len(estimator.buf_ang_vel), 0)
        np.testing.assert_array_equal(sensors.get_visual_velocity(), np.zeros(4))

    def test_sensor_constructs_visual_estimator_by_default(self):
        self.assertIsNotNone(self.make_sensor().visual_estimator)

    def test_yaw_compensation_uses_configured_camera_offset(self):
        estimator = self.make_sensor(
            camera_offset_from_yaw_center=(0.02, 0.0)
        ).visual_estimator
        estimator.frame_size = (48, 64)
        estimator.dt = 1.0
        estimator.avg_ang_vel = np.zeros(3)
        estimator.avg_att = np.zeros(3)
        estimator.drone_vel = np.zeros(3)
        points = np.array([[32.0, 24.0, 2.0], [32.0, 24.0, 2.0]])

        with patch(
            "sensors.visual.visualVelEst.np.linalg.lstsq",
            return_value=(np.zeros(3), None, None, None),
        ) as least_squares:
            estimator.estimate_velocities(np.zeros((2, 2)), points)

        matrix = least_squares.call_args.args[0]
        self.assertAlmostEqual(matrix[1, 2], 0.0)
        self.assertAlmostEqual(matrix[3, 2], 0.02)
        np.testing.assert_array_equal(STEREO_CAM.camera_offset_from_yaw_center, (0.02, 0.0))

    def test_sensor_update_processes_new_topic_frame(self):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        frame = StereoFrame(image, image.copy(), 1.0, 1.0, 0)
        receiver = Mock()
        receiver.get_frame.return_value = frame
        node = object()
        with patch(
            "communication.CameraStreamReceiver.CameraStreamReceiver",
            return_value=receiver,
        ) as receiver_type, patch(
            "communication.VisualFrameBridge.VisualFrameBridge",
            return_value=Mock(),
        ) as bridge_type:
            sensors = self.make_sensor(node=node, camera_ns="/stereo")
            receiver_type.assert_called_once_with(node, ns="/stereo")
            bridge = sensors.visual_frame_bridge
            bridge_type.assert_called_once_with(node)
            sensors.update(np.array([0.0, 0.0, 9.81]), np.zeros(3), 0.0, 0.01)

        self.assertEqual(sensors.visual_estimator.current_frame.idx, 0)
        self.assertIs(sensors.latest_camera_frame, frame)
        receiver.get_frame.assert_called_once_with()
        bridge.publish.assert_called_once_with(
            sensors.visual_estimator.current_frame,
            sensors.visual_estimator.layer_tracks,
            sensors.visual_estimator.depth_by_layer,
        )

    def test_fused_position_integrates_visual_velocity_at_ten_percent(self):
        sensors = self.make_sensor()
        estimator = sensors.visual_estimator
        estimator.velocity_point_mask = np.array([True, True])
        estimator.camera_velocity = np.array([2.0, 0.0, 0.0, 0.0])
        estimator.depth_velocity_z = np.nan
        estimator.dt = 0.5
        estimator.processing = Mock(
            side_effect=lambda frame, idx: SimpleNamespace(
                idx=frame.sequence_id if idx is None else idx
            )
        )
        image = np.zeros((48, 64, 3), dtype=np.uint8)

        sensors.process_visual_frame(
            StereoFrame(image, image.copy(), 1.0, 1.0, 0)
        )
        np.testing.assert_allclose(sensors.get_visual_position(), [1.0, 0.0, 0.0])
        np.testing.assert_allclose(sensors.get_fused_position(), [0.1, 0.0, 0.0])

        sensors.imu._pos[0] = 2.0
        next_image = image.copy()
        sensors.process_visual_frame(
            StereoFrame(next_image, next_image.copy(), 1.5, 1.5, 1)
        )
        np.testing.assert_allclose(sensors.get_visual_position(), [2.0, 0.0, 0.0])
        np.testing.assert_allclose(sensors.get_fused_position(), [2.0, 0.0, 0.0])

    def test_fused_position_ignores_visual_fallback_without_valid_tracks(self):
        sensors = self.make_sensor()
        estimator = sensors.visual_estimator
        estimator.velocity_point_mask = np.zeros(0, dtype=bool)
        estimator.camera_velocity = np.array([5.0, 4.0, 3.0, 2.0])
        estimator.dt = 0.5
        estimator.processing = Mock(
            side_effect=lambda frame, idx: SimpleNamespace(
                idx=frame.sequence_id if idx is None else idx
            )
        )
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        sensors.imu._pos[:] = [3.0, 4.0, 5.0]

        sensors.process_visual_frame(
            StereoFrame(image, image.copy(), 1.0, 1.0, 0)
        )

        self.assertIsNone(sensors.get_visual_position())
        np.testing.assert_array_equal(sensors.get_fused_position(), [3.0, 4.0, 5.0])


if __name__ == "__main__":
    unittest.main()
