import unittest
from unittest.mock import Mock, patch

import numpy as np

from sensors.sensors import Sensors
from sensors.visual import StereoCameraConfig, StereoFrame


class SensorsVisualEstimatorTests(unittest.TestCase):
    def make_camera(self):
        return StereoCameraConfig(
            res=(64, 48),
            left_focal=50.0,
            right_focal=50.0,
            baseline=0.1,
            epipolar_direction=(-1.0, 0.0),
        )

    def make_sensor(self, **kwargs):
        return Sensors(
            camera_config=self.make_camera(),
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

    def test_sensor_update_processes_new_topic_frame(self):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        frame = StereoFrame(image, image.copy(), 1.0, 1.0, 0)
        receiver = Mock()
        receiver.get_frame.return_value = frame
        node = object()
        with patch(
            "communication.CameraStreamReceiver.CameraStreamReceiver",
            return_value=receiver,
        ) as receiver_type:
            sensors = self.make_sensor(node=node, camera_ns="/stereo")
            receiver_type.assert_called_once_with(node, ns="/stereo")
            sensors.update(np.array([0.0, 0.0, 9.81]), np.zeros(3), 0.0, 0.01)

        self.assertEqual(sensors.visual_estimator.current_frame.idx, 0)
        self.assertIs(sensors.latest_camera_frame, frame)
        receiver.get_frame.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
