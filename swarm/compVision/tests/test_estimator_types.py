"""Stereo input contracts and the simulation-to-estimator boundary."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from swarm.compVision import estimatorTypes as types
from swarm.compVision.frame import Frame, FrameProcessor
from swarm.compVision.smallFeatures import LayerTracks
from swarm.compVision.visualDepthEst import FrameDepth, StereoDepth
from swarm.compVision.visualEst import EnumFrame
from swarm.compVision.tests.estimator_factory import make_estimator


class EstimatorTypesTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((120, 160, 3), np.uint8)

    def pair(self, sequence=0, timestamp=1.0, right_timestamp=None):
        return types.StereoFrame(self.image, self.image.copy(), timestamp,
                                 timestamp if right_timestamp is None else right_timestamp,
                                 sequence)

    def test_existing_imports_reference_shared_types(self):
        for old, new in ((Frame, types.Frame), (EnumFrame, types.Frame),
                         (LayerTracks, types.LayerTracks),
                         (FrameDepth, types.FrameDepth), (StereoDepth, types.StereoDepth)):
            self.assertIs(old, new)

    def test_rejects_invalid_capture_contract(self):
        for image in (self.image.astype(float), self.image[:, :, 0], self.image[:0]):
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                types.StereoFrame(image, image, 1., 1., 0, pixel_format="BGR")
        with self.assertRaises(ValueError):
            types.StereoFrame(self.image, self.image[:10], 1., 1., 0)
        for timestamp in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                self.pair(timestamp=timestamp)
        for sequence in (-1, True, 0.5):
            with self.assertRaises(ValueError):
                self.pair(sequence=sequence)

    def test_preparation_preserves_capture_metadata_and_owns_images(self):
        pair = self.pair(7, 2., 2.001)
        processor = FrameProcessor(
            image_filters=(), filter_kernel_size=3, pyramid_levels=3,
            convert_to_grayscale=True, filters_enabled=True,
        )
        prepared = processor.process(pair)
        display = prepared.display_copy()
        self.assertEqual((prepared.idx, display.idx), (7, 7))
        self.assertEqual(display.timestamp, 2.)
        self.assertEqual(display.right_timestamp, 2.001)
        display.frames[0][:] = 255
        self.assertFalse(prepared.frames[0].any())
        prepared.frames[0][:] = 128
        self.assertFalse(pair.left.any())
        self.assertIsNone(processor.process([pair.left, pair.right], 8).timestamp)
        with self.assertRaises(ValueError):
            processor.process(pair, 8)

    def test_invalid_order_skew_and_size_leave_previous_state_intact(self):
        estimator = make_estimator((160, 120), 60)
        estimator.processing(self.pair())
        previous = estimator.current_frame
        previous_depth = estimator.current_depth
        previous_velocity = estimator.camera_velocity.copy()
        for pair in (self.pair(0, 1.1), self.pair(1, 1.),
                     self.pair(1, .9), self.pair(1, 1.1, 1.12),
                     types.StereoFrame(self.image[:10], self.image[:10], 1.1, 1.1, 1)):
            with self.subTest(pair=pair.sequence_id), self.assertRaises(ValueError):
                estimator.processing(pair)
            self.assertEqual(len(estimator.buffer), 1)
            self.assertIs(estimator.current_frame, previous)
            self.assertIs(estimator.current_depth, previous_depth)
            np.testing.assert_array_equal(estimator.camera_velocity, previous_velocity)
        # Left time increased, but right exposure moved backwards within allowed skew.
        estimator.processing(self.pair(1, 1.1, 1.104))
        with self.assertRaises(ValueError):
            estimator.processing(self.pair(2, 1.101, 1.103))

    def test_timestamped_pipeline_matches_legacy_and_exposes_state(self):
        size = (321, 241)
        texture = np.random.default_rng(4).integers(0, 256, (61, 81), dtype=np.uint8)
        image = cv2.cvtColor(cv2.resize(texture, size, interpolation=cv2.INTER_NEAREST),
                             cv2.COLOR_GRAY2BGR)
        typed, legacy = (make_estimator(size, 60) for _ in range(2))
        for est in (typed, legacy):
            est.image_filters = ()
            est.drone_pos[2] = 10
            est.disparity_direction = np.array([0., 1.])
            est.feature_min_distance = 15
        for index, timestamp in enumerate((1., 1.125, 1.375)):
            def shift(x, y):
                return cv2.warpAffine(image, np.float32([[1, 0, x], [0, 1, y]]), size,
                                      borderMode=cv2.BORDER_REFLECT)
            images = [shift(2 * index, 0), shift(2 * index, 4)]
            if index:
                legacy.set_dt((.125, .25)[index - 1])
            typed.processing(types.StereoFrame(*images, timestamp, timestamp, index))
            legacy.processing(images, index)
            np.testing.assert_allclose(typed.camera_velocity, legacy.camera_velocity)
            np.testing.assert_allclose(typed.depth_points, legacy.depth_points)
            self.assertEqual(typed.current_frame.timestamp, timestamp)
            self.assertEqual(typed.current_frame.idx, index)
            self.assertIs(typed.depth_points, typed.current_depth.points)
            self.assertIs(typed.depth_colors, typed.current_depth.colors)
            self.assertIs(typed.depth_sigma, typed.current_depth.sigma)
            if index == 0:
                self.assertFalse(typed.velocity_point_mask.any())
            if index == 1:
                self.assertAlmostEqual(typed.dt, .125)
                self.assertGreater(len(typed.depth_points), 5)
        self.assertAlmostEqual(typed.dt, .25)

    def test_processing_accepts_sensor_sample_with_frame(self):
        estimator = make_estimator((160, 120), 60)
        image = np.zeros((120, 160, 3), np.uint8)
        position = np.array([1.0, 2.0, 3.0])
        velocity = np.array([0.1, 0.2, 0.3])
        angular_velocity = np.array([0.01, 0.02, 0.03])
        attitude = np.array([0.1, 0.2, 0.3])

        estimator.processing(
            [image, image],
            0,
            drone_pos=position,
            drone_vel=velocity,
            drone_ang_vel=angular_velocity,
            imu_att=attitude,
        )

        np.testing.assert_array_equal(estimator.avg_pos, position)
        np.testing.assert_array_equal(estimator.avg_ang_vel, angular_velocity)
        np.testing.assert_array_equal(estimator.avg_att, attitude)
        np.testing.assert_array_equal(estimator.camera_velocity, [0.1, 0.2, 0.3, 0.03])
        self.assertEqual(len(estimator.buf_ang_vel), 0)

    def test_processing_rejects_partial_sensor_sample(self):
        estimator = make_estimator((160, 120), 60)
        image = np.zeros((120, 160, 3), np.uint8)
        with self.assertRaisesRegex(ValueError, "Provide drone_pos"):
            estimator.processing([image, image], 0, drone_ang_vel=np.zeros(3))

    def test_drone_passes_simulation_time_and_sequence_to_real_estimator(self):
        from swarm.droneClasses.droneStruct import DroneStruct
        drone = DroneStruct.__new__(DroneStruct)
        sim = SimpleNamespace(get_time=Mock(side_effect=[np.array([3.]), np.array([3.2])]))
        drone._drone = SimpleNamespace(sim=sim, get_pos=lambda: np.zeros(3),
                                       get_quat=lambda: np.array([1., 0., 0., 0.]))
        rgb = self.image.copy()
        rgb[:, :, 0] = 200
        drone.left_cam = SimpleNamespace(read=lambda: SimpleNamespace(rgb=rgb))
        drone.right_cam = SimpleNamespace(read=lambda: SimpleNamespace(rgb=rgb[None]))
        drone.steps_cam = 0
        drone.frame_processor = make_estimator((160, 120), 60)
        drone.point_map = Mock()
        with patch('swarm.droneClasses.droneStruct.tensor_to_array', side_effect=np.asarray):
            legacy = drone.get_two_frames()
            self.assertIsInstance(legacy, list)
            np.testing.assert_array_equal(legacy[0][0, 0], [0, 0, 200])
            with patch.object(drone.frame_processor, 'processing',
                              wraps=drone.frame_processor.processing) as process:
                drone.camera_step()
                drone.camera_step()
            pairs = [call.args[0] for call in process.call_args_list]
            self.assertTrue(all(isinstance(pair, types.StereoFrame) for pair in pairs))
            self.assertEqual([pair.sequence_id for pair in pairs], [0, 1])
            self.assertEqual([pair.timestamp for pair in pairs], [3., 3.2])
            self.assertTrue(all(pair.stereo_skew == 0 for pair in pairs))
            self.assertAlmostEqual(drone.frame_processor.dt, .2)
            self.assertEqual(drone.frame_processor.current_frame.idx, 1)
            self.assertEqual(drone.point_map.add.call_args.kwargs['frame_idx'], 1)
            explicit = drone.get_stereo_frame(timestamp=4.)
            self.assertEqual(explicit.timestamp, 4.)
            self.assertEqual(sim.get_time.call_count, 2)


if __name__ == '__main__':
    unittest.main()
