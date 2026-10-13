"""Measured sparse depth and 3D registration, without camera hardware or GUI."""
import unittest
from unittest.mock import patch
import cv2
import numpy as np

from swarm.compVision.visualDepthEst import SparseStereoDepth
from swarm.compVision.visualEst import VisulaEst
from swarm.mapping.terrainMap import PointMap
from swarm.config.cameraConfig import StereoCameraConfig


def texture():
    a = np.random.default_rng(37).integers(0, 256, (480, 640), np.uint8)
    return cv2.GaussianBlur(a, (3, 3), .6)


def translate(im, dx, dy=0):
    return cv2.warpAffine(im, np.float32([[1, 0, dx], [0, 1, dy]]), (640, 480))


class StereoTests(unittest.TestCase):
    def test_processing_fits_velocity_before_independent_stereo_depth(self):
        est = VisulaEst((640, 480), 60)
        est.image_filters = ()
        est.foc_l, est.camera_saperation = 400.0, .12
        est.disparity_direction = np.array([-1.0, 0.0])
        est.drone_pos[2] = 99.0
        est.drone_vel[2] = .75
        base = cv2.cvtColor(texture(), cv2.COLOR_GRAY2BGR)
        est.processing([base, translate(base, -8)], 0)

        current = translate(base, 2, 1)
        with patch.object(est, 'estimate_velocities', wraps=est.estimate_velocities) as fit:
            est.processing([current, translate(current, -8)], 1)

        fit.assert_called_once()
        flow, points = fit.call_args.args
        self.assertGreaterEqual(len(flow), 2)
        np.testing.assert_allclose(points[:, 2], 99.0)
        self.assertGreater(len(est.depth_points), 2)
        np.testing.assert_allclose(est.depth_points[:, 2], 6.0, atol=.15)
        self.assertEqual(est.camera_velocity[2], .75)

    def test_velocity_uses_supplied_height_regardless_of_stereo_availability(self):
        est = VisulaEst((640, 480), 60)
        est.image_filters = ()
        est.foc_l, est.camera_saperation = 400.0, .12
        est.disparity_direction = np.array([-1.0, 0.0])
        est.drone_pos[2] = 99.0
        base = cv2.cvtColor(texture(), cv2.COLOR_GRAY2BGR)
        blank_right = np.zeros_like(base)
        est.processing([base, blank_right], 0)

        current = translate(base, 2, 1)
        with patch.object(est, 'estimate_velocities', wraps=est.estimate_velocities) as fit:
            est.processing([current, blank_right], 1)

        fit.assert_called_once()
        _, points = fit.call_args.args
        self.assertGreaterEqual(len(points), 2)
        np.testing.assert_allclose(points[:, 2], 99.0)
        self.assertEqual(est.depth_valid_frac, 0.0)

    def test_two_surface_depths_at_three_scales(self):
        left = texture()
        right = translate(left, -8)
        right[240:] = translate(left, -16)[240:]
        points = np.float32([[x, y] for y in (140, 180, 300, 340) for x in (220, 300, 380, 460)])
        truth = np.where(points[:, 1] < 240, 6., 3.)
        matcher = SparseStereoDepth()
        for level in range(3):
            out = matcher.estimate(left, right, points, focal_px=400, baseline=.12,
                                   direction=[-1, 0], pixel_scale=2**level)
            with self.subTest(layer=level):
                self.assertGreaterEqual(out.valid.mean(), .8)
                np.testing.assert_allclose(out.depths[out.valid], truth[out.valid], rtol=.015)
                np.testing.assert_allclose(out.right_points[out.valid, 1], points[out.valid, 1], atol=.1)
            left, right = cv2.pyrDown(left), cv2.pyrDown(right)

    def test_vertical_baseline_and_no_groundtruth_height_input(self):
        left = texture(); points = np.float32([[200, 180], [300, 240], [400, 300]])
        out = SparseStereoDepth().estimate(left, translate(left, 0, 12), points,
                                          focal_px=400, baseline=.12, direction=[0, 1])
        self.assertTrue(out.valid.all())
        np.testing.assert_allclose(out.depths, 4, rtol=.005)

    def test_occluded_flat_wrong_sign_and_ambiguous_patches_are_not_depth(self):
        left = texture(); points = np.float32([[200, 180], [300, 240], [400, 300]])
        matcher = SparseStereoDepth()
        for right in (np.zeros_like(left), translate(left, 12), left):
            with self.subTest(kind=right.sum()):
                out = matcher.estimate(left, right, points, focal_px=400, baseline=.12, direction=[-1, 0])
                self.assertFalse(out.valid.any())
                self.assertTrue(np.isnan(out.depths).all())
        tile = np.random.default_rng(3).integers(0, 256, (24, 24), np.uint8)
        repeated = np.tile(tile, (20, 27))[:, :640].copy()
        out = matcher.estimate(repeated, translate(repeated, -12), points,
                               focal_px=400, baseline=.12, direction=[-1, 0])
        self.assertFalse(out.valid.any())

    def test_invalid_rows_and_empty_inputs_keep_alignment(self):
        left = texture(); right = translate(left, -8)
        points = np.float32([[200, 180], [np.nan, 180], [-1, 10], [639, 479]])
        out = SparseStereoDepth().estimate(left, right, points, focal_px=400, baseline=.12, direction=[-1, 0])
        np.testing.assert_array_equal(out.valid, [True, False, False, False])
        self.assertTrue(np.isnan(out.right_points[1:]).all())
        with patch('swarm.compVision.visualDepthEst.cv2.calcOpticalFlowPyrLK') as lk:
            out = SparseStereoDepth().estimate(left, right, [], focal_px=400, baseline=.12, direction=[-1, 0])
        lk.assert_not_called(); self.assertEqual(out.depths.shape, (0,))

    def test_failed_reverse_tracking_rejects_depth(self):
        left=texture(); right=translate(left,-8)
        real=cv2.calcOpticalFlowPyrLK; calls=[]
        def fail_reverse(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs) if len(calls)==1 else (None,None,None)
        with patch('swarm.compVision.visualDepthEst.cv2.calcOpticalFlowPyrLK', side_effect=fail_reverse):
            out=SparseStereoDepth().estimate(left,right,[[200,180]],focal_px=400,baseline=.12,direction=[-1,0])
        self.assertFalse(out.valid.any())

    def test_pipeline_uses_current_stereo_when_disparity_changes_and_clears_disabled_depth(self):
        base = cv2.cvtColor(texture(), cv2.COLOR_GRAY2BGR)
        est = VisulaEst((640, 480), 60); est.image_filters=()
        est.frame_processor.pyramid_levels=3; est.feature_min_distance=20
        est.foc_l=400; est.camera_saperation=.12; est.disparity_direction=np.array([-1.,0.])
        est.drone_pos[2]=99 # must not become measured depth
        est.processing([base, translate(base,-8)],0)
        current=translate(base,4)
        est.processing([current,translate(current,-16)],1)
        self.assertGreater(len(est.depth_points),20)
        self.assertAlmostEqual(est.depth_median,3.,delta=.03)
        self.assertEqual(est.depth_frame_idx,1)
        self.assertEqual(len(est.depth_colors),len(est.depth_points))
        for points, depth in zip(est.all_points,est.depth_by_layer):
            np.testing.assert_allclose(depth.right_points[depth.valid]-points[depth.valid],
                                      np.tile([-16,0],(depth.valid.sum(),1)),atol=.15)
        est.use_stereo_depth=False
        with patch.object(est.stereo,'estimate',side_effect=AssertionError('Stereo disabled')):
            est.processing([current,current],2)
        self.assertEqual(est.depth_points.shape,(0,3));self.assertTrue(np.isnan(est.depth_median))
        self.assertEqual(est.depth_frame_idx,2)

class PointMapTests(unittest.TestCase):
    def setUp(self):
        self.cfg=StereoCameraConfig(res=(640,480),left_focal=400,right_focal=400,baseline=.12,mount_pos=(.06,0,0))

    def add(self, m, points, idx=0, position=(1,2,5), quat=(1,0,0,0)):
        return m.add(points,position,quat,colors=np.tile([255,0,0],(len(points),1)),
                     sigma_depth=np.full(len(points),.05),frame_idx=idx)

    def test_camera_axes_match_explicit_rendering_orientation(self):
        np.testing.assert_allclose(self.cfg.camera_axes(),np.eye(3))
        np.testing.assert_allclose(self.cfg.disparity_direction(),[-1,0])
        options=self.cfg.camera_options('left')
        np.testing.assert_allclose(options['up'],[0,1,0])

    def test_ground_and_roof_at_same_xy_remain_separate(self):
        m=PointMap(self.cfg)
        self.add(m,[[320,240,5],[320,240,3]])
        points=m.points();self.assertEqual(len(points),2)
        np.testing.assert_allclose(sorted(points[:,2]),[0,2])
        np.testing.assert_allclose(points[:,:2],[[.94,2],[.94,2]])
        np.testing.assert_allclose(m.colors(),[[0,0,1],[0,0,1]])

    def test_registration_respects_body_rotation_and_camera_offset(self):
        m=PointMap(self.cfg)
        self.add(m,[[420,240,4]],quat=[np.sqrt(.5),0,0,np.sqrt(.5)])
        np.testing.assert_allclose(m.points(),[[1,2.94,1]],atol=1e-9)

    def test_one_update_per_frame_invalid_points_and_bounded_storage(self):
        m=PointMap(self.cfg,max_voxels=2)
        self.add(m,[[320,240,5],[320,240,np.nan],[320,240,-1]],idx=0)
        initial=m.points().copy()
        self.add(m,[[320,240,2]],idx=0)
        np.testing.assert_array_equal(m.points(),initial)
        self.add(m,[[320,240,4],[320,240,3],[320,240,2]],idx=1)
        self.assertEqual(len(m.points()),2)
        self.assertTrue(np.isfinite(m.points()).all())
        with self.assertRaises(ValueError):self.add(m,[[320,240,5]],idx=2,quat=[0,0,0,0])

    def test_simulation_camera_step_registers_current_depth_once_with_capture_pose(self):
        import importlib.util
        from pathlib import Path
        from types import SimpleNamespace
        source=Path(__file__).resolve().parents[2]/'droneClasses'/'droneStruct.py'
        spec=importlib.util.spec_from_file_location('_depth_map_drone_test',source)
        module=importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules',{'genesis.utils.misc':SimpleNamespace(tensor_to_array=np.asarray)}):
            spec.loader.exec_module(module)
        pose=np.array([1.,2.,5.])
        body=SimpleNamespace(get_pos=lambda:pose.copy(),get_quat=lambda:np.array([1.,0.,0.,0.]))
        drone=module.DroneStruct(body,None,None,None,self.cfg)
        drone.frame_processor.image_filters=()
        image=cv2.cvtColor(texture(),cv2.COLOR_GRAY2BGR)
        with patch.object(drone,'get_two_frames',return_value=[image,translate(image,-16)]):
            drone.camera_step()
            with patch.object(drone.point_map,'add',wraps=drone.point_map.add) as add:
                drone.camera_step()
        add.assert_called_once()
        np.testing.assert_array_equal(add.call_args.args[1],pose)
        self.assertEqual(add.call_args.kwargs['frame_idx'],1)
        self.assertGreater(len(drone.point_map.points()),20)
        # Known depth 3m at camera/body height 5m gives a plane at world Z=2m.
        np.testing.assert_allclose(drone.point_map.points()[:,2],2.,atol=.05)

    def test_multilayer_duplicates_do_not_multiply_frame_weight(self):
        m=PointMap(self.cfg)
        self.add(m,[[320,240,5]]*3)
        self.assertEqual(len(m.points()),1)
        self.assertAlmostEqual(next(iter(m._voxels.values()))[1],400)

if __name__=='__main__':unittest.main()
