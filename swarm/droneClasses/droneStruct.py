import cv2
import numpy as np
from swarm.compVision.terrainMap import PointMap

from swarm.compVision.visualEst import VisulaEst
from swarm.compVision.frame import prepare_camera_frame, compose_camera_display


class DroneStruct:
    def __init__(self, drone, imu, left_cam, right_cam, cam_cfg):
        self.imu = imu
        self._drone = drone
        self.left_cam = left_cam
        self.right_cam = right_cam
        self.cam_cfg = cam_cfg
        self.postpocessed_frames = []
        self._camera_display = None
        self._camera_window_ready = False
        self._layer_windows = set()
        self._depth_window_ready = False
        self.point_map = PointMap(cam_cfg)
        self.map_pose_position = np.zeros(3)

        self.frame_processor = VisulaEst(cam_cfg.res, cam_cfg.fov)
        # This simulation path uses parallel cameras with a common projection.
        separation = np.asarray(cam_cfg.camera_pos('right')) - cam_cfg.camera_pos('left')
        if (not np.isclose(cam_cfg.left_focal, cam_cfg.right_focal)
                or not np.allclose(cam_cfg.camera_axes('left'), cam_cfg.camera_axes('right'))
                or not np.isclose(np.linalg.norm(separation), cam_cfg.baseline)
                or abs(separation @ cam_cfg.camera_axes()[:, 2]) > 1e-8):
            raise ValueError('Stereo depth needs matching simulation projections and a transverse calibrated baseline')
        self.frame_processor.foc_l = cam_cfg.focal_px
        self.frame_processor.camera_saperation = cam_cfg.baseline
        self.frame_processor.disparity_direction = cam_cfg.disparity_direction()
        self.steps_cam = 0

    def __getattr__(self, attr):
        return getattr(self._drone, attr)

    def camera_show(self):
        if self._camera_display is not None:
            if not self._camera_window_ready:
                cv2.namedWindow("Stereo cameras", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                h, w = self._camera_display.shape[:2]
                cv2.resizeWindow("Stereo cameras", w, h)
                self._camera_window_ready = True
            cv2.imshow("Stereo cameras", self._camera_display)
        layer_views = self.frame_processor.layer_views
        for layer in self._layer_windows - set(range(len(layer_views))):
            cv2.destroyWindow(f'Tracking layer {layer}')
        self._layer_windows.intersection_update(range(len(layer_views)))
        for layer, image in enumerate(layer_views):
            title = f'Tracking layer {layer}'
            if layer not in self._layer_windows:
                cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                h, w = image.shape[:2]
                cv2.resizeWindow(title, w, h)
                self._layer_windows.add(layer)
            cv2.imshow(title, image)
        if self.frame_processor.depth_view is not None:
            if not self._depth_window_ready:
                cv2.namedWindow('Stereo depth', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                self._depth_window_ready = True
            cv2.imshow('Stereo depth', self.frame_processor.depth_view)
        cv2.waitKey(1)
        

    def get_two_frames(self):
        # Cameras are created in their mounted orientation (via `up`), so the frames come out
        # already oriented -- no software rotation needed here.
        lr_cam = []
        for cam in (self.left_cam, self.right_cam):
            rgb = cam.read().rgb
            if rgb.ndim > 3:
                rgb = rgb[0]
            lr_cam.append(prepare_camera_frame(tensor_to_array(rgb), color_order="RGB"))
        return lr_cam


    def camera_step(self):
        from genesis.utils.misc import tensor_to_array
        # Capture pose in the same simulation step as the stereo exposure.
        # Ground-truth pose is used ONLY to register/validate the map, never for stereo depth.
        self.map_pose_position = tensor_to_array(self._drone.get_pos()).reshape(-1, 3)[0].copy()
        map_quat = tensor_to_array(self._drone.get_quat()).reshape(-1, 4)[0].copy()
        raw_frames = self.get_two_frames()
        frames = self.frame_processor.processing(raw_frames, self.steps_cam)
        fp = self.frame_processor
        self.point_map.add(fp.depth_points, self.map_pose_position, map_quat,
                           colors=fp.depth_colors, sigma_depth=fp.depth_sigma, frame_idx=fp.depth_frame_idx)
        self.postpocessed_frames = frames.frames
        # Build only when a new camera sample arrives, not on every physics step.
        self._camera_display = self._compose_camera_display(self.postpocessed_frames, frames.gray_frames)
        self.steps_cam += 1
        return frames

    @staticmethod
    def _compose_camera_display(processed, tracking_gray):
        return compose_camera_display(processed, tracking_gray)

    def set_camera_dt(self, dt):
        self.frame_processor.set_dt(dt)

    