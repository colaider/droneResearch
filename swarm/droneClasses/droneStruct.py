import cv2
from genesis.utils.misc import tensor_to_array
from swarm.compVision.visualAccEst import VisulaAcEst
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

        self.frame_processor = VisulaAcEst(cam_cfg.res, cam_cfg.fov)
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
        raw_frames = self.get_two_frames()
        frames = self.frame_processor.processing(raw_frames, self.steps_cam)
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

    