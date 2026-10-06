import cv2
from genesis.utils.misc import tensor_to_array
from swarm.compVision.visualAccEst import VisulaAcEst


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
            lr_cam.append(cv2.cvtColor(tensor_to_array(rgb), cv2.COLOR_RGB2BGR))
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
        """Filtered color above the exact grayscale inputs used for feature tracking."""
        if len(processed) < 2 or len(tracking_gray) < 2:
            return None
        h, w = processed[0].shape[:2]
        scale = min(640 / w, 450 / h, 1.0)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))

        def tile(frame, label, grayscale=False):
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            image = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            image = cv2.copyMakeBorder(image, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
            cv2.putText(image, label, (10, 19), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (235, 235, 235), 1, cv2.LINE_AA)
            return image

        top = cv2.hconcat([tile(processed[0], "Left camera"), tile(processed[1], "Right camera")])
        bottom = cv2.hconcat([tile(tracking_gray[0], "Left tracking input", True), tile(tracking_gray[1], "Right tracking input", True)])
        return cv2.vconcat([top, bottom])

    def set_camera_dt(self, dt):
        self.frame_processor.set_dt(dt)

    