import cv2
from genesis.utils.misc import tensor_to_array
from swarm.compVision.visualAccEst import VisulaAcEst
import numpy as np


class DroneStruct:
    def __init__(self, drone, imu, left_cam, right_cam, res = 0, fov = 0, camera_sapartion = 0):
        """Bundle a drone entity, its sensors, and the camera frame processor.

        Args:
            drone: Genesis drone entity providing state and motor-RPM methods.
            imu: Attached Genesis IMU sensor, read through imu.read().
            left_cam, right_cam: Attached camera sensors exposing read().rgb.
            res: Required (width, height) camera resolution in pixels.
            fov: Required vertical field of view in degrees, shared by the cameras.
            camera_sapartion: Camera baseline in metres along body y. The legacy
                argument spelling is preserved. Left is at -baseline/2, right +baseline/2.

        Despite the historical zero defaults, res/fov must contain real calibration
        values; CustomScene.add_drone supplies them.
        """
        self.imu = imu
        self._drone = drone
        self.left_cam = left_cam
        self.right_cam = right_cam
        self.postpocessed_frames = []

        self.frame_processor = VisulaAcEst(res, fov)
        self.frame_processor.camera_saperation = camera_sapartion
        self.steps_cam = 0

    def __getattr__(self, attr):
        """Forward attr (an attribute-name string) to the wrapped Genesis drone entity."""
        return getattr(self._drone, attr)

    def camera_show(self):
        """Display the latest annotated BGR images and process OpenCV window events."""
        for i, frame in enumerate(self.postpocessed_frames): cv2.imshow(str(i), frame)
        cv2.waitKey(1)
        

    def get_two_frames(self):
        """Read [left, right] images as BGR NumPy arrays of shape (height, width, 3).

        Converts Genesis RGB tensors to OpenCV BGR. If a batch dimension exists,
        only environment zero is selected: this wrapper is intended for one drone.
        """
        lr_cam = []
        for name, cam in [('left', self.left_cam), ('right', self.right_cam)]:
            rgb = cam.read().rgb
            if rgb.ndim > 3:
                rgb = rgb[0]
            bgr = cv2.cvtColor(tensor_to_array(rgb), cv2.COLOR_RGB2BGR)
            lr_cam.append(bgr)
        return lr_cam        


    def camera_step(self):
        """Read/process one image pair, cache annotations, and increment the camera counter.

        Set frame_processor's pose/velocity fields and call set_camera_dt() first.
        The processor stores velocity and validity on itself; this method returns None.
        """
        frames = self.frame_processor.processing(self.get_two_frames(), self.steps_cam)
        self.postpocessed_frames = frames 
        self.steps_cam += 1

    def set_camera_dt(self, dt):
        """Pass dt, the actual interval between camera samples in seconds, to the estimator."""
        self.frame_processor.set_dt(dt)

    