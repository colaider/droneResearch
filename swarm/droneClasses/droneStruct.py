import cv2
from genesis.utils.misc import tensor_to_array
from swarm.compVision.visualAccEst import VisulaAcEst

class DroneStruct:
    def __init__(self, drone, imu, left_cam, right_cam, res = 0, fov = 0, camera_sapartion = 0):
        self.imu = imu
        self._drone = drone
        self.left_cam = left_cam
        self.right_cam = right_cam
        self.postpocessed_frames = []

        self.frame_processor = VisulaAcEst(res, fov)
        self.frame_processor.camera_saperation = camera_sapartion
        self.steps_cam = 0

    def __getattr__(self, attr):
        return getattr(self._drone, attr)

    def camera_show(self):
        for i, frame in enumerate(self.postpocessed_frames): cv2.imshow(str(i), frame)
        cv2.waitKey(1)
        

    def get_two_frames(self):
        lr_cam = []
        for name, cam in [('left', self.left_cam), ('right', self.right_cam)]:
            rgb = cam.read().rgb
            if rgb.ndim > 3:
                rgb = rgb[0]
            bgr = cv2.cvtColor(tensor_to_array(rgb), cv2.COLOR_RGB2BGR)
            lr_cam.append(bgr)
        return lr_cam        


    def camera_step(self):
        frames = self.frame_processor.processing(self.get_two_frames(), self.steps_cam)
        self.postpocessed_frames = frames 
        self.steps_cam += 1

    def set_camera_dt(self, dt):
        self.frame_processor.set_dt(dt)
       