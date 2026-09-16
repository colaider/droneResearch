import cv2
from genesis.utils.misc import tensor_to_array


class DroneStruct:
    def __init__(self, drone, imu, left_cam, right_cam):
        self.imu = imu
        self._drone = drone
        self.left_cam = left_cam
        self.right_cam = right_cam
    
    def __getattr__(self, attr):
        return getattr(self._drone, attr)

    def cameraShow(self):
        """Display left and right cameras in OpenCV windows. Call once per scene.step()."""
        for name, cam in [('left', self.left_cam), ('right', self.right_cam)]:
            rgb = cam.read().rgb
            if rgb.ndim > 3:
                rgb = rgb[0]
            bgr = cv2.cvtColor(tensor_to_array(rgb), cv2.COLOR_RGB2BGR)
            cv2.imshow(name, bgr)
        cv2.waitKey(1)