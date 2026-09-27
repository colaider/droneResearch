"""A synchronized downward (or forward) stereo pair attached to one Genesis drone."""
import numpy as np
import cv2
import genesis as gs
from genesis.utils.misc import tensor_to_array
from swarm.utilities.geometry import as_numpy, rotation, quat_to_rpy


class StereoRig:
    def __init__(self, scene, drone, calibration, front_offset_m=None,
                 height_offset_m=None, direction='down'):
        """Attach cameras BEFORE scene.build().

        Args:
            scene: Genesis Scene. drone: entity or DroneStruct wrapper.
            calibration: StereoCalibration with resolution, FOV and baseline.
            direction: 'down' (default) looks along body -Z, 'forward' along +X.
            front_offset_m: common body +X displacement, metres; defaults to
                0 for downward cameras, 0.22 for forward cameras.
            height_offset_m: common body +Z displacement, metres; defaults to
                -0.12 underneath the drone for down, +0.05 for forward.

        Left is at body +Y and right at body -Y. Image right is body -Y in
        BOTH modes, so the baseline stays horizontal in the images and
        d=u_left-u_right stays positive. Downward image up is body +X.
        The cameras are rigidly mounted: they tilt with the aircraft, but stay
        parallel to each other. These are not world-stabilized gimbals.
        """
        if direction not in ('down', 'forward'):
            raise ValueError("direction must be 'down' or 'forward'")
        self.direction = direction
        down = direction == 'down'
        if front_offset_m is None:
            front_offset_m = 0. if down else 0.22
        if height_offset_m is None:
            height_offset_m = -0.12 if down else 0.05
        self.drone, self.calibration = drone, calibration
        self.sensors = []
        self.body_from_camera = []  # OpenCV optical frame -> drone body frame.
        # OpenGL camera axes are right/up/backward; its viewing direction is -Z.
        R_gl = (np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]]) if down else
                np.array([[0.,0.,-1.],[-1.,0.,0.],[0.,1.,0.]]))
        R_cv = R_gl @ np.diag([1.,-1.,-1.])
        for side in (1., -1.):
            mount = np.array([front_offset_m, side*calibration.baseline_m/2, height_offset_m])
            T = np.eye(4); T[:3,:3] = R_gl; T[:3,3] = mount
            self.sensors.append(scene.add_sensor(gs.sensors.RasterizerCameraOptions(
                entity_idx=drone.idx, link_idx_local=0,
                res=(calibration.width, calibration.height), fov=calibration.vertical_fov_deg,
                offset_T=tuple(map(tuple,T)), near=0.05, far=120.)))
            T_cv = T.copy(); T_cv[:3,:3] = R_cv
            self.body_from_camera.append(T_cv)

    def read(self):
        """Return [left,right] uint8 BGR HxWx3 images from the SAME physics step.

        No scene.step() occurs between reads. Camera sensor rendering caches
        the current simulation step, ensuring a synchronized pair.
        """
        result = []
        for sensor in self.sensors:
            rgb = sensor.read().rgb
            if rgb.ndim == 4:
                rgb = rgb[0]
            result.append(cv2.cvtColor(tensor_to_array(rgb),cv2.COLOR_RGB2BGR))
        return result

    def world_from_left_camera(self):
        """Return (4,4) left optical -> world transform for exporting point clouds.

        This uses simulator pose ONLY to locate a reconstruction in the city.
        Pose is not used in disparity matching or triangulation.
        """
        T = np.eye(4)
        T[:3,:3] = rotation(quat_to_rpy(as_numpy(self.drone.get_quat()).reshape(4)))
        T[:3,3] = as_numpy(self.drone.get_pos()).reshape(3)
        return T @ self.body_from_camera[0]
