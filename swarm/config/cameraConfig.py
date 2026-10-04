from dataclasses import dataclass
from typing import Tuple
import numpy as np


_AXES = {'x': (1.0, 0.0, 0.0), 'y': (0.0, 1.0, 0.0), 'z': (0.0, 0.0, 1.0)}


@dataclass
class StereoCameraConfig:
    """Stereo rig matched to camera_params.txt.

    LEFT is the dominant camera: optical flow (velocity) is estimated on the left image, the
    right image is only used for coarse stereo correspondence. Each camera uses its own mean
    focal (fx+fy)/2 -> vertical FOV, principal point at the frame centre. The cameras look down
    and are rolled 90 deg in opposite senses; that roll is baked straight into their `up`
    vectors (Genesis convention), so no software rotation is needed.
    """
    res: Tuple[int, int] = (1920, 1080)        # (width, height) px -- sensor_size (1080, 1920) h:w
    near: float = 0.01
    far: float = 100.0

    # per-camera mean focal length (px) = (fx + fy) / 2 from the TXT
    left_focal: float = 1500     # Camera 1 (Left - Long)
    right_focal: float = 1500   # Camera 2 (Right - Short)

    # looking down, rolled 90 deg: left 90 CCW -> up = -Y, right 90 CW -> up = +Y
    left_up: Tuple[float, float, float] = (0.0,    0.0, 0.0)
    right_up: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # rig geometry (drone body frame)
    baseline: float = 0.125                    # m -- "125mm apart in Y"
    baseline_axis: str = 'y'
    mount_pos: Tuple[float, float, float] = (baseline / 2, 0.0, 0.0)
    look_dir: Tuple[float, float, float] = (0.0, 0.0, -1.0)   # looking down
    link_idx_local: int = 0

    def __post_init__(self):
        if self.baseline_axis not in _AXES:
            raise ValueError(f"baseline_axis must be one of {list(_AXES)}, got {self.baseline_axis!r}")

    # ---------------- per-camera values ----------------

    def fov_of(self, side):
        """Vertical FOV (deg) so Genesis renders at that camera's mean focal length."""
        focal = self.left_focal if side == 'left' else self.right_focal
        return float(np.degrees(2.0 * np.arctan(self.res[1] / (2.0 * focal))))

    def camera_pos(self, side):
        """Optical centre of 'left' or 'right' camera in the body frame."""
        sign = -1.0 if side == 'left' else 1.0
        return tuple(np.array(self.mount_pos) * sign)

    def camera_options(self, side):
        """Pose + intrinsics kwargs for gs.sensors.RasterizerCameraOptions, per camera.

        `up` already carries the 90 deg roll, so the camera is created in its mounted orientation.
        """
        pos = self.camera_pos(side)
        lookat = tuple(np.array(pos) + np.array(self.look_dir))
        up = self.left_up if side == 'left' else self.right_up
        return dict(pos=pos, lookat=lookat, up=up, link_idx_local=self.link_idx_local,
                    res=self.res, fov=self.fov_of(side))

    # ---------------- left-dominant values ----------------

    @property
    def focal_px(self):
        return self.left_focal

    @property
    def fov(self):
        return self.fov_of('left')

    # ---------------- 3D geometry (left image) ----------------!!!!  rewn=moive froim here 

    def camera_axes(self):
        """Columns x (image right), y (image up), z (backwards) of the left image, in body frame."""
        z = -np.array(self.look_dir, float)
        z /= np.linalg.norm(z)
        x = np.cross(self.left_up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        return np.column_stack([x, y, z])

    def backproject(self, pts, depth):
        """(N, 2) left-image pixels + (N,) depth along the optical axis -> (N, 3) body-frame points."""
        W, H = self.res
        f = self.focal_px
        p_cam = np.column_stack([(pts[:, 0] - W / 2) * depth / f,
                                 -(pts[:, 1] - H / 2) * depth / f,
                                 -depth])
        return np.einsum('ij,nj->ni', self.camera_axes(), p_cam) + np.array(self.camera_pos('left'))

    def disparity_direction(self):
        """Unit (du, dv) in the left image: direction a static point shifts left -> right image."""
        x, y, _ = self.camera_axes().T
        shift = np.array(self.camera_pos('right')) - np.array(self.camera_pos('left'))
        d = np.array([-shift @ x, shift @ y])
        return d / np.linalg.norm(d)


# Stereo rig from camera_params.txt: 1920x1080, 125 mm baseline in Y, left (long) mean focal
# ~1719 px rolled 90 CCW (dominant), right (short) mean focal ~1176 px rolled 90 CW.
STEREO_CAM = StereoCameraConfig()
