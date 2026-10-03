from dataclasses import dataclass
from typing import Tuple
import numpy as np
import cv2


_AXES = {'x': (1.0, 0.0, 0.0), 'y': (0.0, 1.0, 0.0), 'z': (0.0, 0.0, 1.0)}
_ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


@dataclass
class StereoCameraConfig:
    # ---------------- intrinsics ----------------
    res: Tuple[int, int] = (500, 600)           # (width, height) px, as Genesis expects
    fov: float = 85.0                           # VERTICAL field of view, deg
    near: float = 0.01
    far: float = 100.0

    # ---------------- rig geometry (drone body frame) ----------------
    baseline: float = 0.05                      # m, distance between left and right optical centres
    baseline_axis: str = 'y'                    # body axis the cameras are separated along ('x', 'y', 'z')
    mount_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # rig centre relative to the drone link
    look_dir: Tuple[float, float, float] = (0.0, 0.0, -1.0)   # viewing direction (down)
    up_dir: Tuple[float, float, float] = (0.0, 1.0, 0.0)      # body direction that shows as image "up"
    link_idx_local: int = 0

    # ---------------- image orientation (applied after capture) ----------------
    rotate_deg: int = 0                         # 0, 90, 180, 270 (clockwise)
    flip_horizontal: bool = False               # mirror left <-> right
    flip_vertical: bool = False                 # mirror top <-> bottom
    swap_left_right: bool = False               # exchange which camera is treated as "left"

    def __post_init__(self):
        if self.baseline_axis not in _AXES:
            raise ValueError(f"baseline_axis must be one of {list(_AXES)}, got {self.baseline_axis!r}")
        if self.rotate_deg not in _ROTATIONS:
            raise ValueError(f"rotate_deg must be one of {list(_ROTATIONS)}, got {self.rotate_deg}")

    # ---------------- derived values ----------------

    @property
    def focal_px(self):
        return (self.res[1] / 2) / np.tan(np.deg2rad(self.fov) / 2)

    def camera_pos(self, side):
        """Optical centre of 'left' or 'right' camera in the body frame."""
        sign = -1.0 if side == 'left' else 1.0
        return tuple(np.array(self.mount_pos) + sign * self.baseline / 2 * np.array(_AXES[self.baseline_axis]))

    def camera_options(self, side):
        """Pose kwargs for gs.sensors.RasterizerCameraOptions."""
        pos = self.camera_pos(side)
        lookat = tuple(np.array(pos) + np.array(self.look_dir))
        return dict(pos=pos, lookat=lookat, up=self.up_dir, link_idx_local=self.link_idx_local,
                    res=self.res, fov=self.fov, near=self.near, far=self.far)

    def orient(self, frame):
        """Apply rotation / flips to one captured frame."""
        if self.rotate_deg:
            frame = cv2.rotate(frame, _ROTATIONS[self.rotate_deg])
        if self.flip_horizontal and self.flip_vertical:
            return cv2.flip(frame, -1)
        if self.flip_horizontal:
            return cv2.flip(frame, 1)
        if self.flip_vertical:
            return cv2.flip(frame, 0)
        return frame

    def orient_pair(self, left, right):
        left, right = self.orient(left), self.orient(right)
        return (right, left) if self.swap_left_right else (left, right)

    def disparity_direction(self):
        """
        Unit (du, dv) in the ORIENTED image: direction a static point moves from the
        left image to the right image (v grows downwards). Disparity = dot(p_right - p_left, dir).
        """
        z = -np.array(self.look_dir, float)
        z /= np.linalg.norm(z)
        x = np.cross(self.up_dir, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        shift = np.array(self.camera_pos('right')) - np.array(self.camera_pos('left'))
        if self.swap_left_right:
            shift = -shift
        # camera moves +x_cam -> point moves left in image; camera moves +y_cam (up) -> point moves down
        d = np.array([-shift @ x, shift @ y])

        k = self.rotate_deg // 90
        for _ in range(k):                      # 90 deg clockwise: (u, v) -> (-v, u)
            d = np.array([-d[1], d[0]])
        if self.flip_horizontal:
            d[0] = -d[0]
        if self.flip_vertical:
            d[1] = -d[1]
        return d / np.linalg.norm(d)


# Edit this to change the drone's stereo rig.
STEREO_CAM = StereoCameraConfig()
