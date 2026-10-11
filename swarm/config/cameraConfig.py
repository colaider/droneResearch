from dataclasses import dataclass
from typing import Tuple
import numpy as np


_AXES = {'x': (1.0, 0.0, 0.0), 'y': (0.0, 1.0, 0.0), 'z': (0.0, 0.0, 1.0)}


@dataclass
class StereoCameraConfig:
    """Parallel downward-looking simulation stereo rig with a shared projection.

    Camera axes are image-right, image-up, backwards (OpenGL convention).
    Pixel v increases downwards. The explicit common up vector matches the
    renderer and backprojection; USB calibration/rectification is separate.
    """
    res: Tuple[int, int] = (630, 400)        # (width, height) px -- sensor_size (1080, 1920) h:w
    near: float = 0.01
    far: float = 100.0

    # Simulation focal length in pixels (shared by both cameras)
    left_focal: float = 1500
    right_focal: float = 1500

    # Both cameras share the same image orientation.
    left_up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    right_up: Tuple[float, float, float] = (0.0, 1.0, 0.0)

    # rig geometry (drone body frame)
    baseline: float = 0.125                    # metres between optical centres
    baseline_axis: str = 'x'
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

        The explicit up vector is also used for metric backprojection.
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

    # ---------------- 3D geometry (left image) ----------------

    def camera_axes(self, side="left"):
        """Columns x (image right), y (image up), z (backwards) of the left image, in body frame."""
        z = -np.array(self.look_dir, float)
        z /= np.linalg.norm(z)
        
        up = np.array(self.left_up if side == "left" else self.right_up, float)
        if np.linalg.norm(up) < 1e-6:
            raise ValueError("Camera up vector must be explicit and nonzero")
        
        x = np.cross(up, z)
        if np.linalg.norm(x) < 1e-6:
            raise ValueError("Camera up vector must not be parallel to viewing direction")
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


# Simulation: common 1500 px focal, 125 mm baseline along body X.
STEREO_CAM = StereoCameraConfig()
