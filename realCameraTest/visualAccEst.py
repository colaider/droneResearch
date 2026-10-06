"""USB-camera adapter for the current swarm estimator; no Genesis runtime needed.

Tracking, filters, drawing and velocity fitting are inherited from
swarm.compVision.visualAccEst. Only calibration and missing-sensor behavior
are adapted here. Keep filter experiments in the shared swarm implementation.
"""
from collections import deque
from pathlib import Path
import sys

import cv2
import numpy as np

# Support launching testRealCameras.py directly from any working directory.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from swarm.compVision.visualAccEst import VisulaAcEst as SwarmVisulaAcEst


class VisulaAcEst(SwarmVisulaAcEst):
    def __init__(self, focal_px, baseline, res=None, dt=1.0 / 30.0,
                 disparity_dir=(1.0, 0.0), default_depth=1.0,
                 epipolar_tol=2.0, min_disparity=0.3):
        focal_px, baseline = float(focal_px), float(baseline)
        if not np.isfinite(focal_px) or focal_px <= 0:
            raise ValueError("focal_px must be positive and finite")
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError("baseline must be positive and finite")
        res = (1920, 1080) if res is None else res
        if len(res) != 2 or min(res) <= 0:
            raise ValueError("res must contain positive width and height")
        fov = np.degrees(2 * np.arctan(res[1] / (2 * focal_px)))
        super().__init__(res, fov)
        self.foc_l = focal_px
        self.camera_saperation = baseline
        self.set_dt(dt)
        self.disp_dir = np.asarray(disparity_dir, dtype=float)
        if (self.disp_dir.shape != (2,) or not np.isfinite(self.disp_dir).all()
                or np.linalg.norm(self.disp_dir) == 0):
            raise ValueError("disparity_dir must be a finite, nonzero 2D vector")
        self.disp_dir = self.disp_dir / np.linalg.norm(self.disp_dir)
        for name, value in [('default_depth', default_depth), ('epipolar_tol', epipolar_tol),
                            ('min_disparity', min_disparity)]:
            value = float(value)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
            setattr(self, name, value)
        self.drone_ang = np.zeros(2)  # Used by the shared low-feature fallback.
        self.buffer = deque(maxlen=2)

    def update_frame(self, frame, idx):
        left, right = frame
        if right.shape[:2] != left.shape[:2]:
            right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
        if hasattr(self, 'frame_size') and self.frame_size[:2] != left.shape[:2]:
            self.buffer.clear()
            self.tracked_points.clear()
            self.vkfs.clear()
        self.frame_size = left.shape
        super().update_frame([left, right], idx)

    def _take_sensor_window(self):
        # No flight controller: do not invent attitude or rotation compensation.
        self.avg_ang_vel[:] = 0
        self.avg_att[:] = 0
        self.avg_pos[:] = (0, 0, self.default_depth)

    def _current_height(self):
        return self.default_depth

    def point_prediction_filtering(self, old, new):
        # A hand-held rig has no commanded velocity to compare tracks against.
        return np.ones(len(new), dtype=bool)

    def trinagulate_altitude(self, news, flow, olds, triangles, links):
        delta = olds[1][:, :2] - olds[0][:, :2]
        disparity = np.abs(delta @ self.disp_dir)
        perpendicular = np.array([-self.disp_dir[1], self.disp_dir[0]])
        offset = np.abs(delta @ perpendicular)
        keep = (offset < self.epipolar_tol) & (disparity > self.min_disparity)
        depth = np.full(len(disparity), self.default_depth)
        if keep.any():
            depth[keep] = self.foc_l * self.camera_saperation / disparity[keep]
            depth[~keep] = np.median(depth[keep])
        self.depth_valid_frac = float(keep.mean()) if len(keep) else 0.0
        self.depth_median = float(np.median(depth[keep])) if keep.any() else np.nan
        for camera in range(2):
            olds[camera][:, 2] = depth
            news[camera][:, 2] = depth
        self.depth_points = news[0][keep].copy()
        self.depth_frame_idx = self.current_frame.idx
        return news

    def _fallback_velocity(self):
        self.depth_median = np.nan
        self.depth_valid_frac = 0.0
        self.depth_points = np.empty((0, 3))
        self.depth_frame_idx = self.current_frame.idx
        return np.zeros(3)

    def set_dt(self, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        super().set_dt(float(dt))
