"""Same-frame stereo matching and metric optical-axis depth.

Images must already be rectified to a common pinhole projection with aligned
principal points. ``focal_px`` is the rectified focal length along the disparity
axis, in pixels; ``baseline`` and returned depths use the same metric unit.
This module does not perform calibration or infer camera mounting geometry.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DepthResult:
    """Row-aligned stereo results; fallback depths are explicitly not valid.

    ``right_points`` is NaN for invalid matches. ``depths`` contains a median
    valid-depth fallback, or the supplied height if there are no valid matches.
    Consumers should inspect ``valid`` to distinguish measured from filled depth.
    ``points`` contains image coordinates and depth, not Cartesian XYZ.
    """

    left_points: np.ndarray
    right_points: np.ndarray
    depths: np.ndarray
    valid: np.ndarray
    frame_idx: int

    @property
    def median(self):
        return float(np.median(self.depths[self.valid])) if self.valid.any() else np.nan

    @property
    def valid_fraction(self):
        return float(self.valid.mean()) if len(self.valid) else 0.0

    @property
    def points(self):
        return np.column_stack((self.left_points[self.valid], self.depths[self.valid]))


class DepthEstimator:
    """Independent stereo stage using actual left/right images of one frame.

    ``disparity_direction`` points along the expected *right minus left* image
    displacement. It is normalized before use, and can be horizontal or vertical.
    Temporal optical flow is deliberately not used to synthesize stereo matches.
    """

    def __init__(
        self,
        win_size=(100, 100),
        max_level=4,
        criteria=None,
        epipolar_threshold=1.0,
        min_disparity=0.3,
        fb_threshold=1.0,
    ):
        self.lk = dict(
            winSize=tuple(win_size),
            maxLevel=int(max_level),
            criteria=criteria if criteria is not None else (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01
            ),
        )
        self.epipolar_threshold = float(epipolar_threshold)
        self.min_disparity = float(min_disparity)
        self.fb_threshold = float(fb_threshold)
        if not all(np.isfinite(x) and x > 0 for x in (
            self.epipolar_threshold, self.min_disparity, self.fb_threshold
        )):
            raise ValueError("Stereo thresholds must be positive and finite")

    @staticmethod
    def _points(points):
        points = np.asarray(points, dtype=np.float32)
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        if points.ndim == 3 and points.shape[1:] == (1, 2):
            points = points[:, 0, :]
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points must have shape (N, 2) or (N, 1, 2)")
        return points.copy()

    @staticmethod
    def _calibration(focal_px, baseline, disparity_direction):
        focal_px, baseline = float(focal_px), float(baseline)
        if not np.isfinite(focal_px) or focal_px <= 0:
            raise ValueError("focal_px must be positive and finite")
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError("baseline must be positive and finite")
        direction = np.asarray(disparity_direction, dtype=float)
        if direction.shape != (2,) or not np.isfinite(direction).all():
            raise ValueError("disparity_direction must contain two finite values")
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("disparity_direction must be nonzero")
        return focal_px, baseline, direction / norm

    def triangulate(
        self, left_points, right_points, match_valid, focal_px, baseline,
        disparity_direction, fallback_height, frame_idx=0,
    ):
        """Validate given correspondences and compute ``focal_px * baseline / d``.

        The caller supplies status/forward-backward/bounds checks in
        ``match_valid``. This method checks finite coordinates, epipolar residual
        and signed disparity. ``estimate`` additionally performs image-bounds
        and bidirectional LK checks. Inputs are never modified.
        """
        focal_px, baseline, direction = self._calibration(
            focal_px, baseline, disparity_direction
        )
        left = self._points(left_points)
        right = self._points(right_points)
        valid = np.asarray(match_valid, dtype=bool).reshape(-1).copy()
        if len(left) != len(right) or len(left) != len(valid):
            raise ValueError("Points and match_valid must have the same row count")

        delta = right.astype(float) - left.astype(float)
        disparity = delta @ direction
        perpendicular = np.array([-direction[1], direction[0]])
        epipolar_error = np.abs(delta @ perpendicular)
        valid &= (
            np.isfinite(left).all(axis=1)
            & np.isfinite(right).all(axis=1)
            & np.isfinite(disparity)
            & (disparity > self.min_disparity)
            & (epipolar_error < self.epipolar_threshold)
        )

        depths = np.full(len(left), np.nan, dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            depths[valid] = focal_px * baseline / disparity[valid]
        valid &= np.isfinite(depths) & (depths > 0)
        if valid.any():
            fallback = float(np.median(depths[valid]))
        else:
            fallback = float(fallback_height) if fallback_height is not None else np.nan
            if not np.isfinite(fallback) or fallback <= 0:
                fallback = np.nan
        depths[~valid] = fallback
        right[~valid] = np.nan
        return DepthResult(left, right, depths, valid, int(frame_idx))

    @staticmethod
    def _in_bounds(points, shape):
        height, width = shape[:2]
        return (
            np.isfinite(points).all(axis=1)
            & (points[:, 0] >= 0) & (points[:, 0] < width)
            & (points[:, 1] >= 0) & (points[:, 1] < height)
        )

    def estimate(
        self, left_gray, right_gray, points, focal_px, baseline,
        disparity_direction, fallback_height, frame_idx=0,
    ):
        """Match same-time rectified uint8 grayscale images at left-image points.

        Failed LK calls and empty/invalid points yield an aligned result with
        invalid matches and fallback depths. Malformed images or calibration
        raise ``ValueError`` rather than silently using an incorrect camera model.
        """
        self._calibration(focal_px, baseline, disparity_direction)
        left_gray, right_gray = np.asarray(left_gray), np.asarray(right_gray)
        if (
            left_gray.ndim != 2 or right_gray.ndim != 2
            or left_gray.shape != right_gray.shape or left_gray.size == 0
            or left_gray.dtype != np.uint8 or right_gray.dtype != np.uint8
        ):
            raise ValueError("Stereo images must be nonempty same-size uint8 grayscale arrays")
        left = self._points(points)
        right = np.full_like(left, np.nan)
        valid = np.zeros(len(left), dtype=bool)

        def result():
            return self.triangulate(
                left, right, valid, focal_px, baseline, disparity_direction,
                fallback_height, frame_idx,
            )

        input_ids = np.flatnonzero(self._in_bounds(left, left_gray.shape))
        if not len(input_ids):
            return result()
        source = left[input_ids].reshape(-1, 1, 2)
        try:
            matched, status, _ = cv2.calcOpticalFlowPyrLK(
                left_gray, right_gray, source, None, **self.lk
            )
            if matched is None or status is None:
                return result()
            matched = matched.reshape(-1, 2)
            forward_good = (status.ravel() == 1) & self._in_bounds(matched, right_gray.shape)
            forward_ids = input_ids[forward_good]
            if not len(forward_ids):
                return result()
            forward_points = matched[forward_good]
            right[forward_ids] = forward_points

            back, status_back, _ = cv2.calcOpticalFlowPyrLK(
                right_gray, left_gray,
                forward_points.reshape(-1, 1, 2), None, **self.lk
            )
            if back is None or status_back is None:
                return result()
            back = back.reshape(-1, 2)
            fb_error = np.max(np.abs(back - left[forward_ids]), axis=1)
            valid[forward_ids] = (
                (status_back.ravel() == 1)
                & self._in_bounds(back, left_gray.shape)
                & (fb_error < self.fb_threshold)
            )
        except cv2.error:
            # LK may fail on unsupported pyramid sizes or unavailable image data.
            # No unsuccessful match is promoted to a stereo depth measurement.
            valid[:] = False
        return result()
