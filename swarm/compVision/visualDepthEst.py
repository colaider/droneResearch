"""Stereo point-depth calculations, independent of tracking and estimator state."""
from dataclasses import dataclass

import cv2
import numpy as np


def triangulate_altitude(news, olds, *, focal_px, baseline, disparity_direction,
                         frame_size=None, use_stereo_depth=True,
                         min_disparity_px=0.3, max_epipolar_error_px=1.0):
    """Write optical-axis depth Z into column 2 of both supplied stereo pairs.

    Each of olds/news contains writable floating-point Nx3 left/right arrays.
    Rows must be real, same-time correspondences in undistorted, aligned images
    with common focal length and principal point. disparity_direction follows
    right minus left; Z = focal_px * baseline / signed_disparity. Z uses the
    baseline's units and is neither world altitude nor slant range.

    frame_size, when supplied, is (height, width, ...). Invalid pairs get NaN.
    Both time steps are validated/calculated before either is modified. Returns
    the original news container; olds and news depth columns are updated in place.
    This function does not find stereo correspondences or use temporal flow.
    """
    focal, baseline = float(focal_px), float(baseline)
    direction = disparity_direction
    direction = np.asarray(direction, dtype=float)
    if (not np.isfinite(focal) or focal <= 0 or
            not np.isfinite(baseline) or baseline <= 0 or
            not np.isfinite(focal * baseline) or
            direction.shape != (2,) or not np.isfinite(direction).all()):
        raise ValueError('Stereo calibration requires positive finite focal length/baseline and a finite 2D direction')
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('Stereo disparity direction must be nonzero')
    if (not np.isfinite(min_disparity_px) or min_disparity_px <= 0 or
            not np.isfinite(max_epipolar_error_px) or max_epipolar_error_px <= 0):
        raise ValueError('Stereo disparity and epipolar thresholds must be positive and finite')
    direction = direction / norm
    perpendicular = np.array([-direction[1], direction[0]])

    # Calculate both time steps before modifying any caller-owned arrays.
    results = []
    for pairs in (olds, news):
        if len(pairs) != 2:
            raise ValueError('Each stereo sample must contain left and right arrays')
        for points in pairs:
            if (not isinstance(points, np.ndarray) or points.ndim != 2 or
                    points.shape[1] != 3 or
                    not np.issubdtype(points.dtype, np.floating) or
                    not points.flags.writeable):
                raise ValueError('Stereo points must be writable floating-point Nx3 arrays')
        left, right = pairs
        if len(left) != len(right):
            raise ValueError('Left and right stereo point counts must match')
        valid = np.isfinite(left[:, :2]).all(axis=1) & np.isfinite(right[:, :2]).all(axis=1)
        if frame_size is not None:
            height, width = frame_size[:2]
            for points in pairs:
                valid &= ((points[:, 0] >= 0) & (points[:, 0] < width) &
                          (points[:, 1] >= 0) & (points[:, 1] < height))
        delta = np.full((len(left), 2), np.nan)
        with np.errstate(over='ignore', invalid='ignore'):
            delta[valid] = right[valid, :2].astype(float) - left[valid, :2].astype(float)
            disparity = delta @ direction
            epipolar_error = np.abs(delta @ perpendicular)
        valid &= (np.isfinite(disparity) & (disparity > min_disparity_px) &
                  (epipolar_error < max_epipolar_error_px))
        if not use_stereo_depth:
            valid[:] = False
        depths = np.full(len(left), np.nan)
        with np.errstate(over='ignore', invalid='ignore'):
            depths[valid] = focal * baseline / disparity[valid]
        # Ensure both output arrays can store finite positive depths.
        limit = min(np.finfo(left.dtype).max, np.finfo(right.dtype).max)
        valid &= np.isfinite(depths) & (depths > 0) & (depths <= limit)
        depths[~valid] = np.nan
        results.append(depths)

    for pairs, depths in zip((olds, news), results):
        for points in pairs:
            points[:, 2] = depths
    return news


@dataclass
class StereoDepth:
    """Same-frame matches aligned with the input point rows; invalid depth is NaN."""
    right_points: np.ndarray  # full-resolution pixels
    depths: np.ndarray       # optical-axis metres, NOT world altitude
    valid: np.ndarray
    scores: np.ndarray       # normalized patch correlation, not a probability
    sigma_depth: np.ndarray  # approximate depth uncertainty for map weighting


class SparseStereoDepth:
    """Sparse stereo for equal-intrinsic, rectified horizontal/vertical image pairs.

    Search along the signed epipolar line, reject ambiguous/flat patches, then
    refine with local LK and check the reverse match. Search and LK run in each
    layer's native pixels; geometry and rejection thresholds use full pixels.
    No pose, altitude, temporal flow or simulator depth is used to infer depth.
    """
    def __init__(self, patch_size=15, min_score=.80, uniqueness=.06,
                 max_disparity=512.0, min_depth=.25, max_depth=30.0,
                 max_fb_error=1.0, max_epipolar_error=1.0):
        self.patch_size = patch_size
        self.min_score, self.uniqueness = min_score, uniqueness
        self.max_disparity = max_disparity
        self.min_depth, self.max_depth = min_depth, max_depth
        self.max_fb_error, self.max_epipolar_error = max_fb_error, max_epipolar_error

    def estimate(self, left, right, points, *, focal_px, baseline, direction, pixel_scale=1.0):
        points = np.asarray(points, np.float32).reshape(-1, 2)
        direction = np.asarray(direction, float)
        if (not np.isfinite(focal_px) or focal_px <= 0 or not np.isfinite(baseline) or baseline <= 0
                or direction.shape != (2,) or not np.isfinite(direction).all()
                or np.linalg.norm(direction) == 0 or not np.isfinite(pixel_scale) or pixel_scale <= 0):
            raise ValueError('Invalid stereo calibration or pyramid scale')
        if left.ndim != 2 or right.shape != left.shape or left.dtype != np.uint8 or right.dtype != np.uint8:
            raise ValueError('Stereo requires equal-size uint8 grayscale images')
        direction = direction / np.linalg.norm(direction)
        axis = int(np.argmax(np.abs(direction)))
        if abs(direction[1-axis]) > 1e-6:
            raise ValueError('Rectify stereo images to horizontal or vertical epipolar lines first')
        n = len(points)
        result = StereoDepth(np.full((n, 2), np.nan, np.float32), np.full(n, np.nan),
                             np.zeros(n, bool), np.full(n, np.nan), np.full(n, np.nan))
        if not n:
            return result
        native = points / pixel_scale
        h, w = left.shape
        dims = np.array([w, h])
        radius = self.patch_size // 2
        fB = focal_px * baseline
        lo = max(1, int(np.ceil(fB / self.max_depth / pixel_scale)))
        hi = int(np.floor(min(self.max_disparity, fB / self.min_depth) / pixel_scale))
        offsets = np.arange(lo, hi+1) * np.sign(direction[axis])
        rows, seeds = [], []
        for i, point in enumerate(native):
            if not np.isfinite(point).all() or np.any(point < radius) or np.any(point >= dims-radius-1):
                continue
            template = cv2.getRectSubPix(left, (self.patch_size, self.patch_size), tuple(point))
            if template.std() < 5:
                continue
            candidates = point[axis] + offsets
            candidates = candidates[(candidates >= radius) & (candidates < dims[axis]-radius-1)]
            if len(candidates) < 3:
                continue
            start, end = candidates.min(), candidates.max()
            center = point.copy(); center[axis] = (start+end)/2
            size = [self.patch_size, self.patch_size]
            size[axis] += int(round(float(end-start)))
            strip = cv2.getRectSubPix(right, tuple(size), tuple(center))
            scores = cv2.matchTemplate(strip, template, cv2.TM_CCOEFF_NORMED).ravel()
            best = int(np.argmax(scores)); score = float(scores[best])
            other = scores[np.abs(np.arange(len(scores))-best) > 2]
            if not np.isfinite(score) or score < self.min_score or (len(other) and score-other.max() < self.uniqueness):
                continue
            seed = point.copy(); seed[axis] = start + best
            rows.append(i); seeds.append(seed); result.scores[i] = score
        if not rows:
            return result
        rows = np.asarray(rows)
        old = native[rows, None, :].copy()
        seeds = np.asarray(seeds, np.float32).reshape(-1, 1, 2)
        lk = dict(winSize=(self.patch_size, self.patch_size), maxLevel=0,
                  flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                  criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .001))
        matched, status, _ = cv2.calcOpticalFlowPyrLK(left, right, old, seeds.copy(), **lk)
        if matched is None or status is None:
            return result
        matched = matched.reshape(-1, 2)
        good = ((status.ravel() == 1) & np.isfinite(matched).all(axis=1)
                & (np.linalg.norm(matched-seeds[:, 0], axis=1) < 2)
                & (matched >= radius).all(axis=1) & (matched < dims-radius-1).all(axis=1))
        rows, old, matched = rows[good], old[good], matched[good]
        if not len(rows):
            return result
        back, status, _ = cv2.calcOpticalFlowPyrLK(right, left, matched[:, None].copy(), old.copy(), **lk)
        if back is None or status is None:
            return result
        good = ((status.ravel() == 1) & np.isfinite(back.reshape(-1, 2)).all(axis=1)
                & (np.linalg.norm(back[:, 0]-old[:, 0], axis=1)*pixel_scale < self.max_fb_error))
        rows, matched = rows[good], matched[good]*pixel_scale
        delta = matched-points[rows]
        disparity = delta @ direction
        epipolar = abs(delta @ np.array([-direction[1], direction[0]]))
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = fB/disparity
            sigma = depth**2/fB * (.25*pixel_scale)
        good = ((disparity > .3) & (disparity <= self.max_disparity)
                & (epipolar < self.max_epipolar_error) & np.isfinite(depth)
                & (depth >= self.min_depth) & (depth <= self.max_depth)
                & (sigma/depth < .15))
        rows = rows[good]
        result.right_points[rows] = matched[good]
        result.depths[rows] = depth[good]
        result.sigma_depth[rows] = sigma[good]
        result.valid[rows] = True
        return result
