"""Stereo point-depth calculations, independent of tracking and estimator state."""
from dataclasses import dataclass

import cv2
import numpy as np

from swarm.compVision.frame import compose_depth_display
from swarm.compVision.smallFeatures import LAYER_COLORS, unique_point_mask


@dataclass
class StereoDepth:
    """Same-frame matches aligned with the input point rows; invalid depth is NaN."""
    right_points: np.ndarray  # full-resolution pixels
    depths: np.ndarray       # optical-axis metres, NOT world altitude
    valid: np.ndarray
    scores: np.ndarray       # normalized patch correlation, not a probability
    sigma_depth: np.ndarray  # approximate depth uncertainty for map weighting


@dataclass
class FrameDepth:
    """Aggregated stereo-depth results for one frame."""
    by_layer: list
    points: np.ndarray
    colors: np.ndarray
    sigma: np.ndarray
    median: float
    valid_fraction: float
    view: np.ndarray


def estimate_vertical_velocity(previous, current, dt):
    """Estimate positive-up velocity from consecutive downward-camera depth frames.

    This approximation assumes a static, mostly level scene and a near-level
    downward-facing camera. It returns NaN until both frames contain at least
    three valid stereo points.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Frame interval must be finite and positive")
    if previous is None or current is None:
        return np.nan

    previous_depths = previous.points[:, 2]
    current_depths = current.points[:, 2]
    previous_depths = previous_depths[np.isfinite(previous_depths)]
    current_depths = current_depths[np.isfinite(current_depths)]
    if len(previous_depths) < 3 or len(current_depths) < 3:
        return np.nan

    return float((np.median(current_depths) - np.median(previous_depths)) / dt)


def estimate_frame_depth(frame, all_points, point_colors, layer_tracks, stereo, *,
                         use_stereo_depth, focal_px, baseline, direction,
                         cross_layer_min_distance):
    """Estimate and aggregate current-frame stereo depth for all feature layers.

    Per-layer results remain aligned with their input points. The returned
    ``points`` contains valid, de-duplicated full-resolution (u, v, Z) samples.
    The right image in ``frame`` is annotated with valid stereo matches.
    """
    depth_by_layer = []
    depth_points = np.empty((0, 3))
    depth_colors = np.empty((0, 3), np.uint8)
    depth_sigma = np.empty(0)
    depth_median, depth_valid_frac = np.nan, 0.0
    points, colors, sigma = [], [], []
    valid_count = 0

    for layer, uv in enumerate(all_points):
        if use_stereo_depth:
            result = stereo.estimate(
                frame.pyramids[0][layer], frame.pyramids[1][layer], uv,
                focal_px=focal_px, baseline=baseline,
                direction=direction, pixel_scale=2 ** layer,
            )
        else:
            n = len(uv)
            result = StereoDepth(
                np.full((n, 2), np.nan, np.float32),
                np.full(n, np.nan),
                np.zeros(n, bool),
                np.full(n, np.nan),
                np.full(n, np.nan),
            )
        depth_by_layer.append(result)
        valid = result.valid
        valid_count += int(valid.sum())
        points.append(np.column_stack((uv[valid], result.depths[valid])))
        colors.append(point_colors[layer][valid])
        sigma.append(result.sigma_depth[valid])

        right = frame.frames[1]
        for a, b in layer_tracks[layer].links:
            if valid[a] and valid[b]:
                cv2.line(
                    right,
                    tuple(result.right_points[a].astype(int)),
                    tuple(result.right_points[b].astype(int)),
                    LAYER_COLORS[layer % len(LAYER_COLORS)],
                    1,
                )
        for x, y in result.right_points[valid].astype(int):
            cv2.circle(right, (x, y), 2, LAYER_COLORS[layer % len(LAYER_COLORS)], -1)

    total = sum(map(len, all_points))
    depth_valid_frac = valid_count / total if total else 0.0
    if valid_count:
        points, colors, sigma = np.vstack(points), np.vstack(colors), np.concatenate(sigma)
        keep = unique_point_mask(points[:, :2], cross_layer_min_distance)
        depth_points, depth_colors, depth_sigma = points[keep], colors[keep], sigma[keep]
        depth_median = float(np.median(depth_points[:, 2]))

    depth_view = compose_depth_display(frame.gray_frames[0], depth_points, total)
    return FrameDepth(
        by_layer=depth_by_layer,
        points=depth_points,
        colors=depth_colors,
        sigma=depth_sigma,
        median=depth_median,
        valid_fraction=depth_valid_frac,
        view=depth_view,
    )




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

    @staticmethod
    def _empty_result(point_count):
        return StereoDepth(
            np.full((point_count, 2), np.nan, np.float32),
            np.full(point_count, np.nan),
            np.zeros(point_count, bool),
            np.full(point_count, np.nan),
            np.full(point_count, np.nan),
        )

    @staticmethod
    def _validate_inputs(left, right, points, focal_px, baseline, direction, pixel_scale):
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
        return points, direction, axis

    def _find_patch_matches(self, left, right, native_points, direction, axis,
                            dimensions, pixel_scale, focal_px, baseline, result):
        radius = self.patch_size // 2
        fB = focal_px * baseline
        lo = max(1, int(np.ceil(fB / self.max_depth / pixel_scale)))
        hi = int(np.floor(min(self.max_disparity, fB / self.min_depth) / pixel_scale))
        offsets = np.arange(lo, hi + 1) * np.sign(direction[axis])
        rows, seeds = [], []
        for i, point in enumerate(native_points):
            if (not np.isfinite(point).all() or np.any(point < radius)
                    or np.any(point >= dimensions - radius - 1)):
                continue
            template = cv2.getRectSubPix(left, (self.patch_size, self.patch_size), tuple(point))
            if template.std() < 5:
                continue
            candidates = point[axis] + offsets
            candidates = candidates[
                (candidates >= radius) & (candidates < dimensions[axis] - radius - 1)
            ]
            if len(candidates) < 3:
                continue
            start, end = candidates.min(), candidates.max()
            center = point.copy()
            center[axis] = (start + end) / 2
            size = [self.patch_size, self.patch_size]
            size[axis] += int(round(float(end - start)))
            strip = cv2.getRectSubPix(right, tuple(size), tuple(center))
            scores = cv2.matchTemplate(strip, template, cv2.TM_CCOEFF_NORMED).ravel()
            best = int(np.argmax(scores))
            score = float(scores[best])
            other = scores[np.abs(np.arange(len(scores)) - best) > 2]
            if (not np.isfinite(score) or score < self.min_score
                    or (len(other) and score - other.max() < self.uniqueness)):
                continue
            seed = point.copy()
            seed[axis] = start + best
            rows.append(i)
            seeds.append(seed)
            result.scores[i] = score
        return rows, seeds, radius, fB

    def _track_matches(self, left, right, native_points, rows, seeds,
                       dimensions, radius, pixel_scale):
        if not rows:
            return None
        rows = np.asarray(rows)
        old = native_points[rows, None, :].copy()
        seeds = np.asarray(seeds, np.float32).reshape(-1, 1, 2)
        lk = dict(
            winSize=(self.patch_size, self.patch_size),
            maxLevel=0,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .001),
        )
        matched, status, _ = cv2.calcOpticalFlowPyrLK(
            left, right, old, seeds.copy(), **lk
        )
        if matched is None or status is None:
            return None
        matched = matched.reshape(-1, 2)
        good = ((status.ravel() == 1) & np.isfinite(matched).all(axis=1)
                & (np.linalg.norm(matched - seeds[:, 0], axis=1) < 2)
                & (matched >= radius).all(axis=1)
                & (matched < dimensions - radius - 1).all(axis=1))
        rows, old, matched = rows[good], old[good], matched[good]
        if not len(rows):
            return None
        back, status, _ = cv2.calcOpticalFlowPyrLK(
            right, left, matched[:, None].copy(), old.copy(), **lk
        )
        if back is None or status is None:
            return None
        good = ((status.ravel() == 1) & np.isfinite(back.reshape(-1, 2)).all(axis=1)
                & (np.linalg.norm(back[:, 0] - old[:, 0], axis=1) * pixel_scale
                   < self.max_fb_error))
        return rows[good], matched[good] * pixel_scale

    def _store_depths(self, result, points, direction, matched_rows,
                      matched_points, fB, pixel_scale):
        rows, matched = matched_rows, matched_points
        delta = matched - points[rows]
        disparity = delta @ direction
        epipolar = abs(delta @ np.array([-direction[1], direction[0]]))
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = fB / disparity
            sigma = depth**2 / fB * (.25 * pixel_scale)
        good = ((disparity > .3) & (disparity <= self.max_disparity)
                & (epipolar < self.max_epipolar_error) & np.isfinite(depth)
                & (depth >= self.min_depth) & (depth <= self.max_depth)
                & (sigma / depth < .15))
        rows = rows[good]
        result.right_points[rows] = matched[good]
        result.depths[rows] = depth[good]
        result.sigma_depth[rows] = sigma[good]
        result.valid[rows] = True

    def estimate(self, left, right, points, *, focal_px, baseline, direction, pixel_scale=1.0):
        points, direction, axis = self._validate_inputs(
            left, right, points, focal_px, baseline, direction, pixel_scale
        )
        result = self._empty_result(len(points))
        if not len(points):
            return result
        native_points = points / pixel_scale
        height, width = left.shape
        dimensions = np.array([width, height])
        rows, seeds, radius, fB = self._find_patch_matches(
            left, right, native_points, direction, axis, dimensions,
            pixel_scale, focal_px, baseline, result,
        )
        tracked = self._track_matches(
            left, right, native_points, rows, seeds, dimensions, radius, pixel_scale
        )
        if tracked is None:
            return result
        rows, matched = tracked
        self._store_depths(result, points, direction, rows, matched, fB, pixel_scale)
        return result
