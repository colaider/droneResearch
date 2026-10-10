"""Stereo point-depth calculations, independent of tracking and estimator state."""
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
