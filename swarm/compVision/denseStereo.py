"""Dense stereo depth on a pyramid layer, plus depth for structural features.

The sim renders an already aligned stereo pair. Real cameras first go through
``StereoRectifier`` (from a ``scripts/calibrate_stereo.py`` calibration file).
``DenseStereo`` then runs SGBM on one existing pyramid layer, so no extra
downsampling is needed. Depth is along the optical axis in baseline units.

Smooth box tops and roofs often have too little texture for SGBM (which then
invents depth, so textureless pixels are invalidated). ``fill_quad_plane``
fits a plane (inverse depth is affine in u, v for a 3D plane) to the depth
inside a smooth polygon, or, when it has none, matches the polygon's
silhouette along the baseline, and fills it.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from swarm.compVision.largeFeatures import LargeFeatures, structure_points


@dataclass
class DepthMap:
    """Depth at one pyramid layer: ``depth`` float32 (NaN = invalid).

    ``left``/``right`` are the layer images it was computed from, and
    ``direction`` / ``focal_px`` (layer pixels) / ``baseline`` the stereo
    geometry, so later stages can match textureless surfaces themselves.
    """

    depth: np.ndarray
    valid: np.ndarray
    level: int = 0
    left: Optional[np.ndarray] = None
    right: Optional[np.ndarray] = None
    direction: Optional[np.ndarray] = None
    focal_px: Optional[float] = None
    baseline: Optional[float] = None
    max_disparity: Optional[int] = None

    @property
    def scale(self):
        return 2 ** self.level


# ---------------- rectification ----------------

class StereoRectifier:
    """Undistort + rectify a stereo pair; identity when no calibration is given.

    After rectification both images share one pinhole model: ``focal_px``,
    ``baseline`` (calibration units, normally metres) and the image-space
    ``disparity_direction`` (right minus left) for ``DenseStereo``.
    """

    def __init__(self, maps=None, focal_px=None, baseline=None,
                 disparity_direction=None, size=None):
        self.maps = maps
        self.focal_px = focal_px
        self.baseline = baseline
        self.disparity_direction = (None if disparity_direction is None
                                    else np.asarray(disparity_direction, float))
        self.size = size

    @classmethod
    def identity(cls):
        return cls()

    @property
    def is_identity(self):
        return self.maps is None

    @classmethod
    def from_calibration(cls, K1, D1, K2, D2, R, T, size, alpha=0.0):
        """``size`` is (width, height) of the images the calibration was made on."""
        size = (int(size[0]), int(size[1]))
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
            np.asarray(K1, float), np.asarray(D1, float),
            np.asarray(K2, float), np.asarray(D2, float),
            size, np.asarray(R, float), np.asarray(T, float).reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
        maps = (cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_16SC2),
                cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_16SC2))
        return cls._from_projections(maps, P1, P2, size)

    @classmethod
    def from_npz(cls, path, alpha=0.0):
        """Calibration saved by ``scripts/calibrate_stereo.py``."""
        data = np.load(path)
        return cls.from_calibration(data['K1'], data['D1'], data['K2'], data['D2'],
                                    data['R'], data['T'], tuple(data['size']), alpha)

    @classmethod
    def from_ros_yaml(cls, left_path, right_path):
        """``left.yaml`` / ``right.yaml`` from ROS ``camera_calibration`` (RB3 rig).

        Those already hold the rectification (R) and projection (P) matrices.
        """
        import yaml

        def load(path):
            with open(path) as f:
                c = yaml.safe_load(f)
            mat = lambda key: np.array(c[key]['data'], float).reshape(c[key]['rows'], c[key]['cols'])
            return (mat('camera_matrix'), mat('distortion_coefficients').ravel(),
                    mat('rectification_matrix'), mat('projection_matrix'),
                    (int(c['image_width']), int(c['image_height'])))

        (K1, D1, R1, P1, size), (K2, D2, R2, P2, _) = load(left_path), load(right_path)
        maps = (cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_16SC2),
                cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_16SC2))
        return cls._from_projections(maps, P1, P2, size)

    @classmethod
    def _from_projections(cls, maps, P1, P2, size):
        focal = float(P1[0, 0])
        # P2 = [f 0 cx f*Tx; 0 f cy f*Ty]: one of Tx, Ty is nonzero after rectification.
        shift = np.array([P2[0, 3], P2[1, 3]]) / focal
        baseline = float(np.linalg.norm(shift))
        # A point at depth Z appears at x_r = x_l + f*Tx/Z, so right-left follows shift.
        return cls(maps, focal, baseline, shift / baseline, size)

    def rectify(self, left, right):
        if self.is_identity:
            return left, right
        h, w = left.shape[:2]
        if (w, h) != self.size:
            raise ValueError(f'Images are {w}x{h}, calibration is {self.size[0]}x{self.size[1]}')
        (lx, ly), (rx, ry) = self.maps
        return (cv2.remap(left, lx, ly, cv2.INTER_LINEAR),
                cv2.remap(right, rx, ry, cv2.INTER_LINEAR))


def estimate_disparity_direction(left_gray, right_gray, max_points=300):
    """Measure the dominant right-minus-left image shift with sparse LK.

    A check for ``cameraConfig.disparity_direction()``: returns the unit
    direction and median shift length in pixels, or ``(None, 0.0)``.
    """
    pts = cv2.goodFeaturesToTrack(left_gray, max_points, 0.01, 10)
    if pts is None:
        return None, 0.0
    moved, st, _ = cv2.calcOpticalFlowPyrLK(left_gray, right_gray, pts, None,
                                            winSize=(31, 31), maxLevel=4)
    ok = st.ravel() == 1
    if ok.sum() < 5:
        return None, 0.0
    shift = np.median((moved - pts).reshape(-1, 2)[ok], axis=0)
    length = float(np.linalg.norm(shift))
    return (shift / length if length > 0 else None), length


# ---------------- dense stereo ----------------

def _to_canonical(img, axis_vertical, flip):
    if axis_vertical:
        img = np.ascontiguousarray(img.T)
    if flip:
        img = np.ascontiguousarray(img[:, ::-1])
    return img


def _from_canonical(img, axis_vertical, flip):
    if flip:
        img = img[:, ::-1]
    if axis_vertical:
        img = img.T
    return np.ascontiguousarray(img)


class DenseStereo:
    """SGBM depth on one pyramid layer of a rectified/aligned stereo pair.

    ``focal_px`` is the FULL-resolution focal length; ``disparity_direction``
    is the unit right-minus-left image shift (``cameraConfig`` convention) and
    must be along an image axis. Images are rotated/flipped so SGBM always
    sees its standard layout (match to the left in the right image), and
    results are mapped back. ``min_depth`` sets the disparity search range.

    SGBM happily invents a smooth disparity across textureless areas, so
    pixels whose local std-dev is below ``min_texture`` gray levels are
    marked invalid instead (see ``fill_quad_plane`` for those surfaces).
    """

    def __init__(self, focal_px, baseline, disparity_direction, min_depth=2.0,
                 block_size=5, lr_check_px=1.0, uniqueness=10,
                 speckle_window=100, speckle_range=2, min_texture=2.0):
        d = np.asarray(disparity_direction, float).reshape(2)
        norm = np.linalg.norm(d)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError('disparity_direction must be a nonzero 2D vector')
        d = d / norm
        if max(abs(d[0]), abs(d[1])) < np.cos(np.deg2rad(10)):
            raise ValueError('disparity_direction must lie along an image axis; rectify the images first')
        self.axis_vertical = abs(d[1]) > abs(d[0])
        # Canonical SGBM layout has right-minus-left pointing to -x.
        self.flip = (d[1] if self.axis_vertical else d[0]) > 0
        self.focal_px = float(focal_px)
        self.baseline = float(baseline)
        self.min_depth = float(min_depth)
        self.block_size = int(block_size)
        self.lr_check_px = lr_check_px
        self.uniqueness = int(uniqueness)
        self.speckle_window = int(speckle_window)
        self.speckle_range = int(speckle_range)
        self.min_texture = min_texture
        self.direction = d
        self._matchers = {}

    def num_disparities(self, level):
        max_disp = self.focal_px * self.baseline / self.min_depth / 2 ** level
        return int(max(16, 16 * np.ceil(max_disp / 16)))

    def _matcher(self, level):
        if level not in self._matchers:
            b = self.block_size
            self._matchers[level] = cv2.StereoSGBM_create(
                minDisparity=0, numDisparities=self.num_disparities(level), blockSize=b,
                P1=8 * b * b, P2=32 * b * b, disp12MaxDiff=-1,
                uniquenessRatio=self.uniqueness, speckleWindowSize=self.speckle_window,
                speckleRange=self.speckle_range, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        return self._matchers[level]

    def compute(self, left_gray, right_gray, level=0) -> DepthMap:
        """Depth for the LEFT image of one pyramid layer."""
        if left_gray.shape != right_gray.shape or left_gray.ndim != 2:
            raise ValueError('Expected two equal-size grayscale images')
        L = _to_canonical(left_gray, self.axis_vertical, self.flip)
        R = _to_canonical(right_gray, self.axis_vertical, self.flip)
        matcher = self._matcher(level)
        disp = matcher.compute(L, R).astype(np.float32) / 16.0
        valid = disp > 0.5
        if self.min_texture:
            g = L.astype(np.float32)
            k = (self.block_size, self.block_size)
            mean = cv2.boxFilter(g, -1, k)
            std = np.sqrt(np.maximum(cv2.boxFilter(g * g, -1, k) - mean * mean, 0))
            valid &= std >= self.min_texture
        if self.lr_check_px is not None:
            # Right-image disparity from the mirrored problem, for a left/right check.
            disp_r = matcher.compute(np.ascontiguousarray(R[:, ::-1]),
                                     np.ascontiguousarray(L[:, ::-1]))[:, ::-1].astype(np.float32) / 16.0
            ys, xs = np.nonzero(valid)
            xr = np.rint(xs - disp[ys, xs]).astype(int)
            inside = xr >= 0
            agree = np.zeros(len(xs), bool)
            agree[inside] = np.abs(disp_r[ys[inside], xr[inside]] - disp[ys[inside], xs[inside]]) <= self.lr_check_px
            valid[ys[~agree], xs[~agree]] = False
        focal = self.focal_px / 2 ** level
        depth = np.full(disp.shape, np.nan, np.float32)
        depth[valid] = focal * self.baseline / disp[valid]
        depth = _from_canonical(depth, self.axis_vertical, self.flip)
        return DepthMap(depth=depth, valid=np.isfinite(depth), level=level,
                        left=left_gray, right=right_gray, direction=self.direction,
                        focal_px=focal, baseline=self.baseline,
                        max_disparity=self.num_disparities(level))


# ---------------- depth for features ----------------

def sample_depth(depth_map: DepthMap, pts, percentile=10.0, win_px=5):
    """Robust depth at original-image points: a low percentile of a window.

    Seen from above the nearer side of an edge is the top of the structure,
    so a low percentile snaps edge/corner points onto the roof, not the ground.
    """
    pts = np.asarray(pts, float).reshape(-1, 2)
    if len(pts) == 0:
        return np.empty(0)
    h, w = depth_map.depth.shape
    p = pts / depth_map.scale
    r = max(1, int(round(win_px / 2)))
    off = np.arange(-r, r + 1)
    xs = np.clip(np.rint(p[:, 0])[:, None, None].astype(int) + off[None, None, :], 0, w - 1)
    ys = np.clip(np.rint(p[:, 1])[:, None, None].astype(int) + off[None, :, None], 0, h - 1)
    window = depth_map.depth[ys, xs].reshape(len(p), -1)
    out = np.full(len(p), np.nan)
    has = np.isfinite(window).any(axis=1)
    out[has] = np.nanpercentile(window[has], percentile, axis=1)
    return out


def _affine(A, coef):
    # Explicit sum: numpy 2.x matmul on macOS Accelerate raises spurious FP warnings here.
    return A[:, 0] * coef[0] + A[:, 1] * coef[1] + A[:, 2] * coef[2]


def _fit_inverse_depth_plane(uv, z, iters=100, tol=0.02, rng=None):
    """RANSAC fit 1/Z = a*u + b*v + c. Returns (a, b, c) or None."""
    if len(z) < 6:
        return None
    rng = np.random.default_rng(0) if rng is None else rng
    uv = np.asarray(uv, float)
    # Normalise pixels so near-collinear samples are detectable and well conditioned.
    centre, spread = uv.mean(axis=0), max(float(uv.std()), 1.0)
    A = np.column_stack([(uv - centre) / spread, np.ones(len(uv))])
    inv = 1.0 / z
    best = None
    for _ in range(iters):
        pick = rng.choice(len(z), 3, replace=False)
        if abs(np.linalg.det(A[pick])) < 1e-3:
            continue
        coef = np.linalg.solve(A[pick], inv[pick])
        inliers = np.abs(_affine(A, coef) - inv) <= tol * np.abs(inv)
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.sum() < max(6, 0.3 * len(z)):
        return None
    a, b, c = np.linalg.lstsq(A[best], inv[best], rcond=None)[0]
    # Back to raw pixels: 1/Z = a*u + b*v + c.
    return np.array([a / spread, b / spread, c - (a * centre[0] + b * centre[1]) / spread])


def silhouette_disparity(depth_map: DepthMap, mask, max_disparity=None, min_contrast=0.7,
                         valley_tol=0.1, valley_abs=0.1):
    """Disparity (layer px) of a textureless face from its outline alone.

    Slides the masked left-image face along the baseline over the right image.
    Only near the true disparity does the whole face land on itself rather
    than partly on the surrounding texture. ``mask`` should lie strictly
    inside the face (erode it): a flat face then gives a zero-cost valley
    ``inset`` px either side of the truth, so the valley centre is returned.
    Resolution is about +-0.5 layer px (the outline's pixel discretisation),
    so prefer a finer depth level when the disparity is small.
    Returns None without a clear minimum.
    """
    if depth_map.left is None or depth_map.right is None or depth_map.direction is None:
        return None
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    h, w = depth_map.left.shape
    dx, dy = np.rint(depth_map.direction).astype(int)
    if max_disparity is None:
        max_disparity = depth_map.max_disparity or (w if dx else h)
    values = depth_map.left[ys, xs].astype(np.float32)
    costs = np.full(int(max_disparity) + 1, np.inf)
    for k in range(len(costs)):
        xr, yr = xs + k * dx, ys + k * dy
        inside = (xr >= 0) & (xr < w) & (yr >= 0) & (yr < h)
        if inside.mean() < 0.8:
            break
        costs[k] = np.abs(depth_map.right[yr[inside], xr[inside]].astype(np.float32) - values[inside]).mean()
    finite = np.isfinite(costs)
    if finite.sum() < 3:
        return None
    best = int(np.argmin(costs))
    lo, med = costs[best], float(np.median(costs[finite]))
    if lo > min_contrast * med:
        return None
    # Contiguous run of near-minimal cost around the best shift.
    flat = costs <= lo * (1 + valley_tol) + valley_abs
    a = b = best
    while a > 0 and flat[a - 1]:
        a -= 1
    while b < len(costs) - 1 and flat[b + 1]:
        b += 1
    disparity = 0.5 * (a + b)
    return disparity if disparity > 0 else None


def fill_quad_plane(depth_map: DepthMap, quad, spacing_px=16.0, min_valid_frac=0.3,
                    max_fit_points=2000):
    """Fill a smooth polygon with depth from a plane fit.

    Uses valid depth inside the (eroded) polygon when at least
    ``min_valid_frac`` of it has depth (a textured, possibly tilted roof).
    Otherwise the face is textureless: its disparity comes from
    ``silhouette_disparity`` (a fronto-parallel top), or, without the images,
    from depth sampled along its outline. Returns ``(G, 3)`` (u, v, Z) on a
    ``spacing_px`` grid.
    """
    quad = np.asarray(quad, np.float32).reshape(-1, 2)
    scale = depth_map.scale
    h, w = depth_map.depth.shape
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.rint(quad / scale).astype(np.int32).reshape(-1, 1, 2)], 255)
    inside = mask > 0
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero((mask > 0) & depth_map.valid)
    coef = None
    if inside.sum() and len(xs) >= min_valid_frac * inside.sum():
        uv, z = np.column_stack([xs, ys]).astype(float) * scale, depth_map.depth[ys, xs].astype(float)
    elif depth_map.left is not None:
        # Strictly inside the face: the quad outline may include a grass rim.
        disp = silhouette_disparity(depth_map, cv2.erode(inside.astype(np.uint8), np.ones((5, 5), np.uint8)))
        if disp is None:
            return np.empty((0, 3))
        coef = np.array([0.0, 0.0, disp / (depth_map.focal_px * depth_map.baseline)])
        uv = z = np.empty(0)
    else:
        outline = structure_points(LargeFeatures(lines=_outline_lines(quad)), spacing_px / 2,
                                   quad_interior=False)
        z = sample_depth(depth_map, outline)
        ok = np.isfinite(z)
        uv, z = outline[ok].astype(float), z[ok]
    if coef is None and len(z) > max_fit_points:
        keep = np.random.default_rng(0).choice(len(z), max_fit_points, replace=False)
        uv, z = uv[keep], z[keep]
    if coef is None:
        coef = _fit_inverse_depth_plane(uv, z)
    if coef is None:
        return np.empty((0, 3))
    x0, y0 = quad.min(axis=0)
    x1, y1 = quad.max(axis=0)
    gx, gy = np.meshgrid(np.arange(x0 + spacing_px / 2, x1, spacing_px),
                         np.arange(y0 + spacing_px / 2, y1, spacing_px))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    if len(grid) == 0:
        return np.empty((0, 3))
    gi = np.clip(np.rint(grid / scale).astype(int), 0, [w - 1, h - 1])
    grid = grid[inside[gi[:, 1], gi[:, 0]]]
    inv = grid[:, 0] * coef[0] + grid[:, 1] * coef[1] + coef[2]
    ok = inv > 0
    return np.column_stack([grid[ok], 1.0 / inv[ok]])


def _outline_lines(quad):
    return np.column_stack([quad, np.roll(quad, -1, axis=0)]).astype(np.float32)


def map_points(features_by_level, depth_map: Optional[DepthMap], dense_spacing_px=16.0,
               line_spacing_px=12.0, structure_weight=4.0):
    """Points for ``TerrainMap.add``: ``(N, 4)`` of (u, v, Z, weight).

    Dense SGBM samples get weight 1; structure points (corners, line samples)
    and plane-filled smooth polygons get ``structure_weight`` so the map keeps
    sharp roof edges instead of averaging them with the ground behind.
    """
    if depth_map is None:
        return np.empty((0, 4))
    parts = []
    step = max(1, int(round(dense_spacing_px / depth_map.scale)))
    sub = depth_map.depth[::step, ::step]
    ys, xs = np.nonzero(np.isfinite(sub))
    if len(xs):
        uv = np.column_stack([xs, ys]).astype(float) * step * depth_map.scale
        parts.append(np.column_stack([uv, sub[ys, xs], np.ones(len(xs))]))
    if isinstance(features_by_level, LargeFeatures):
        features_by_level = {features_by_level.level: features_by_level}
    for features in (features_by_level or {}).values():
        pts = structure_points(features, line_spacing_px, quad_interior=False)
        z = sample_depth(depth_map, pts)
        ok = np.isfinite(z)
        if ok.any():
            parts.append(np.column_stack([pts[ok], z[ok], np.full(ok.sum(), structure_weight)]))
        for quad in features.quads:
            filled = fill_quad_plane(depth_map, quad, dense_spacing_px)
            if len(filled):
                parts.append(np.column_stack([filled, np.full(len(filled), structure_weight)]))
    return np.vstack(parts) if parts else np.empty((0, 4))
