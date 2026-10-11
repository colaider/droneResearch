"""Large structural features (boxes, buildings) found on coarse pyramid layers.

The frame stage builds a Gaussian pyramid for every camera. Fine layers keep
the GFTT tracker; on coarse layers ``pyrDown`` has already blurred small grass
texture away, so this stage looks for *structure* there instead:

* a texture map (local std-dev / local mean) separating smooth faces from grass,
* long LSD line segments that border a smooth face (grass-only lines are dropped),
* corners where two structural lines meet at roughly a right angle,
* convex smooth polygons (box tops, building roofs and walls),
* padded regions grouping nearby lines (the original coarse-proposal contract).

All returned coordinates are ORIGINAL (full-resolution) image pixels. Layer
pixels map to full resolution by ``2**level``, which is exact for ``pyrDown``
including odd image sizes.
"""

from dataclasses import dataclass, field
from numbers import Integral
from typing import Optional, Sequence

import cv2
import numpy as np


# BGR overlay colour per pyramid level, distinct from the small-feature LAYER_COLORS.
STRUCTURE_COLORS = ((255, 255, 255), (255, 255, 255), (0, 255, 255), (203, 192, 255), (0, 128, 255))


def _empty_lines():
    return np.empty((0, 4), dtype=np.float32)


def _empty_points():
    return np.empty((0, 2), dtype=np.float32)


@dataclass
class LargeFeatures:
    """Structural features of one pyramid layer, in original-image pixels.

    ``regions`` contains half-open ``(x0, y0, x1, y1)`` rectangles suitable for
    ``gray[y0:y1, x0:x1]``. ``lines`` is ``(N, 4)`` float32 ``(x0, y0, x1, y1)``.
    ``corners`` is ``(M, 2)`` float32. ``quads`` is a list of ``(K, 2)`` float32
    convex polygons. ``smooth_mask`` is a uint8 mask at the LAYER resolution
    (255 = smooth surface), or None when nothing was analysed.
    """

    regions: list = field(default_factory=list)
    lines: np.ndarray = field(default_factory=_empty_lines)
    corners: np.ndarray = field(default_factory=_empty_points)
    quads: list = field(default_factory=list)
    smooth_mask: Optional[np.ndarray] = None
    level: int = 0


class LargeFeatureDetector:
    """Find structural lines, corners and smooth polygons on pyramid layers.

    Distance parameters are in ORIGINAL-image pixels and are scaled to the
    layer being processed. ``level`` is the layer used by ``detect(pyramid)``
    without ``levels``; ``detect(pyramid, levels=(1, 2))`` runs several layers.

    ``smooth_thresh`` is on the relative texture map ``std / (mean + 16)``.
    None picks Otsu's threshold, clamped to ``[min_smooth, max_smooth]`` and
    to ``rel_smooth * median(texture)``. The caps stop an image with no smooth
    surfaces (only grass) being split in half, and keep coarse layers, where
    pyrDown has already softened the grass, from calling grass smooth. The
    floor keeps genuinely flat areas smooth when Otsu lands near 0.
    """

    def __init__(self, level=2, min_length_px=None, join_px=12, pad_px=20,
                 max_regions=12, texture_win_px=28, smooth_thresh=None,
                 min_smooth=0.03, max_smooth=0.12, rel_smooth=0.5, side_px=12, filter_lines=True,
                 corner_join_px=32, corner_angle_deg=(60.0, 120.0),
                 quad_eps=0.03, min_quad_area_px=None, max_quad_frac=0.5,
                 min_quad_fill=0.8, median_ksize=5):
        if not isinstance(level, Integral) or level < 0:
            raise ValueError("level must be a nonnegative integer")
        if not isinstance(max_regions, Integral) or max_regions < 0:
            raise ValueError("max_regions must be a nonnegative integer")
        if (min_length_px is not None and
                (not np.isfinite(min_length_px) or min_length_px <= 0)):
            raise ValueError("min_length_px must be positive and finite")
        if (not np.isfinite(join_px) or join_px < 0 or
                not np.isfinite(pad_px) or pad_px < 0):
            raise ValueError("join_px and pad_px must be nonnegative and finite")
        self.level = int(level)
        self.min_length_px = min_length_px
        self.join_px = float(join_px)
        self.pad_px = float(pad_px)
        self.max_regions = int(max_regions)
        self.texture_win_px = float(texture_win_px)
        self.smooth_thresh = smooth_thresh
        self.min_smooth = float(min_smooth)
        self.max_smooth = float(max_smooth)
        self.rel_smooth = float(rel_smooth)
        self.side_px = float(side_px)
        self.filter_lines = bool(filter_lines)
        self.corner_join_px = float(corner_join_px)
        self.corner_angle_deg = tuple(corner_angle_deg)
        self.quad_eps = float(quad_eps)
        self.min_quad_area_px = min_quad_area_px
        self.max_quad_frac = float(max_quad_frac)
        self.min_quad_fill = float(min_quad_fill)
        self.median_ksize = int(median_ksize)
        # scale=1.0 avoids LSD's additional internal resize: the pyramid level
        # already chose the detection resolution.
        self._detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD, 1.0)

    # ---------------- public API ----------------

    def detect(self, pyramid: Sequence[np.ndarray], levels=None):
        """Detect on a grayscale pyramid (consecutive ``pyrDown`` levels).

        ``levels=None`` returns one ``LargeFeatures`` for ``self.level`` (or the
        coarsest available level). A sequence of levels returns
        ``{level: LargeFeatures}`` for the levels that exist in the pyramid.
        Invalid image types or non-pyrDown dimensions raise ``ValueError``.
        """
        single = levels is None
        if len(pyramid) == 0:
            return LargeFeatures() if single else {}
        gray = np.asarray(pyramid[0])
        self._check_gray(gray)
        wanted = [min(self.level, len(pyramid) - 1)] if single else [
            int(lv) for lv in levels if 0 <= int(lv) < len(pyramid)]
        out = {}
        for level in wanted:
            small = np.asarray(pyramid[level])
            self._check_gray(small)
            scale = 2 ** level
            h, w = gray.shape
            if small.shape != ((h + scale - 1) // scale, (w + scale - 1) // scale):
                raise ValueError("Pyramid dimensions must match consecutive pyrDown levels")
            out[level] = self.detect_layer(small, level, gray.shape)
        return out[wanted[0]] if single else out

    def detect_layer(self, gray, level, full_shape=None) -> LargeFeatures:
        """Detect on ONE pyramid layer. ``full_shape`` is the level-0 (h, w).

        Without ``full_shape`` the full size is assumed to be layer size *
        ``2**level``, which is only exact for even original dimensions.
        """
        gray = np.asarray(gray)
        self._check_gray(gray)
        if not isinstance(level, Integral) or level < 0:
            raise ValueError("level must be a nonnegative integer")
        level = int(level)
        scale = 2 ** level
        if full_shape is None:
            full_shape = (gray.shape[0] * scale, gray.shape[1] * scale)
        h, w = int(full_shape[0]), int(full_shape[1])
        empty = LargeFeatures(level=level)
        if gray.size == 0 or min(gray.shape) < 8 or int(gray.max()) == int(gray.min()):
            return empty

        texture = self.texture_map(gray, level)
        thresh = self._smooth_threshold(texture)
        smooth = (texture < thresh).astype(np.uint8) * 255
        smooth = self._clean_mask(smooth, level)
        # Pixels within half a texture window of a smooth face's border see the
        # neighbouring texture too, so grow the mask back by that radius.
        k = max(3, int(round(self.texture_win_px / 2 ** level)) | 1)
        smooth = cv2.dilate(smooth, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

        lines = self._lines(gray, level, (h, w))
        if self.filter_lines and len(lines):
            lines = lines[self._borders_smooth(lines / scale, texture, thresh, level)]
        corners = self._corners(lines)
        quads = self._quads(smooth, level, (h, w))
        regions = self._regions(lines, gray.shape, level, (h, w))
        return LargeFeatures(regions=regions, lines=lines, corners=corners,
                             quads=quads, smooth_mask=smooth, level=level)

    def texture_map(self, gray, level=0):
        """Relative local contrast ``std / (mean + 16)`` at the layer resolution."""
        k = max(3, int(round(self.texture_win_px / 2 ** level)) | 1)
        g = gray.astype(np.float32)
        mean = cv2.boxFilter(g, -1, (k, k), borderType=cv2.BORDER_REFLECT)
        sq = cv2.boxFilter(g * g, -1, (k, k), borderType=cv2.BORDER_REFLECT)
        std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
        return std / (mean + 16.0)

    # ---------------- stages ----------------

    @staticmethod
    def _check_gray(image):
        if image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError("Expected uint8 grayscale pyramid images")

    def _smooth_threshold(self, texture):
        if self.smooth_thresh is not None:
            return float(self.smooth_thresh)
        top = 4.0 * self.max_smooth
        scaled = np.clip(texture / top * 255.0, 0, 255).astype(np.uint8)
        otsu, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cap = min(self.max_smooth, self.rel_smooth * float(np.median(texture)))
        return float(max(self.min_smooth, min(float(otsu) / 255.0 * top, cap)))

    @staticmethod
    def _clean_mask(mask, level):
        r = max(1, int(round(8 / 2 ** level)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def _lines(self, gray, level, full_shape):
        h, w = full_shape
        scale = float(2 ** level)
        # Median blur removes leftover grass blades but keeps straight step edges.
        blurred = cv2.medianBlur(np.ascontiguousarray(gray), self.median_ksize) if self.median_ksize else gray
        detected = self._detector.detect(np.ascontiguousarray(blurred))[0]
        if detected is None:
            return _empty_lines()
        lines = np.asarray(detected, dtype=np.float32).reshape(-1, 4)
        lines = lines[np.isfinite(lines).all(axis=1)] * scale
        lines[:, (0, 2)] = np.clip(lines[:, (0, 2)], 0, w - 1)
        lines[:, (1, 3)] = np.clip(lines[:, (1, 3)], 0, h - 1)
        lengths = np.linalg.norm(lines[:, 2:] - lines[:, :2], axis=1)
        min_length = (max(24.0, 0.06 * min(h, w))
                      if self.min_length_px is None else self.min_length_px)
        return lines[lengths >= min_length].astype(np.float32, copy=False)

    def _borders_smooth(self, layer_lines, texture, thresh, level):
        """True for lines with a smooth surface on at least one side."""
        th, tw = texture.shape
        p0, p1 = layer_lines[:, :2], layer_lines[:, 2:]
        d = p1 - p0
        length = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
        normal = np.column_stack([-d[:, 1], d[:, 0]]) / length
        # Sample outside the texture window, or the edge itself looks textured.
        k = max(3, int(round(self.texture_win_px / 2 ** level)) | 1)
        offset = max(k / 2.0 + 1.0, self.side_px / 2 ** level)
        t = np.linspace(0.15, 0.85, 7)[None, :, None]
        along = p0[:, None, :] + t * d[:, None, :]                 # (N, 7, 2)
        keep = np.zeros(len(layer_lines), bool)
        for sign in (-1.0, 1.0):
            pts = along + sign * offset * normal[:, None, :]
            xs = np.clip(np.rint(pts[..., 0]).astype(int), 0, tw - 1)
            ys = np.clip(np.rint(pts[..., 1]).astype(int), 0, th - 1)
            keep |= np.median(texture[ys, xs], axis=1) < thresh
        return keep

    def _corners(self, lines):
        """Intersections of near-perpendicular lines with nearby endpoints."""
        if len(lines) < 2:
            return _empty_points()
        p0, p1 = lines[:, :2].astype(float), lines[:, 2:].astype(float)
        d = p1 - p0
        u = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        cos = np.abs(u @ u.T)
        lo, hi = self.corner_angle_deg
        max_cos = np.cos(np.deg2rad(min(lo, 180.0 - hi)))
        ends = np.stack([p0, p1], axis=1)                          # (N, 2, 2)
        gap = np.linalg.norm(ends[:, None, :, None, :] - ends[None, :, None, :, :], axis=-1)
        near = gap.min(axis=(2, 3)) < self.corner_join_px
        i, j = np.nonzero(np.triu(near & (cos < max_cos), k=1))
        if len(i) == 0:
            return _empty_points()
        # Infinite-line intersection: p0_i + s*d_i = p0_j + t*d_j.
        cross = d[i, 0] * d[j, 1] - d[i, 1] * d[j, 0]
        diff = p0[j] - p0[i]
        s = (diff[:, 0] * d[j, 1] - diff[:, 1] * d[j, 0]) / cross
        pts = p0[i] + s[:, None] * d[i]
        # The corner must sit near an endpoint of both lines.
        to_i = np.linalg.norm(ends[i] - pts[:, None, :], axis=2).min(axis=1)
        to_j = np.linalg.norm(ends[j] - pts[:, None, :], axis=2).min(axis=1)
        pts = pts[(to_i < self.corner_join_px) & (to_j < self.corner_join_px)]
        return _merge_points(pts, self.corner_join_px / 2).astype(np.float32)

    def _quads(self, smooth, level, full_shape):
        h, w = full_shape
        scale = 2 ** level
        layer_area = smooth.shape[0] * smooth.shape[1]
        min_area = (self.min_quad_area_px if self.min_quad_area_px is not None
                    else (0.04 * min(h, w)) ** 2) / scale ** 2
        contours, _ = cv2.findContours(smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        quads = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > self.max_quad_frac * layer_area:
                continue
            poly = cv2.approxPolyDP(contour, self.quad_eps * cv2.arcLength(contour, True), True)
            if not 4 <= len(poly) <= 6 or not cv2.isContourConvex(poly):
                continue
            (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
            if rw * rh <= 0 or area / (rw * rh) < self.min_quad_fill:
                continue
            q = poly.reshape(-1, 2).astype(np.float32) * scale
            q[:, 0] = np.clip(q[:, 0], 0, w - 1)
            q[:, 1] = np.clip(q[:, 1], 0, h - 1)
            quads.append(q)
        quads.sort(key=lambda q: -cv2.contourArea(q))
        return quads

    def _regions(self, lines, layer_shape, level, full_shape):
        """Group nearby lines into padded proposals (largest groups first)."""
        if len(lines) == 0 or self.max_regions == 0:
            return []
        h, w = full_shape
        scale = 2 ** level
        line_mask = np.zeros(layer_shape, dtype=np.uint8)
        for line in lines / float(scale):
            start = tuple(np.rint(line[:2]).astype(int))
            end = tuple(np.rint(line[2:]).astype(int))
            cv2.line(line_mask, start, end, 255, 1)
        # Quantization means joins are approximate within a coarse-grid pixel.
        radius = int(np.ceil(self.join_px / (2.0 * scale)))
        radius = min(radius, max(layer_shape))
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        joined = cv2.dilate(line_mask, kernel)
        count, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
        labels = sorted(range(1, count), key=lambda i: -int(stats[i, cv2.CC_STAT_AREA]))
        regions = []
        for label in labels:
            x, y, rw, rh, _ = (int(value) for value in stats[label])
            x0 = max(0, int(np.floor(x * scale - self.pad_px)))
            y0 = max(0, int(np.floor(y * scale - self.pad_px)))
            x1 = min(w, int(np.ceil((x + rw) * scale + self.pad_px)))
            y1 = min(h, int(np.ceil((y + rh) * scale + self.pad_px)))
            if x1 > x0 and y1 > y0:
                regions.append((x0, y0, x1, y1))
            if len(regions) >= self.max_regions:
                break
        return regions


# ---------------- helpers for later stages ----------------

def _merge_points(pts, radius):
    """Greedy merge of points closer than ``radius`` (keeps cluster means)."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    used = np.zeros(len(pts), bool)
    merged = []
    for k in range(len(pts)):
        if used[k]:
            continue
        close = (~used) & (np.linalg.norm(pts - pts[k], axis=1) < radius)
        used |= close
        merged.append(pts[close].mean(axis=0))
    return np.array(merged)


def structure_points(features: LargeFeatures, spacing_px=12.0, quad_interior=True):
    """Points worth giving depth: corners, samples along lines, quad interiors.

    Returns ``(P, 2)`` float32 original-image pixels. Interior samples use a
    grid of ``2 * spacing_px``.
    """
    parts = [np.asarray(features.corners, np.float32).reshape(-1, 2)]
    for x0, y0, x1, y1 in np.asarray(features.lines, np.float32).reshape(-1, 4):
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) // spacing_px) + 1)
        t = np.linspace(0.0, 1.0, n)[:, None]
        parts.append((np.array([x0, y0]) + t * np.array([x1 - x0, y1 - y0])).astype(np.float32))
    if quad_interior:
        step = 2.0 * spacing_px
        for quad in features.quads:
            x0, y0 = quad.min(axis=0)
            x1, y1 = quad.max(axis=0)
            gx, gy = np.meshgrid(np.arange(x0 + step / 2, x1, step),
                                 np.arange(y0 + step / 2, y1, step))
            grid = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32)
            contour = quad.reshape(-1, 1, 2)
            inside = [cv2.pointPolygonTest(contour, (float(x), float(y)), False) > 0
                      for x, y in grid]
            parts.append(grid[np.asarray(inside, bool)] if len(grid) else _empty_points())
    return np.vstack(parts).astype(np.float32) if parts else _empty_points()


def draw_large_features(image, features: LargeFeatures, color=(0, 255, 255), scale=1.0):
    """Draw lines, quads and corners on a BGR display copy (in place).

    ``scale`` maps original-image pixels to ``image`` pixels, e.g. 0.5 for a
    half-size preview or ``1 / 2**level`` for a layer view.
    """
    def px(p):
        return tuple(int(v) for v in np.rint(np.asarray(p) * scale))

    for quad in features.quads:
        cv2.polylines(image, [np.rint(quad * scale).astype(np.int32).reshape(-1, 1, 2)],
                      True, color, 1, cv2.LINE_AA)
    for x0, y0, x1, y1 in np.asarray(features.lines).reshape(-1, 4):
        cv2.line(image, px((x0, y0)), px((x1, y1)), color, 2, cv2.LINE_AA)
    for corner in np.asarray(features.corners).reshape(-1, 2):
        cv2.circle(image, px(corner), 5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(image, px(corner), 3, color, -1, cv2.LINE_AA)
    return image
