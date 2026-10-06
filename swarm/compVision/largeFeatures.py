"""Coarse structural candidates for the visual-navigation pipeline.

The frame stage owns grayscale conversion, filtering, and its Gaussian pyramid.
This stage uses one supplied pyramid level to detect long line segments, then
groups nearby segments into regions for the fine-feature stage. A region is a
proposal for extra analysis, not a detected building or a depth measurement.
"""

from dataclasses import dataclass, field
from numbers import Integral
from typing import Sequence

import cv2
import numpy as np


@dataclass
class LargeFeatures:
    """Structural proposals expressed in the original image coordinates.

    ``regions`` contains half-open ``(x0, y0, x1, y1)`` pixel rectangles, suitable
    for ``gray[y0:y1, x0:x1]``. ``lines`` has shape ``(N, 4)`` and float32
    endpoints ``(x0, y0, x1, y1)``. Lines are diagnostic geometric features;
    their endpoints are not automatically stable point tracks.
    """

    regions: list[tuple[int, int, int, int]] = field(default_factory=list)
    lines: np.ndarray = field(
        default_factory=lambda: np.empty((0, 4), dtype=np.float32)
    )


class LargeFeatureDetector:
    """Find long lines in a frame's pyramid and propose padded image regions.

    All distance parameters are in ORIGINAL-image pixels. ``level=2`` uses the
    quarter-resolution image, which suppresses fine texture before detection.
    The detector does not downsample again. If fewer levels are supplied, the
    coarsest available level is used. Levels must be consecutive ``pyrDown``
    outputs starting with the full-resolution grayscale image.

    ``join_px`` controls approximate grouping distance: rasterized lines are
    dilated by half that distance on the coarse grid. Connected line groups
    become padded proposals, with the largest groups retained first. Grouping
    does not assert collinearity, a rectangular object, or a physical surface.
    """

    def __init__(self, level=2, min_length_px=None, join_px=12,
                 pad_px=20, max_regions=12):
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
        # scale=1.0 avoids LSD's additional internal image resize: the frame
        # stage has already chosen the detection resolution.
        self._detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD, 1.0)

    def detect(self, pyramid: Sequence[np.ndarray]) -> LargeFeatures:
        """Return finite lines and bounded ROI proposals from a gray pyramid.

        Empty pyramids, empty images, tiny coarse levels, and constant images
        produce empty results. Invalid image types or non-pyrDown dimensions
        raise ``ValueError`` rather than silently corrupting coordinates.
        """
        if len(pyramid) == 0:
            return LargeFeatures()
        level = min(self.level, len(pyramid) - 1)
        gray = np.asarray(pyramid[0])
        small = np.asarray(pyramid[level])
        for image in (gray, small):
            if image.ndim != 2 or image.dtype != np.uint8:
                raise ValueError("Expected uint8 grayscale pyramid images")
        if gray.size == 0 or small.size == 0:
            return LargeFeatures()

        h, w = gray.shape
        # pyrDown samples a power-of-two grid even for odd image dimensions.
        # Using w / small.shape[1] would incorrectly shift mapped coordinates.
        scale = 2 ** level
        expected_shape = ((h + scale - 1) // scale,
                          (w + scale - 1) // scale)
        if small.shape != expected_shape:
            raise ValueError("Pyramid dimensions must match consecutive pyrDown levels")
        if min(small.shape) < 8 or int(small.max()) == int(small.min()):
            return LargeFeatures()

        detected = self._detector.detect(np.ascontiguousarray(small))[0]
        if detected is None:
            return LargeFeatures()
        coarse_lines = np.asarray(detected, dtype=np.float32).reshape(-1, 4)
        coarse_lines = coarse_lines[np.isfinite(coarse_lines).all(axis=1)]
        lines = coarse_lines * float(scale)
        lines[:, (0, 2)] = np.clip(lines[:, (0, 2)], 0, w - 1)
        lines[:, (1, 3)] = np.clip(lines[:, (1, 3)], 0, h - 1)
        lengths = np.linalg.norm(lines[:, 2:] - lines[:, :2], axis=1)
        min_length = (max(24.0, 0.06 * min(h, w))
                      if self.min_length_px is None else self.min_length_px)
        lines = lines[lengths >= min_length].astype(np.float32, copy=False)
        if len(lines) == 0 or self.max_regions == 0:
            return LargeFeatures(lines=lines)

        line_mask = np.zeros(small.shape, dtype=np.uint8)
        for line in lines / float(scale):
            start = tuple(np.rint(line[:2]).astype(int))
            end = tuple(np.rint(line[2:]).astype(int))
            cv2.line(line_mask, start, end, 255, 1)
        # Quantization means joins are approximate within a coarse-grid pixel.
        radius = int(np.ceil(self.join_px / (2.0 * scale)))
        # A radius wider than the coarse image has the same grouping effect.
        radius = min(radius, max(small.shape))
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        joined = cv2.dilate(line_mask, kernel)
        count, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
        labels = sorted(range(1, count),
                        key=lambda i: -int(stats[i, cv2.CC_STAT_AREA]))
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
        return LargeFeatures(regions=regions, lines=lines)
