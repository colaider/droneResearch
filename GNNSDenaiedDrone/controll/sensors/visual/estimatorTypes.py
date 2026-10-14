"""Shared estimator data contracts; no camera, renderer or algorithm imports.

Pixels are (u, v), full resolution unless explicitly stated. Capture timestamps
are seconds in one monotonic source clock (simulation seconds in Genesis).
A timestamp does not itself prove stereo synchronization or rectification.
"""
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class StereoFrame:
    """One captured pair of equal-size uint8 BGR or grayscale images.

    Calibration must describe these exact images and their raw/rectified state.
    The caller owns the arrays and must not mutate them during processing;
    preprocessing makes its own copies. The sequence ID identifies this capture.
    Image format is inferred from shape unless explicitly supplied. Native
    (height, width) mono images feed tracking without a BGR-to-gray conversion.
    Conversion and filter settings are supplied to the estimator/frame processor.
    skip_preprocessing can additionally bypass intensity filters for one pair;
    required pyramids remain.
    BGR display copies are still built for the existing colored overlays.
    """
    left: np.ndarray
    right: np.ndarray
    left_timestamp: float
    right_timestamp: float
    sequence_id: int
    pixel_format: Literal['BGR', 'GRAY'] | None = None
    skip_preprocessing: bool = False
    left_timestamp_ns: int | None = None
    right_timestamp_ns: int | None = None

    def __post_init__(self):
        if self.pixel_format is None:
            inferred = 'GRAY' if isinstance(self.left, np.ndarray) and self.left.ndim == 2 else 'BGR'
            object.__setattr__(self, 'pixel_format', inferred)
        if self.pixel_format not in ('BGR', 'GRAY'):
            raise ValueError('pixel_format must be BGR or GRAY')
        if not isinstance(self.skip_preprocessing, bool):
            raise ValueError('skip_preprocessing must be a bool')
        for image in (self.left, self.right):
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                    or not image.size
                    or (self.pixel_format == 'GRAY' and image.ndim != 2)
                    or (self.pixel_format == 'BGR' and
                        (image.ndim != 3 or image.shape[2] != 3))):
                raise ValueError('StereoFrame images must be nonempty uint8 arrays matching pixel_format')
        if self.left.shape != self.right.shape:
            raise ValueError('StereoFrame images must have equal size')
        if not all(np.isfinite(t) for t in (self.left_timestamp, self.right_timestamp)):
            raise ValueError('Capture timestamps must be finite seconds')
        for field_name in ("left_timestamp_ns", "right_timestamp_ns"):
            timestamp = getattr(self, field_name)
            if timestamp is None:
                seconds_field = field_name.removesuffix("_ns")
                timestamp = int(round(float(getattr(self, seconds_field)) * 1e9))
                object.__setattr__(self, field_name, timestamp)
            elif (isinstance(timestamp, (bool, np.bool_))
                  or not isinstance(timestamp, (int, np.integer))):
                raise ValueError(f'{field_name} must be an integer')
            else:
                object.__setattr__(self, field_name, int(timestamp))
        if (isinstance(self.sequence_id, (bool, np.bool_))
                or not isinstance(self.sequence_id, (int, np.integer)) or self.sequence_id < 0):
            raise ValueError('sequence_id must be a nonnegative integer')

    @property
    def timestamp(self):
        return float(self.left_timestamp)

    @property
    def timestamp_ns(self):
        return int(self.left_timestamp_ns)

    @property
    def stereo_skew(self):
        return abs(float(self.right_timestamp) - float(self.left_timestamp))


@dataclass
class Frame:
    """One stereo sample, with full-resolution inputs and coarse grayscale levels.

    ``frames`` contains BGR display images (expanded from gray for mono input).
    Each camera's ``pyramids`` entry contains its full, half, and quarter-resolution
    grayscale images. The
    full-resolution level is the same array as the corresponding gray frame.
    Empty defaults preserve the former ``EnumFrame()`` construction pattern.
    """

    frames: list = field(default_factory=list)
    gray_frames: list = field(default_factory=list)
    pyramids: list = field(default_factory=list)
    idx: int = 0
    left_timestamp: float | None = None
    right_timestamp: float | None = None
    left_timestamp_ns: int | None = None
    right_timestamp_ns: int | None = None

    @property
    def timestamp(self):
        """Left exposure time, used for temporal tracking; None for legacy inputs."""
        return self.left_timestamp

    @property
    def timestamp_ns(self):
        """Exact ROS nanosecond timestamp when sourced from a ROS image message."""
        return self.left_timestamp_ns

    def display_copy(self):
        """Copy color images for overlays while retaining clean tracking inputs.

        Grayscale images and pyramids are shared and must be treated as
        read-only by downstream processing and display code.
        """
        return Frame(
            frames=[image.copy() for image in self.frames],
            gray_frames=self.gray_frames,
            pyramids=self.pyramids,
            idx=self.idx,
            left_timestamp=self.left_timestamp,
            right_timestamp=self.right_timestamp,
            left_timestamp_ns=self.left_timestamp_ns,
            right_timestamp_ns=self.right_timestamp_ns,
        )


@dataclass
class LayerTracks:
    """One layer's current tracks, all expressed in full-resolution pixels.

    The mesh is image-space topology, not a reconstructed physical surface.
    Stereo correspondences/depth are stored separately from temporal tracking.
    Native layer coordinates used to seed LK are kept separately by the estimator.
    """
    layer: int
    frame_idx: int
    old_points: np.ndarray
    new_points: np.ndarray
    flow: np.ndarray
    triangles: np.ndarray
    links: np.ndarray

    @classmethod
    def empty(cls, layer, frame_idx):
        return cls(layer, frame_idx, *(np.empty((0, 2), np.float32) for _ in range(3)),
                   np.empty((0, 3), int), np.empty((0, 2), int))


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
