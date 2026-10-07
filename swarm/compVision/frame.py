"""Stereo frame preprocessing and clean image-pyramid storage.

This module owns image preparation only. Tracking state, feature selection,
depth, and velocity estimation belong to their respective pipeline stages.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Frame:
    """One stereo sample, with full-resolution inputs and coarse grayscale levels.

    ``frames`` contains filtered BGR images. Each camera's ``pyramids`` entry
    contains its full, half, and quarter-resolution grayscale images. The
    full-resolution level is the same array as the corresponding gray frame.
    Empty defaults preserve the former ``EnumFrame()`` construction pattern.
    """

    frames: list = field(default_factory=list)
    gray_frames: list = field(default_factory=list)
    pyramids: list = field(default_factory=list)
    idx: int = 0

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
        )


class FrameProcessor:
    """Apply ordered image filters and prepare a stereo frame without history."""

    SUPPORTED_FILTERS = frozenset(("clahe", "gaussian", "median"))

    def __init__(self, image_filters=("clahe",), filter_kernel_size=3):
        self.image_filters = tuple(image_filters)
        self.filter_kernel_size = filter_kernel_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._validate_configuration()

    def _validate_configuration(self):
        kernel = self.filter_kernel_size
        if (
            isinstance(kernel, (bool, np.bool_))
            or not isinstance(kernel, (int, np.integer))
            or kernel < 1
            or kernel % 2 == 0
        ):
            raise ValueError("filter_kernel_size must be a positive odd integer")
        if isinstance(self.image_filters, str) or any(
            name not in self.SUPPORTED_FILTERS for name in self.image_filters
        ):
            raise ValueError("Supported image filters: clahe, gaussian, median")

    @staticmethod
    def _validate_image(image):
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] == 0
            or image.shape[1] == 0
        ):
            raise ValueError("Each frame must be a nonempty uint8 BGR image")

    def preprocess(self, image):
        """Filter one BGR image, leaving the supplied image unmodified.

        CLAHE follows the existing estimator's LAB-luminance implementation.
        Filters run in the configured order; an empty tuple disables filters.
        Configuration is checked on each call to support runtime settings.
        """
        self._validate_configuration()
        self._validate_image(image)
        result = image.copy()
        kernel = int(self.filter_kernel_size)
        for name in self.image_filters:
            if name == "clahe":
                lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            elif name == "gaussian":
                result = cv2.GaussianBlur(result, (kernel, kernel), 0)
            elif name == "median":
                result = cv2.medianBlur(result, kernel)
        return result

    def process(self, frames, idx):
        """Prepare exactly two equal-size BGR images and their Gaussian pyramids.

        ``pyrDown`` smooths before reducing each image dimension. Tracking and
        stereo matching can keep using the original-resolution gray frames.
        """
        if not isinstance(frames, (list, tuple)) or len(frames) != 2:
            raise ValueError("Expected exactly two stereo images: [left, right]")
        for image in frames:
            self._validate_image(image)
        if frames[0].shape != frames[1].shape:
            raise ValueError("Left and right stereo images must have the same size")

        prepared = [self.preprocess(image) for image in frames]
        gray_frames = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in prepared]
        pyramids = []
        for gray in gray_frames:
            half = cv2.pyrDown(gray)
            pyramids.append([gray, half, cv2.pyrDown(half)])
        return Frame(frames=prepared, gray_frames=gray_frames, pyramids=pyramids, idx=idx)
