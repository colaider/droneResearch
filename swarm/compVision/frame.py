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

    def __init__(self, image_filters=("clahe",), filter_kernel_size=3,
                 pyramid_levels=3):
        if not isinstance(pyramid_levels, int) or pyramid_levels < 1:
            raise ValueError("pyramid_levels must be a positive integer")
        self.pyramid_levels = pyramid_levels
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
        """Normalize two equal-size images to BGR and build grayscale/pyramid inputs.

        ``pyrDown`` smooths before reducing each image dimension. Tracking and
        stereo matching can keep using the original-resolution gray frames.
        """
        if not isinstance(frames, (list, tuple)) or len(frames) != 2:
            raise ValueError("Expected exactly two stereo images: [left, right]")
        frames = [prepare_camera_frame(image) for image in frames]
        if frames[0].shape != frames[1].shape:
            raise ValueError("Left and right stereo images must have the same size")

        prepared = [self.preprocess(image) for image in frames]
        gray_frames = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in prepared]
        pyramids = []
        for gray in gray_frames:
            levels = [gray]
            for _ in range(1, self.pyramid_levels):
                levels.append(cv2.pyrDown(levels[-1]))
            pyramids.append(levels)
        return Frame(frames=prepared, gray_frames=gray_frames, pyramids=pyramids,
                     idx=idx)


def prepare_camera_frame(image, *, color_order="BGR", rotation=None, resolution=None):
    """Normalize a runner-provided uint8 image to BGR without changing its input.

    resolution is optional (width, height), applied before rotation to preserve
    the existing USB-runner convention. The caller must provide calibration for
    the resulting pixel scale/orientation. This does not stereo-rectify images.
    No camera is opened or read here.
    """
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
            image.ndim not in (2, 3) or not image.size):
        raise ValueError("Camera frame must be a nonempty uint8 image")
    if color_order == "GRAY":
        if image.ndim != 2:
            raise ValueError("GRAY input must have shape (height, width)")
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif color_order in ("BGR", "RGB"):
        FrameProcessor._validate_image(image)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if color_order == "RGB" else image.copy()
    else:
        raise ValueError("color_order must be BGR, RGB or GRAY")
    if resolution is not None:
        if (len(resolution) != 2 or any(isinstance(v, (bool, np.bool_)) or
                not isinstance(v, (int, np.integer)) or v <= 0 for v in resolution)):
            raise ValueError("resolution must be positive integer (width, height)")
        resolution = tuple(int(v) for v in resolution)
        if (result.shape[1], result.shape[0]) != resolution:
            result = cv2.resize(result, resolution, interpolation=cv2.INTER_AREA)
    if rotation is not None:
        if rotation not in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
            raise ValueError("rotation must be an OpenCV rotation constant or None")
        result = cv2.rotate(result, rotation)
    return result


def compose_camera_display(processed, tracking_gray):
        """Filtered color above the exact grayscale inputs used for feature tracking."""
        if len(processed) < 2 or len(tracking_gray) < 2:
            return None
        h, w = processed[0].shape[:2]
        scale = min(640 / w, 450 / h, 1.0)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))

        def tile(frame, label, grayscale=False):
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            image = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            image = cv2.copyMakeBorder(image, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
            cv2.putText(image, label, (10, 19), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (235, 235, 235), 1, cv2.LINE_AA)
            return image

        top = cv2.hconcat([tile(processed[0], "Left camera"), tile(processed[1], "Right camera")])
        bottom = cv2.hconcat([tile(tracking_gray[0], "Left tracking input", True), tile(tracking_gray[1], "Right tracking input", True)])
        return cv2.vconcat([top, bottom])
