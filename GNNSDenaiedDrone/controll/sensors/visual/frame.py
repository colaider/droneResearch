"""Stereo frame preprocessing and clean image-pyramid storage.

This module owns image preparation only. Tracking state, feature selection,
depth, and velocity estimation belong to their respective pipeline stages.
"""

from .estimatorTypes import Frame, StereoFrame

import cv2
import numpy as np


class FrameProcessor:
    """Apply ordered image filters and prepare a stereo frame without history."""

    SUPPORTED_FILTERS = frozenset(("clahe", "gaussian", "median"))

    def __init__(self, image_filters: tuple[str, ...], filter_kernel_size: int,
                 pyramid_levels: int, *, convert_to_grayscale: bool,
                 filters_enabled: bool):
        if not isinstance(pyramid_levels, int) or pyramid_levels < 1:
            raise ValueError("pyramid_levels must be a positive integer")
        self.pyramid_levels = pyramid_levels
        self.image_filters = tuple(image_filters)
        self.convert_to_grayscale = convert_to_grayscale
        self.filters_enabled = filters_enabled
        self.filter_kernel_size = filter_kernel_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._validate_configuration()

    def _validate_configuration(self):
        if not isinstance(self.convert_to_grayscale, bool) or not isinstance(self.filters_enabled, bool):
            raise ValueError("CONVERT_TO_GRAYSCALE and FILTERS_ENABLED must be bools")
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
    def _validate_image(image, *, grayscale=False):
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or (grayscale and image.ndim != 2)
            or (not grayscale and (image.ndim != 3 or image.shape[2] != 3))
            or image.shape[0] == 0
            or image.shape[1] == 0
        ):
            kind = "grayscale" if grayscale else "BGR"
            raise ValueError(f"Each frame must be a nonempty uint8 {kind} image")

    def preprocess(self, image, *, skip_filters=False):
        """Filter a BGR or native grayscale image without modifying the input.

        CLAHE uses LAB luminance for BGR and operates directly on mono images.
        skip_filters or an empty image_filters tuple bypasses intensity filters.
        Configuration is checked on each call to support runtime settings.
        """
        self._validate_configuration()
        grayscale = isinstance(image, np.ndarray) and image.ndim == 2
        self._validate_image(image, grayscale=grayscale)
        result = image.copy()
        if skip_filters or not self.filters_enabled:
            return result
        kernel = int(self.filter_kernel_size)
        for name in self.image_filters:
            if name == "clahe":
                if grayscale:
                    result = self.clahe.apply(result)
                else:
                    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
                    lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
                    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            elif name == "gaussian":
                result = cv2.GaussianBlur(result, (kernel, kernel), 0)
            elif name == "median":
                result = cv2.medianBlur(result, kernel)
        return result

    def process(self, frames, idx=None):
        """Build tracking inputs and BGR display copies from a stereo pair.

        Native grayscale is detected from shape for typed pairs and image lists.
        CONVERT_TO_GRAYSCALE=False requires native grayscale input.
        FILTERS_ENABLED=False bypasses filters; a pair can also opt out using
        skip_preprocessing=True. All processor options are supplied at creation.

        ``pyrDown`` smooths before reducing each image dimension. Tracking and
        stereo matching can keep using the original-resolution gray frames.
        """
        left_timestamp = right_timestamp = None
        left_timestamp_ns = right_timestamp_ns = None
        skip_filters = False
        if isinstance(frames, StereoFrame):
            if idx is not None and idx != frames.sequence_id:
                raise ValueError('idx differs from StereoFrame.sequence_id')
            idx = frames.sequence_id
            skip_filters = frames.skip_preprocessing
            left_timestamp, right_timestamp = frames.left_timestamp, frames.right_timestamp
            left_timestamp_ns = frames.left_timestamp_ns
            right_timestamp_ns = frames.right_timestamp_ns
            frames = [frames.left, frames.right]
        elif idx is None:
            raise ValueError('Legacy image lists require a frame index')
        if not isinstance(frames, (list, tuple)) or len(frames) != 2:
            raise ValueError("Expected exactly two stereo images: [left, right]")
        grayscale = isinstance(frames[0], np.ndarray) and frames[0].ndim == 2
        self._validate_configuration()
        if not grayscale and not self.convert_to_grayscale:
            raise ValueError("CONVERT_TO_GRAYSCALE=False requires native grayscale images (H, W)")
        if grayscale:
            for image in frames:
                self._validate_image(image, grayscale=True)
        else:
            frames = [prepare_camera_frame(image) for image in frames]
        if frames[0].shape != frames[1].shape:
            raise ValueError("Left and right stereo images must have the same size")

        filtered = [self.preprocess(image, skip_filters=skip_filters) for image in frames]
        if grayscale:
            gray_frames = filtered
            # Color expansion is only for existing overlays, never a tracking input.
            prepared = [cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in filtered]
        else:
            prepared = filtered
            gray_frames = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in prepared]
        pyramids = []
        for gray in gray_frames:
            levels = [gray]
            for _ in range(1, self.pyramid_levels):
                levels.append(cv2.pyrDown(levels[-1]))
            pyramids.append(levels)
        return Frame(frames=prepared, gray_frames=gray_frames, pyramids=pyramids,
                     idx=idx, left_timestamp=left_timestamp,
                     right_timestamp=right_timestamp,
                     left_timestamp_ns=left_timestamp_ns,
                     right_timestamp_ns=right_timestamp_ns)


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
