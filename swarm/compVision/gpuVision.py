"""Optional OpenCL-backed OpenCV operations for the simulation vision pipeline.

CPU is the default. Set SWARM_VISION_BACKEND=opencl to enable OpenCL where
the installed OpenCV build supports it.
"""
from collections import deque
import os

import cv2
import numpy as np


def _numpy(value):
    return getattr(value, "get", lambda: value)()


class _ImageCache(dict):
    def __init__(self, convert):
        super().__init__()
        self.convert = convert
        self.sources = {}

    def __missing__(self, key):
        image = self.convert(self.sources[key])
        self[key] = image
        return image


class VisionBackend:
    """Share cached frame inputs across CPU or OpenCL OpenCV operations."""

    def __init__(self, mode=None):
        self.mode = mode or os.environ.get("SWARM_VISION_BACKEND", "cpu")
        if self.mode not in ("cpu", "opencl"):
            raise ValueError("Vision backend must be 'cpu' or 'opencl'")
        self._convert = cv2.UMat if self.mode == "opencl" else lambda image: image
        self._images = _ImageCache(self._convert)
        self._frames = deque(maxlen=2)

    def begin_frame(self, frame):
        if self.mode == "opencl":
            cv2.ocl.setUseOpenCL(True)
        self._frames.append(frame)
        sources = {
            id(image): image
            for cached_frame in self._frames
            for image in cached_frame.gray_frames
        }
        for key in self._images.keys() - sources.keys():
            del self._images[key]
        self._images.sources = sources

    def calcOpticalFlowPyrLK(self, previous, current, points, next_points=None,
                            **parameters):
        if self.mode == "cpu":
            result = cv2.calcOpticalFlowPyrLK(
                previous, current, points, next_points, **parameters
            )
        else:
            points = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 1, 2)
            flags = parameters.pop("flags", 0)
            criteria = parameters.pop(
                "criteria",
                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .01),
            )
            initial_flags = flags | (
                bool(flags & cv2.OPTFLOW_USE_INITIAL_FLOW)
                * cv2.OPTFLOW_LK_GET_MIN_EIGENVALS
            )
            tracker = cv2.SparsePyrLKOpticalFlow_create(
                crit=criteria, flags=initial_flags, **parameters
            )
            tracker.setFlags(flags)
            result = tracker.calc(
                self._images[id(previous)], self._images[id(current)],
                points, next_points,
            )
        return tuple(map(_numpy, result))

    def goodFeaturesToTrack(self, image, *, mask=None, **parameters):
        converted_mask = None if mask is None else self._convert(mask)
        return _numpy(cv2.goodFeaturesToTrack(
            self._images[id(image)], mask=converted_mask, **parameters
        ))

    def Canny(self, image, threshold1, threshold2, **parameters):
        return _numpy(cv2.Canny(
            self._images[id(image)], threshold1, threshold2, **parameters
        ))

    def dilate(self, image, kernel, **parameters):
        return _numpy(cv2.dilate(self._convert(image), kernel, **parameters))
