"""Shared OpenCL inputs for LK, GFTT and Canny/dilation.

OpenCV may fall back to CPU. Use from one processing thread; cached source
images are immutable. CPU is the default; set RB3_VISION_BACKEND=opencl to enable OpenCL.
"""
from collections import deque
import os

import cv2
import numpy as np


def _numpy(value):
    return getattr(value, 'get', lambda: value)()


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
    def __init__(self, mode=None):
        self.mode = mode or os.environ.get('RB3_VISION_BACKEND', 'cpu')
        self._activate = {
            'cpu': lambda: None,
            'opencl': lambda: cv2.ocl.setUseOpenCL(True),
        }[self.mode]
        self._convert = {'cpu': lambda image: image, 'opencl': cv2.UMat}[self.mode]
        self._images = _ImageCache(self._convert)
        self._frames = deque(maxlen=2)

    def begin_frame(self, frame):
        self._activate()
        self._frames.append(frame)
        sources = {id(image): image for frame in self._frames
               for image in frame.gray_frames}
        for key in self._images.keys() - sources.keys():
            del self._images[key]
        self._images.sources = sources

    def calcOpticalFlowPyrLK(self, previous, current, points, next_points=None,
                            **parameters):
        points = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 1, 2)
        flags = parameters.pop('flags', 0)
        criteria = parameters.pop('criteria', (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .01))
        # OpenCV 4.12/4.x initializes its OpenCL useInitialFlow member from the
        # MIN_EIGENVALS bit. Set both at construction, then restore the actual
        # flags before calc so seeded LK works without requesting eigenvalue errors.
        initial_flags = flags | (bool(flags & cv2.OPTFLOW_USE_INITIAL_FLOW)
                                 * cv2.OPTFLOW_LK_GET_MIN_EIGENVALS)
        tracker = cv2.SparsePyrLKOpticalFlow_create(crit=criteria, flags=initial_flags, **parameters)
        tracker.setFlags(flags)
        result = tracker.calc(self._images[id(previous)], self._images[id(current)],
                              points, next_points)
        return tuple(map(_numpy, result))

    def goodFeaturesToTrack(self, image, *, mask=None, **parameters):
        points = _numpy(cv2.goodFeaturesToTrack(
            self._images[id(image)], mask=self._convert(mask), **parameters))
        if points is None or mask is None:
            return points
        points = points.reshape(-1, 2)
        pixels = points.astype(int)
        height, width = mask.shape
        inside = (
            np.isfinite(points).all(axis=1)
            & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        )
        valid = np.flatnonzero(inside)
        inside[valid] = mask[pixels[valid, 1], pixels[valid, 0]] > 0
        return points[inside].reshape(-1, 1, 2)

    def Canny(self, image, threshold1, threshold2, **parameters):
        return cv2.Canny(_numpy(self._images[id(image)]), threshold1, threshold2,
                         **parameters)

    def dilate(self, image, kernel, **parameters):
        return _numpy(cv2.dilate(image, kernel, **parameters))
