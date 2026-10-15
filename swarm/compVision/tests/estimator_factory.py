"""Explicit visual-estimator construction helpers for computer-vision tests."""

import numpy as np

from swarm.config.cameraConfig import StereoCameraConfig
from swarm.compVision.visualEst import VisulaEst


def make_estimator(
    res=(640, 480),
    fov=60,
    *,
    focal_px=None,
    baseline=0.12,
    disparity_direction=(-1.0, 0.0),
    image_filters=("clahe",),
    convert_to_grayscale=True,
    filters_enabled=True,
    filter_kernel_size=3,
    max_stereo_skew_s=0.005,
):
    focal = (
        (res[1] / 2) / np.tan(np.deg2rad(fov) / 2)
        if focal_px is None else focal_px
    )
    camera_config = StereoCameraConfig(
        res=res,
        left_focal=focal,
        right_focal=focal,
        baseline=baseline,
        epipolar_direction=disparity_direction,
    )
    return VisulaEst(
        camera_config,
        image_filters=image_filters,
        convert_to_grayscale=convert_to_grayscale,
        filters_enabled=filters_enabled,
        filter_kernel_size=filter_kernel_size,
        max_stereo_skew_s=max_stereo_skew_s,
    )
