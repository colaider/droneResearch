"""Camera geometry and matching checks without a rendering window."""
import cv2
import numpy as np
import pytest
from swarm.compVision.stereo_depth import StereoCalibration, StereoDepth, triangulate_matches


def test_triangulation_recovers_metric_xyz_and_rejects_bad_matches():
    c = StereoCalibration()
    xyz = np.array([[0., 0., 3.], [1., .5, 12.], [-2., -1., 20.]])
    pixels = []
    for P in c.projection_matrices:
        uvw = np.column_stack((xyz, np.ones(len(xyz)))) @ P.T
        pixels.append(uvw[:, :2] / uvw[:, 2, None])
    points, valid = triangulate_matches(*pixels, c)
    assert valid.all()
    np.testing.assert_allclose(points, xyz, atol=1e-8)
    right = pixels[1].copy()
    right[0, 1] += 5  # Violates the horizontal epipolar constraint.
    right[1, 0] = pixels[0][1, 0] + 2  # Negative disparity.
    right[2] = np.nan
    points, valid = triangulate_matches(pixels[0], right, c)
    assert not valid.any() and np.isnan(points).all()


def test_depth_formula_and_point_cloud():
    c = StereoCalibration(320, 240, 90., .2)
    s = StereoDepth(c, num_disparities=64)
    depth = s.depth_from_disparity(np.array([[8., 0., -2., np.nan]]))
    assert depth[0, 0] == pytest.approx(3.)
    assert np.isnan(depth[0, 1:]).all()
    image = np.full((240, 320), np.nan, dtype=np.float32)
    image[120, 160] = 3.
    points, rows, cols = s.point_cloud(image)
    np.testing.assert_allclose(points, [[0., 0., 3.]])
    assert (rows[0], cols[0]) == (120, 160)


def test_shifted_texture_recovers_depth_and_sparse_points():
    c = StereoCalibration(320, 240, 65., .2)
    s = StereoDepth(c, num_disparities=64)
    rng = np.random.default_rng(7)
    left = rng.integers(0, 256, (240, 320), dtype=np.uint8)
    left = cv2.GaussianBlur(left, (3, 3), .7)
    right = np.zeros_like(left)
    right[:, :-8] = left[:, 8:]  # Known positive disparity: u_left-u_right=8.
    result = s.compute(left, right)
    assert result['valid'].mean() > .4
    expected = c.focal_px * c.baseline_m / 8
    assert np.nanmedian(result['depth_m']) == pytest.approx(expected, rel=.01)
    assert len(result['points_m']) > 30
    assert np.median(result['points_m'][:, 2]) == pytest.approx(expected, rel=.02)


def test_blank_images_produce_unknown_depth():
    c = StereoCalibration(320, 240)
    s = StereoDepth(c, num_disparities=64)
    image = np.full((240, 320, 3), 128, np.uint8)
    result = s.compute(image, image)
    assert not result['valid'].any()
    assert np.isnan(result['depth_m']).all()
    assert result['points_m'].shape == (0, 3)


def test_invalid_calibration_and_images_are_rejected():
    with pytest.raises(ValueError):
        StereoCalibration(baseline_m=0)
    c = StereoCalibration(320, 240)
    with pytest.raises(ValueError):
        StereoDepth(c, num_disparities=70)
    s = StereoDepth(c, num_disparities=64)
    with pytest.raises(ValueError):
        s.compute(np.zeros((100, 100), np.uint8), np.zeros((100, 100), np.uint8))
