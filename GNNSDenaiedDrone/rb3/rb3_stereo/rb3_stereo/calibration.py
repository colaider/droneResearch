"""Stereo calibration of the OV9282 pair: checkerboard corners, calibration, rectification.

The calibration file (default rb3_stereo/config/stereo_calib.npz, see default_path) holds the raw
camera model K1 D1 K2 D2 R T size in the same format as scripts/calibrate_stereo.py, so
swarm.compVision.denseStereo.StereoRectifier.from_npz reads it too. It also records how it was made
(board, left camera, flip). StereoCamera loads it and rectifies every pair with a Rectifier.
640x400 is the 1280x800 view binned 2x2, so one calibration serves both resolutions.
No ROS dependency.
"""
import datetime
import os
from pathlib import Path

import cv2
import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "stereo_calib.npz"


def default_path():
    """$RB3_STEREO_CALIBRATION, else config/stereo_calib.npz of this package's source checkout
    (the drone's ~/droneResearch/rb3_stereo with the symlink install), else of the installed share."""
    env = os.environ.get("RB3_STEREO_CALIBRATION")
    if env:
        return Path(env).expanduser()
    if not DEFAULT_PATH.exists():
        try:
            from ament_index_python.packages import get_package_share_directory
            share = Path(get_package_share_directory("rb3_stereo")) / "config" / "stereo_calib.npz"
            if share.exists():
                return share
        except Exception:
            pass
    return DEFAULT_PATH


class Board:
    """Checkerboard of `squares` = (columns, rows) squares of `square_m` metres. OpenCV finds the
    inner corners: (columns - 1) x (rows - 1)."""

    def __init__(self, squares=(9, 7), square_m=0.029):
        self.squares, self.square_m = (int(squares[0]), int(squares[1])), float(square_m)
        self.size = (self.squares[0] - 1, self.squares[1] - 1)
        grid = np.mgrid[0:self.size[0], 0:self.size[1]].T.reshape(-1, 2) * self.square_m
        self.object_points = np.hstack([grid, np.zeros((len(grid), 1))]).astype(np.float32)

    def find(self, gray, fast=False):
        """Inner corners as (N, 2) float32 in image pixels, or None. fast: the classic detector
        (a few ms when the board is visible, for the live preview); otherwise the more accurate
        sector-based one (findChessboardCornersSB) for the corners that go into the calibration."""
        if fast:
            ok, c = cv2.findChessboardCorners(gray, self.size, cv2.CALIB_CB_ADAPTIVE_THRESH
                                              | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
        else:
            ok, c = cv2.findChessboardCornersSB(gray, self.size, cv2.CALIB_CB_NORMALIZE_IMAGE)
            if not ok:
                ok, c = cv2.findChessboardCorners(gray, self.size, cv2.CALIB_CB_ADAPTIVE_THRESH
                                                  | cv2.CALIB_CB_NORMALIZE_IMAGE)
                if ok:
                    c = cv2.cornerSubPix(gray, c, (5, 5), (-1, -1),
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01))
        return c.reshape(-1, 2).astype(np.float32) if ok else None


def orient(corners, like=None):
    """Number the corners consistently. An even x even corner grid looks the same turned by 180
    degrees, so the detector may count from either end - and the left and right views of one pair
    must agree. Counts from the top-left end, or the same way round as `like` (the other view)."""
    ref = np.array([1.0, 1.0]) if like is None else like[-1] - like[0]
    return corners[::-1].copy() if (corners[-1] - corners[0]) @ ref < 0 else corners


def calibrate(left_points, right_points, board, size, max_view_error=None):
    """Calibrate both cameras and the pair from per-view corners (lists of (N, 2) arrays, the same
    views in the same order, oriented alike). Views whose reprojection error is far above the rest
    (2.5x the median, at least 0.5 px, or max_view_error) are dropped once and the calibration
    repeated. Returns a dict with K1 D1 K2 D2 R T size (the calibrate_stereo.py format), the
    errors in pixels and `used`, the indices of the views kept."""
    size = (int(size[0]), int(size[1]))
    used = list(range(len(left_points)))
    for attempt in range(2):
        objs = [board.object_points] * len(used)
        lp = [left_points[i].reshape(-1, 1, 2) for i in used]
        rp = [right_points[i].reshape(-1, 1, 2) for i in used]
        rms_l, K1, D1, *_, err_l = cv2.calibrateCameraExtended(objs, lp, size, None, None)
        rms_r, K2, D2, *_, err_r = cv2.calibrateCameraExtended(objs, rp, size, None, None)
        worst = np.maximum(err_l.ravel(), err_r.ravel())
        limit = max_view_error or max(2.5 * float(np.median(worst)), 0.5)
        keep = worst <= limit
        if attempt or keep.all() or keep.sum() < 8:
            break
        used = [i for i, k in zip(used, keep) if k]
    rms, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(
        objs, lp, rp, K1, D1, K2, D2, size, flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6))
    return dict(K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, size=np.array(size),
                rms_left=float(rms_l), rms_right=float(rms_r), rms_stereo=float(rms), used=used)


def save(path, calib, board, left_camera="cam0a", flip=False):
    """Write the .npz plus left.yaml / right.yaml (ROS camera_calibration format, the rectification
    at the calibration size) next to it. Returns the paths written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(calib[k]) for k in ("K1", "D1", "K2", "D2", "R", "T", "size")}
    meta = dict(rms_left=calib["rms_left"], rms_right=calib["rms_right"], rms_stereo=calib["rms_stereo"],
                pairs=len(calib["used"]), squares=np.array(board.squares), square_m=board.square_m,
                left_camera=str(left_camera), flip=bool(flip),
                created=datetime.datetime.now().isoformat(timespec="seconds"))
    np.savez(path, **arrays, **meta)
    rect = Rectifier(calib, tuple(int(x) for x in calib["size"]))
    written = [path]
    for side, K, D, Rr, P in (("left", calib["K1"], calib["D1"], rect.R1, rect.P1),
                              ("right", calib["K2"], calib["D2"], rect.R2, rect.P2)):
        out = path.with_name(f"{side}.yaml")
        out.write_text(_ros_yaml(side, rect.size, K, D, Rr, P))
        written.append(out)
    return written


def load(path):
    """Calibration dict from save() (or calibrate_stereo.py: then without left_camera/flip)."""
    with np.load(path) as data:
        calib = {k: data[k] for k in data.files}
    for k in ("left_camera", "created"):
        if k in calib:
            calib[k] = str(calib[k])
    if "flip" in calib:
        calib["flip"] = bool(calib["flip"])
    calib["path"] = str(path)
    return calib


def scale_camera_matrix(K, s):
    """K for the same view at s times the resolution (pixel centres: x' = (x + 0.5) s - 0.5)."""
    K = np.array(K, float)
    K[0, 0] *= s
    K[1, 1] *= s
    K[0, 2] = (K[0, 2] + 0.5) * s - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    return K


class Rectifier:
    """Undistorts and rectifies pairs of `size` = (width, height); a calibration made at another
    resolution with the same aspect ratio (1280x800 <-> 640x400) is scaled. Afterwards both images
    share one pinhole model: focal_px, principal point, and a baseline (m) along
    disparity_direction (right minus left image shift). Same attributes as
    swarm.compVision.denseStereo.StereoRectifier, so the swarm code can use either.
    alpha 0 crops to valid pixels (no black borders, slightly narrower view); 1 keeps every pixel.
    The remapping runs on the GPU (OpenCL) when it is available and gives the same result as the
    CPU: on the RB3 that is 3-6x less CPU time than cv2.remap on a little core. RB3_STEREO_OPENCL=0
    forces the CPU."""

    is_identity = False

    def __init__(self, calib, size, alpha=0.0):
        w0, h0 = (int(x) for x in calib["size"])
        self.size = (int(size[0]), int(size[1]))
        s = self.size[0] / w0
        if abs(self.size[1] / h0 - s) > 1e-6:
            raise ValueError(f"calibration is {w0}x{h0}, images are {self.size[0]}x{self.size[1]}")
        K1, K2 = scale_camera_matrix(calib["K1"], s), scale_camera_matrix(calib["K2"], s)
        D1, D2 = np.asarray(calib["D1"], float), np.asarray(calib["D2"], float)
        R, T = np.asarray(calib["R"], float), np.asarray(calib["T"], float).reshape(3, 1)
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            K1, D1, K2, D2, self.size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
        self.maps = (cv2.initUndistortRectifyMap(K1, D1, self.R1, self.P1, self.size, cv2.CV_16SC2),
                     cv2.initUndistortRectifyMap(K2, D2, self.R2, self.P2, self.size, cv2.CV_16SC2))
        self._raw = (K1, D1), (K2, D2)
        self.focal_px = float(self.P1[0, 0])
        shift = self.P2[:2, 3] / self.focal_px          # P2 = [f 0 cx f*Tx; 0 f cy f*Ty; 0 0 1 0]
        self.baseline = float(np.linalg.norm(shift))
        self.disparity_direction = shift / self.baseline
        self.path = calib.get("path")
        self._gpu_maps, self.backend = None, "cpu"
        if os.environ.get("RB3_STEREO_OPENCL", "1") != "0":
            self._try_opencl()

    def _try_opencl(self):
        try:
            if not cv2.ocl.haveOpenCL():
                return
            cv2.ocl.setUseOpenCL(True)
            maps = tuple((cv2.UMat(a), cv2.UMat(b)) for a, b in self.maps)
            test = np.random.default_rng(0).integers(0, 256, (self.size[1], self.size[0]), dtype=np.uint8)
            gpu = cv2.remap(cv2.UMat(test), *maps[0], cv2.INTER_LINEAR).get()
            cpu = cv2.remap(test, *self.maps[0], cv2.INTER_LINEAR)
            if np.abs(gpu.astype(np.int16) - cpu).max() <= 2:
                self._gpu_maps, self.backend = maps, "opencl"
        except cv2.error:
            pass

    def rectify(self, left, right):
        if self._gpu_maps is not None:
            (l1, l2), (r1, r2) = self._gpu_maps
            a = cv2.remap(cv2.UMat(left), l1, l2, cv2.INTER_LINEAR)     # both queued before waiting
            b = cv2.remap(cv2.UMat(right), r1, r2, cv2.INTER_LINEAR)
            return a.get(), b.get()
        (lx, ly), (rx, ry) = self.maps
        return (cv2.remap(left, lx, ly, cv2.INTER_LINEAR),
                cv2.remap(right, rx, ry, cv2.INTER_LINEAR))

    def rectify_points(self, points, side):
        """Raw-image pixel coordinates -> rectified ones (side: 'left' or 'right')."""
        K, D = self._raw[0 if side == "left" else 1]
        Rr, P = (self.R1, self.P1) if side == "left" else (self.R2, self.P2)
        return cv2.undistortPoints(np.asarray(points, np.float32).reshape(-1, 1, 2), K, D, R=Rr,
                                   P=P).reshape(-1, 2)

    def row_error(self, left_points, right_points):
        """Mean distance (px) of matching rectified points from a common epipolar line: ~0 for a
        good calibration. Points are rectified-image coordinates."""
        d = np.asarray(right_points, float) - np.asarray(left_points, float)
        normal = np.array([-self.disparity_direction[1], self.disparity_direction[0]])
        return float(np.abs(d @ normal).mean())

    @property
    def vfov_deg(self):
        return float(np.degrees(2 * np.arctan(self.size[1] / (2 * self.focal_px))))

    @property
    def hfov_deg(self):
        return float(np.degrees(2 * np.arctan(self.size[0] / (2 * self.focal_px))))


def _ros_yaml(name, size, K, D, R, P):
    def mat(a, rows, cols):
        data = ", ".join(f"{float(x):.10g}" for x in np.asarray(a, float).ravel())
        return f"  rows: {rows}\n  cols: {cols}\n  data: [{data}]\n"
    D = np.asarray(D, float).ravel()
    return (f"image_width: {size[0]}\nimage_height: {size[1]}\ncamera_name: {name}\n"
            f"camera_matrix:\n{mat(K, 3, 3)}distortion_model: plumb_bob\n"
            f"distortion_coefficients:\n{mat(D, 1, len(D))}rectification_matrix:\n{mat(R, 3, 3)}"
            f"projection_matrix:\n{mat(P, 3, 4)}")
