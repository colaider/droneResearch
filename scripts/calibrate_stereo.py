"""Calibrate a stereo camera pair with a checkerboard -> .npz for StereoRectifier.from_npz.

Each camera is calibrated on its own first (cv2.calibrateCamera), then the pair with fixed
intrinsics (cv2.stereoCalibrate). The images must be prepared EXACTLY like the runner that will
use the file: the live USB mode below rotates/resizes with testRealCameras.prepare_frame.

Live USB cameras (space = keep pair when the board is seen by both, c = calibrate, q = quit):
    python scripts/calibrate_stereo.py --cameras 0 1 --board 9x6 --square 0.025
Saved image pairs (left*/right* files, sorted names pair up):
    python scripts/calibrate_stereo.py --images calib_pairs/ --board 9x6 --square 0.025
Recorded pairs from the live mode are kept in --save-dir so the calibration can be re-run.

The RB3 rig already has ROS camera_calibration left.yaml/right.yaml; use
StereoRectifier.from_ros_yaml for those instead.
--board is the number of INNER corners (columns x rows); --square is the square size in metres.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from swarm.compVision.denseStereo import StereoRectifier   # noqa: E402


def board_object_points(cols, rows, square):
    grid = np.zeros((rows * cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return grid


def find_corners(gray, cols, rows):
    """Subpixel inner corners, or None."""
    ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows), cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        ok, corners = cv2.findChessboardCorners(gray, (cols, rows))
        if not ok:
            return None
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
    return corners.reshape(-1, 1, 2).astype(np.float32)


def to_gray(image):
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def calibrate(pairs, cols, rows, square):
    """pairs: list of (left, right) images of the same size. Returns a dict for np.savez."""
    obj = board_object_points(cols, rows, square)
    objs, left_pts, right_pts = [], [], []
    size = None
    for left, right in pairs:
        gl, gr = to_gray(left), to_gray(right)
        if size is None:
            size = (gl.shape[1], gl.shape[0])
        if (gl.shape[1], gl.shape[0]) != size or gl.shape != gr.shape:
            raise ValueError('All images must have the same size')
        cl, cr = find_corners(gl, cols, rows), find_corners(gr, cols, rows)
        if cl is not None and cr is not None:
            objs.append(obj)
            left_pts.append(cl)
            right_pts.append(cr)
    if len(objs) < 8:
        raise ValueError(f'Board found in both images of only {len(objs)} pairs; need at least 8')

    rms_l, K1, D1, _, _ = cv2.calibrateCamera(objs, left_pts, size, None, None)
    rms_r, K2, D2, _, _ = cv2.calibrateCamera(objs, right_pts, size, None, None)
    rms, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(
        objs, left_pts, right_pts, K1, D1, K2, D2, size, flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    print(f'pairs used: {len(objs)} | rms left {rms_l:.3f}px right {rms_r:.3f}px stereo {rms:.3f}px')
    return dict(K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, size=np.array(size), rms=rms,
                left_points=np.array(left_pts), right_points=np.array(right_pts))


def epipolar_error(calib):
    """Mean off-epipolar distance (px) of the board corners after rectification."""
    rect = StereoRectifier.from_calibration(calib['K1'], calib['D1'], calib['K2'], calib['D2'],
                                            calib['R'], calib['T'], tuple(calib['size']))
    size = tuple(calib['size'])
    R1, R2, P1, P2, *_ = cv2.stereoRectify(calib['K1'], calib['D1'], calib['K2'], calib['D2'],
                                           size, calib['R'], calib['T'],
                                           flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0)
    pl = cv2.undistortPoints(calib['left_points'].reshape(-1, 1, 2), calib['K1'], calib['D1'], R=R1, P=P1)
    pr = cv2.undistortPoints(calib['right_points'].reshape(-1, 1, 2), calib['K2'], calib['D2'], R=R2, P=P2)
    # Corners must line up along the axis perpendicular to the disparity.
    across = 1 if abs(rect.disparity_direction[0]) > abs(rect.disparity_direction[1]) else 0
    return rect, float(np.mean(np.abs(pl[:, 0, across] - pr[:, 0, across])))


def load_pairs(folder):
    folder = Path(folder)
    lefts = sorted(p for p in folder.iterdir() if p.name.lower().startswith('left'))
    rights = sorted(p for p in folder.iterdir() if p.name.lower().startswith('right'))
    if not lefts or len(lefts) != len(rights):
        raise ValueError(f'{folder}: expected matching left*/right* image files, got {len(lefts)}/{len(rights)}')
    return [(cv2.imread(str(a)), cv2.imread(str(b))) for a, b in zip(lefts, rights)]


def capture_pairs(left_idx, right_idx, rotate, cols, rows, save_dir):
    """Live capture with the same preparation as realCameraTest/testRealCameras.py."""
    sys.path.insert(0, str(Path(_REPO_ROOT) / 'realCameraTest'))
    from testRealCameras import RES, open_camera, prepare_frame

    left_rot = cv2.ROTATE_90_COUNTERCLOCKWISE if rotate else None
    right_rot = cv2.ROTATE_90_CLOCKWISE if rotate else None
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    left, right = open_camera(left_idx, RES), open_camera(right_idx, RES)
    pairs = []
    try:
        while True:
            gl, gr = left.grab(), right.grab()
            ok_l, fl = left.retrieve() if gl else (False, None)
            ok_r, fr = right.retrieve() if gr else (False, None)
            if not ok_l or not ok_r:
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
                continue
            fl, fr = prepare_frame(fl, left_rot), prepare_frame(fr, right_rot)
            small = [cv2.resize(f, None, fx=0.5, fy=0.5) for f in (fl, fr)]
            found = [find_corners(to_gray(s), cols, rows) for s in small]
            view = []
            for s, c in zip(small, found):
                s = s.copy()
                if c is not None:
                    cv2.drawChessboardCorners(s, (cols, rows), c, True)
                view.append(s)
            view = cv2.hconcat(view)
            cv2.putText(view, f'pairs {len(pairs)} | space keep, c calibrate, q quit', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('stereo calibration', view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and all(c is not None for c in found):
                n = len(pairs)
                cv2.imwrite(str(save_dir / f'left_{n:03d}.png'), fl)
                cv2.imwrite(str(save_dir / f'right_{n:03d}.png'), fr)
                pairs.append((fl, fr))
            elif key == ord('c'):
                break
            elif key == ord('q'):
                return []
    finally:
        left.release()
        right.release()
        cv2.destroyAllWindows()
    return pairs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--images', help='folder with left*/right* image pairs')
    src.add_argument('--cameras', nargs=2, type=int, metavar=('LEFT', 'RIGHT'), help='live USB camera indices')
    p.add_argument('--board', default='9x6', help='inner corners COLSxROWS (default 9x6)')
    p.add_argument('--square', type=float, required=True, help='checkerboard square size in metres')
    p.add_argument('--no-rotate', action='store_true', help='live mode: skip the 90-degree mounting rotation')
    p.add_argument('--save-dir', default='calib_pairs', help='live mode: where kept pairs are written')
    p.add_argument('--out', default='swarm/config/stereo_calib.npz', help='output calibration file')
    return p.parse_args(argv)


def main(args):
    cols, rows = (int(v) for v in args.board.lower().split('x'))
    if args.images:
        pairs = load_pairs(args.images)
    else:
        pairs = capture_pairs(*args.cameras, not args.no_rotate, cols, rows, args.save_dir)
    if not pairs:
        print('no pairs, nothing calibrated')
        return 1
    calib = calibrate(pairs, cols, rows, args.square)
    rect, err = epipolar_error(calib)
    print(f'rectified focal {rect.focal_px:.1f}px | baseline {rect.baseline:.4f} m | '
          f'disparity direction {rect.disparity_direction.round(3)} | epipolar error {err:.2f}px')
    if err > 1.0:
        print('WARNING: epipolar error above 1 px; add more varied board poses (tilts, corners of the view)')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: v for k, v in calib.items() if k not in ('left_points', 'right_points')})
    print(f'saved {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(parse_args()))
