"""Test the real swarm estimator with two USB cameras -- no duplicate estimator.

This drives swarm.compVision.visualAccEst.VisulaAcEst directly: it feeds the two camera
frames and supplies every input the estimator reads each step. There is no flight
controller, so all drone/IMU inputs are provided as ZERO. Only the camera calibration
(focal length, baseline) comes from camera_params.txt.

Run directly or with `python -m realCameraTest.testRealCameras`.
Keys: ESC/q quits; s swaps the dominant camera and resets tracking.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Allow running the file directly from any working directory.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from swarm.compVision.visualAccEst import VisulaAcEst   # the real estimator, used as-is
from swarm.compVision.frame import prepare_camera_frame

if __package__:
    from .display import compose_camera_display
else:
    from display import compose_camera_display


# Camera calibration from camera_params.txt (focal lengths in px at this capture resolution).
LEFT_FOCAL = (1718.54 + 1720.12) / 2.0
RIGHT_FOCAL = (1175.14 + 1177.06) / 2.0
BASELINE = 0.125
RES = (1920, 1080)
WINDOW = "Stereo cameras"


def zero_drone_inputs(est):
    """Supply every drone/IMU input the estimator reads as ZERO (no flight controller).

    These are the same attributes the drone controller normally sets each step; with no
    drone we hand the estimator zeros so there is no invented motion/attitude compensation.
    `drone_ang` is included because the estimator's velocity path reads it.
    """
    est.drone_pos = np.zeros(3)
    est.drone_vel = np.zeros(3)
    est.drone_ang_vel = np.zeros(3)
    est.imu_att = np.zeros(3)
    est.previous_cmd_vel = np.zeros(4)
    est.drone_ang = np.zeros(2)
    est.push_sensors()


def make_estimator(focal_px, res, filters=None, kernel_size=None):
    """Instantiate the real estimator and set the camera calibration it uses."""
    fov = float(np.degrees(2.0 * np.arctan(res[1] / (2.0 * focal_px))))
    est = VisulaAcEst(res, fov)
    est.foc_l = float(focal_px)       # exact focal from camera_params.txt
    est.camera_saperation = BASELINE  # stereo baseline (m)
    # The USB images are rotated/resized but not stereo-rectified to a common
    # projection. Enable only after supplying that calibration and rectification.
    est.use_stereo_depth = False
    if filters is not None:
        est.image_filters = tuple(filters)
    if kernel_size is not None:
        est.filter_kernel_size = kernel_size
    return est


def open_camera(index, res):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def prepare_frame(frame, rotation):
    """Compatibility adapter; image preparation lives in frame.py."""
    return prepare_camera_frame(frame, resolution=RES, rotation=rotation)


def main(left_idx=0, right_idx=1, *, filters=None, kernel_size=None, rotate=True):
    if left_idx == right_idx:
        raise ValueError("Choose two different camera indices")
    if filters is not None and any(name not in ('clahe', 'gaussian', 'median') for name in filters):
        raise ValueError("Supported filters: clahe, gaussian, median")
    if kernel_size is not None and (kernel_size < 1 or kernel_size % 2 == 0):
        raise ValueError("kernel_size must be a positive odd integer")

    # Zeroed drone height leaves depth undefined when stereo finds no match; silence the
    # resulting divide warnings (those frames fall back to a zero velocity by design).
    np.seterr(divide='ignore', invalid='ignore')

    left = right = None
    left_focal, right_focal = LEFT_FOCAL, RIGHT_FOCAL
    left_rotation = cv2.ROTATE_90_COUNTERCLOCKWISE if rotate else None
    right_rotation = cv2.ROTATE_90_CLOCKWISE if rotate else None
    tracking_res = (RES[1], RES[0]) if rotate else RES   # 90-deg rotation swaps W/H

    try:
        left = open_camera(left_idx, RES)
        right = open_camera(right_idx, RES)
        if not left.isOpened() or not right.isOpened():
            print(f"ERROR: could not open cameras left={left_idx}, right={right_idx}")
            return 1

        est = make_estimator(left_focal, tracking_res, filters, kernel_size)
        print(f"left={left_idx} right={right_idx} | focal={left_focal:.1f}px "
              f"baseline={BASELINE}m | filters={est.image_filters}")

        idx, failures = 0, 0
        previous_time = None
        window_ready = False
        while True:
            # Grab both cameras before decoding either frame to reduce timing skew.
            grabbed_l, grabbed_r = left.grab(), right.grab()
            ok_l, fl = left.retrieve() if grabbed_l else (False, None)
            ok_r, fr = right.retrieve() if grabbed_r else (False, None)
            if not ok_l or not ok_r or fl is None or fr is None:
                failures += 1
                if failures == 1:
                    print("WARNING: camera frame grab failed; retrying (q/ESC quits).")
                if cv2.waitKey(20) & 0xFF in (27, ord('q')):
                    break
                if failures >= 30:
                    print("ERROR: cameras failed to deliver 30 consecutive frame pairs.")
                    return 1
                continue
            failures = 0

            now = time.monotonic()
            est.set_dt(max(now - previous_time, 1e-3) if previous_time is not None else 1 / 30)
            previous_time = now

            fl, fr = prepare_frame(fl, left_rotation), prepare_frame(fr, right_rotation)

            zero_drone_inputs(est)                 # no drone -> all drone inputs are 0
            result = est.processing([fl, fr], idx)
            idx += 1

            display = compose_camera_display(result.frames, result.gray_frames)
            vx, vy, _, yaw = est.camera_velocity   # [vx, vy, vz, yaw_rate]
            status = (f"vx={vx:+.2f} vy={vy:+.2f} yaw={yaw:+.2f}rad/s "
                      f"depth={est.depth_median:.2f}m valid={est.depth_valid_frac:.0%}")
            display = cv2.copyMakeBorder(display, 28, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(display, status, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            if not window_ready:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(WINDOW, display.shape[1], display.shape[0])
                window_ready = True
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('s'):
                left_idx, right_idx = right_idx, left_idx
                left, right = right, left
                left_focal, right_focal = right_focal, left_focal
                left_rotation, right_rotation = right_rotation, left_rotation
                est = make_estimator(left_focal, tracking_res, filters, kernel_size)  # reset tracking
                idx, previous_time = 0, None
                print(f"swapped -> left={left_idx}, right={right_idx}, focal={left_focal:.1f}px")
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        for cap in (left, right):
            if cap is not None:
                cap.release()
        cv2.destroyAllWindows()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=int, nargs='?', default=0)
    parser.add_argument('right', type=int, nargs='?', default=1)
    parser.add_argument('--filters', nargs='*', choices=['clahe', 'gaussian', 'median'], default=None,
                        help='ordered filters; omit to inherit swarm defaults, or pass alone for no filters')
    parser.add_argument('--kernel-size', type=int, default=None, help='positive odd blur kernel size')
    parser.add_argument('--no-rotate', action='store_true', help='disable the 90-degree mounting corrections')
    args = parser.parse_args(argv)
    if args.left == args.right:
        parser.error('left and right camera indices must differ')
    if args.kernel_size is not None and (args.kernel_size < 1 or args.kernel_size % 2 == 0):
        parser.error('--kernel-size must be a positive odd integer')
    return args


if __name__ == '__main__':
    args = parse_args()
    raise SystemExit(main(args.left, args.right, filters=args.filters,
                          kernel_size=args.kernel_size, rotate=not args.no_rotate))
