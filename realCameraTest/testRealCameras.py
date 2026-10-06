"""Test the current swarm estimator with two USB cameras.

Run directly or with python -m realCameraTest.testRealCameras.
Keys: ESC/q quits; s swaps the dominant camera and resets tracking.
Use --help for filter options. No Genesis installation or IMU is required.
"""
import argparse
import sys
import time

import cv2

if __package__:
    from .visualAccEst import VisulaAcEst
    from .display import compose_camera_display
else:
    from visualAccEst import VisulaAcEst
    from display import compose_camera_display


# Existing USB calibration: focal lengths in pixels at this capture resolution.
LEFT_FOCAL = (1718.54 + 1720.12) / 2.0
RIGHT_FOCAL = (1175.14 + 1177.06) / 2.0
BASELINE = 0.125
RES = (1920, 1080)
WINDOW = "Stereo cameras"


def open_camera(index, res):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def prepare_frame(frame, rotation):
    # Keep focal calibration in the same pixel scale even if capture ignores RES.
    if (frame.shape[1], frame.shape[0]) != RES:
        frame = cv2.resize(frame, RES, interpolation=cv2.INTER_AREA)
    return cv2.rotate(frame, rotation) if rotation is not None else frame


def main(left_idx=0, right_idx=1, *, filters=None, kernel_size=None, rotate=True):
    if left_idx == right_idx:
        raise ValueError("Choose two different camera indices")
    if filters is not None and any(name not in ('clahe', 'gaussian', 'median') for name in filters):
        raise ValueError("Supported filters: clahe, gaussian, median")
    if kernel_size is not None and (kernel_size < 1 or kernel_size % 2 == 0):
        raise ValueError("kernel_size must be a positive odd integer")
    left = right = None
    left_focal, right_focal = LEFT_FOCAL, RIGHT_FOCAL
    left_rotation = cv2.ROTATE_90_COUNTERCLOCKWISE if rotate else None
    right_rotation = cv2.ROTATE_90_CLOCKWISE if rotate else None
    tracking_res = (RES[1], RES[0]) if rotate else RES

    def make_estimator():
        est = VisulaAcEst(focal_px=left_focal, baseline=BASELINE, res=tracking_res)
        # Omission inherits the current shared swarm filter settings.
        if filters is not None:
            est.image_filters = tuple(filters)
        if kernel_size is not None:
            est.filter_kernel_size = kernel_size
        return est

    try:
        left = open_camera(left_idx, RES)
        right = open_camera(right_idx, RES)
        if not left.isOpened() or not right.isOpened():
            print(f"ERROR: could not open cameras left={left_idx}, right={right_idx}")
            return 1
        est = make_estimator()
        print(f"left={left_idx} right={right_idx} | focal={left_focal:.1f}px | filters={est.image_filters}")
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
            result = est.processing([fl, fr], idx)
            idx += 1
            display = compose_camera_display(result.frames, result.gray_frames)
            vx, vy, yaw = est.camera_velocity
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
                est = make_estimator()  # No tracks/filter history from the previous dominant eye.
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
    parser.add_argument('--no-rotate', action='store_true', help='disable the existing opposite 90-degree mounting corrections')
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
