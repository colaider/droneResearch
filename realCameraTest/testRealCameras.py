"""Run the visual velocity estimator on two real USB cameras.

Usage:
    python testRealCameras.py                 # left=camera 0, right=camera 1
    python testRealCameras.py 2 3             # left=camera 2, right=camera 3

Keys:  ESC / q = quit,  s = swap left/right indices.

Focal length, baseline and resolution come from camera_params.txt. The estimator is
left-dominant, so it is fed the LEFT camera's focal length.
"""
import sys
import time
import cv2
import numpy as np

from visualAccEst import VisulaAcEst


# ---- constants from camera_params.txt ----
LEFT_FOCAL = (1718.54 + 1720.12) / 2.0     # Camera 1 (Left - Long), mean(fx, fy) px
RIGHT_FOCAL = (1175.14 + 1177.06) / 2.0    # Camera 2 (Right - Short), mean(fx, fy) px
BASELINE = 0.125                           # m, "125mm apart in Y"
RES = (1920, 1080)                         # (width, height) the focal lengths were calibrated at


def open_camera(index, res):
    # CAP_DSHOW avoids slow MSMF startup on Windows; drop it on Linux/mac.
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def main(left_idx=0, right_idx=1):
    left = open_camera(left_idx, RES)
    right = open_camera(right_idx, RES)
    if not left.isOpened() or not right.isOpened():
        print(f"ERROR: could not open cameras (left={left_idx}, right={right_idx}). "
              f"left_ok={left.isOpened()} right_ok={right.isOpened()}")
        return

    w = int(left.get(cv2.CAP_PROP_FRAME_WIDTH)) or RES[0]
    h = int(left.get(cv2.CAP_PROP_FRAME_HEIGHT)) or RES[1]
    print(f"left={left_idx} right={right_idx} | capture {w}x{h} | LEFT_FOCAL={LEFT_FOCAL:.1f}px baseline={BASELINE}m")

    est = VisulaAcEst(focal_px=LEFT_FOCAL, baseline=BASELINE, res=(w, h), dt=1.0 / 30.0)

    idx = 0
    prev_t = time.time()
    while True:
        ok_l, fl = left.read()
        ok_r, fr = right.read()
        if not (ok_l and ok_r):
            print("WARNING: frame grab failed, retrying...")
            continue

        # Cameras may ignore the requested resolution and return different sizes. Force both
        # onto RES so sizes match and the LEFT_FOCAL (calibrated at RES) stays valid.
        if (fl.shape[1], fl.shape[0]) != RES:
            fl = cv2.resize(fl, RES, interpolation=cv2.INTER_AREA)
        if (fr.shape[1], fr.shape[0]) != RES:
            fr = cv2.resize(fr, RES, interpolation=cv2.INTER_AREA)

        # Cameras are physically mounted rotated (camera_params.txt): left 90 CCW, right 90 CW.
        # Rotate only the frames to match that mounting. (1920x1080 -> 1080x1920 for both.)
        fl = cv2.rotate(fl, cv2.ROTATE_90_COUNTERCLOCKWISE)
        fr = cv2.rotate(fr, cv2.ROTATE_90_CLOCKWISE)

        now = time.time()
        est.set_dt(max(now - prev_t, 1e-3))      # real inter-frame time
        prev_t = now

        result = est.processing([fl, fr], idx)
        idx += 1

        vx, vy, yaw = est.camera_velocity
        disp_l, disp_r = result.frames
        txt = f"vx={vx:+.2f} vy={vy:+.2f} yaw={yaw:+.2f} rad/s  depth={est.depth_median:.2f}m  valid={est.depth_valid_frac*100:.0f}%"
        cv2.putText(disp_l, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("left (dominant)", disp_l)
        cv2.imshow("right", disp_r)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        if key == ord('s'):
            left_idx, right_idx = right_idx, left_idx
            left, right = right, left
            print(f"swapped -> left={left_idx} right={right_idx}")

    left.release()
    right.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    li = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ri = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    main(li, ri)
