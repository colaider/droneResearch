#!/usr/bin/env python3
"""Calibrate the drone's stereo cameras with a checkerboard, with a live view over SSH X forwarding.

Board: 9 x 7 squares of 29 mm (8 x 6 inner corners); other boards: --squares / --square-mm.

On the drone, in a terminal with X forwarding (WSL or Linux: ssh -Y -C ubuntu@192.168.0.172):
    python3 ~/droneResearch/scripts/calibrate_rb3.py
Hold the board in front of both cameras and capture 20-30 pairs (SPACE, or A for automatic capture
whenever the board is still and in a new place): near and far, tilted left/right/up/down, and in every
part of the image - the green outlines show what is covered. C calibrates and switches to the rectified
view with guide lines: the board's corners must sit on the same line in both images. S saves the
calibration to rb3_stereo/config/stereo_calib.npz; from then on rb3_stereo rectifies every frame
(drone-stereo and drone-vo, which this script stops while it runs, start again with it at the end).

Keys: SPACE capture   A automatic capture on/off   U undo last   C calibrate   S save
      R back to recording   E auto exposure on/off   + / - exposure   Q quit

    --images DIR    calibrate pairs saved earlier (left_*.png / right_*.png), no camera or window
    --check         live rectified view with the saved calibration
Captured pairs are kept in ~/rb3_calibration/<resolution>-<time>/. One calibration serves 640x400 and
1280x800 (same view, 640x400 is binned); 1280x800 gives the more accurate corners.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("QT_X11_NO_MITSHM", "1")       # Qt over ssh X forwarding: no shared memory
import cv2  # noqa: E402
import numpy as np  # noqa: E402

_PKG = str(Path(__file__).resolve().parents[1] / "rb3_stereo")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)                          # rb3_stereo from this checkout, ROS not needed
from rb3_stereo import calibration as cal  # noqa: E402

SERVICES = ("drone-stereo", "drone-vo")              # the units that open the cameras
WIN = "rb3 stereo calibration"
BIG_CORES, LITTLE_CORES = {4, 5, 6, 7}, {0, 1, 2, 3}
GREEN, YELLOW, RED, GREY = (0, 220, 0), (0, 220, 255), (0, 0, 255), (160, 160, 160)


def pin(cores):
    try:
        os.sched_setaffinity(0, cores)
    except (AttributeError, OSError):
        pass


def find_pair(board, pool, left, right, fast):
    """Corners in both views, numbered alike, or None for a view without the board."""
    cl, cr = pool.map(lambda img: board.find(img, fast=fast), (left, right))
    if cl is not None:
        cl = cal.orient(cl)
        if cr is not None:
            cr = cal.orient(cr, like=cl)
    return cl, cr


def calibrate_pairs(board, size, left_points, right_points):
    calib = cal.calibrate(left_points, right_points, board, size)
    rect = cal.Rectifier(calib, size)
    rows = [rect.row_error(rect.rectify_points(left_points[i], "left"),
                           rect.rectify_points(right_points[i], "right")) for i in calib["used"]]
    calib["row_error"] = float(np.mean(rows))
    return calib, rect


def report(calib, rect, total):
    used = calib["used"]
    dropped = sorted(set(range(total)) - set(used))
    w, h = rect.size
    return "\n".join([
        f"calibrated from {len(used)} of {total} pairs" + (f" (dropped as outliers: {dropped})" if dropped else ""),
        f"  reprojection error: left {calib['rms_left']:.2f} px, right {calib['rms_right']:.2f} px, "
        f"stereo {calib['rms_stereo']:.2f} px   (good: below 0.5 px)",
        f"  baseline {np.linalg.norm(calib['T']) * 1000:.1f} mm   (check against the lens spacing)",
        f"  rectified: focal {rect.focal_px:.1f} px at {w}x{h}, view {rect.hfov_deg:.1f} x {rect.vfov_deg:.1f} deg, "
        f"row error {calib['row_error']:.2f} px   (good: below 0.5 px)"])


class Session:
    """Live recording / checking state for the camera mode."""

    def __init__(self, args, cam, ae, board, pool):
        self.args, self.cam, self.ae, self.board, self.pool = args, cam, ae, board, pool
        self.size = (cam.w, cam.h)
        self.save_dir = Path(args.save_dir).expanduser() if args.save_dir else \
            Path.home() / "rb3_calibration" / f"{cam.w}x{cam.h}-{time.strftime('%Y%m%d-%H%M%S')}"
        self.pairs, self.hulls = [], []                   # full-resolution corners, coverage outlines
        self.auto, self.ae_on = False, ae is not None
        self.calib, self.rect, self.saved = None, cam.rectifier, False
        self.mode = "check" if args.check else "record"
        self.prev, self.last_capture = None, 0.0
        self.msg, self.msg_t = "", 0.0
        self.view_w = min(args.view_width, 2 * cam.w)

    def say(self, text):
        self.msg, self.msg_t = text, time.monotonic()
        print(text, flush=True)

    # ---- capture

    def preview(self, left, right):
        """Fast corner search on a <= 640 px wide copy; corners in full-resolution pixels."""
        f = 1
        while left.shape[1] // f > 640:
            f *= 2
        if f > 1:
            left, right = (cv2.resize(i, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
                           for i in (left, right))
        cl, cr = find_pair(self.board, self.pool, left, right, fast=True)
        return (None if cl is None else (cl + 0.5) * f - 0.5), (None if cr is None else (cr + 0.5) * f - 0.5)

    def novelty(self, cl):
        """Mean corner distance (px) to the closest pose captured so far."""
        if not self.pairs:
            return np.inf
        return min(float(np.linalg.norm(cl - p[0], axis=1).mean()) for p in self.pairs)

    def capture(self, left, right, found, auto=False):
        if found[0] is None or found[1] is None:
            if not auto:
                self.say("the board must be seen by BOTH cameras")
            return
        if auto:
            still = self.prev is not None and float(np.linalg.norm(found[0] - self.prev, axis=1).mean()) \
                < 0.004 * self.size[0]
            if not still or time.monotonic() - self.last_capture < 1.0 \
                    or self.novelty(found[0]) < 0.08 * self.size[0]:
                return
        cl, cr = find_pair(self.board, self.pool, left, right, fast=False)
        if cl is None or cr is None:
            self.say("corners not found at full resolution - hold the board still")
            return
        if not self.pairs:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            (self.save_dir / "meta.json").write_text(json.dumps(dict(
                resolution=list(self.size), left_camera=self.cam.left_cam, flip=self.cam.flip,
                squares=list(self.board.squares), square_m=self.board.square_m), indent=1))
        n = len(self.pairs)
        cv2.imwrite(str(self.save_dir / f"left_{n:03d}.png"), left)
        cv2.imwrite(str(self.save_dir / f"right_{n:03d}.png"), right)
        self.pairs.append((cl, cr))
        self.hulls.append((cv2.convexHull(cl), cv2.convexHull(cr)))
        self.last_capture = time.monotonic()
        self.say(f"captured pair {n + 1}" + (" (auto)" if auto else ""))

    def undo(self):
        if not self.pairs:
            return
        n = len(self.pairs) - 1
        self.pairs.pop()
        self.hulls.pop()
        for side in ("left", "right"):
            (self.save_dir / f"{side}_{n:03d}.png").unlink(missing_ok=True)
        self.say(f"removed pair {n + 1}")

    def calibrate(self):
        if len(self.pairs) < self.args.min_pairs:
            self.say(f"{len(self.pairs)} pairs - capture at least {self.args.min_pairs} (20-30 is better)")
            return
        self.say("calibrating ...")
        self.calib, self.rect = calibrate_pairs(self.board, self.size, [p[0] for p in self.pairs],
                                                [p[1] for p in self.pairs])
        print(report(self.calib, self.rect, len(self.pairs)), flush=True)
        self.saved, self.mode = False, "check"
        self.say(f"calibrated: stereo error {self.calib['rms_stereo']:.2f} px, row error "
                 f"{self.calib['row_error']:.2f} px - S saves, R records more")

    def save(self):
        if self.calib is None:
            self.say("calibrate first (C)")
            return
        written = cal.save(self.args.out, self.calib, self.board, self.cam.left_cam, self.cam.flip)
        cal.save(self.save_dir / "stereo_calib.npz", self.calib, self.board, self.cam.left_cam, self.cam.flip)
        self.saved = True
        self.say(f"saved {written[0]} (+ left.yaml, right.yaml)")

    # ---- drawing

    def render(self, left, right, found):
        s = self.view_w / (2 * self.size[0])
        views = [cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
                 for img in (left, right)]
        for i, view in enumerate(views):
            if self.mode == "record":
                for hull in self.hulls:
                    cv2.polylines(view, [np.int32(hull[i] * s)], True, (0, 120, 0), 1)
            else:
                self._guides(view, s)
            if found[i] is not None:
                cv2.drawChessboardCorners(view, self.board.size, (found[i] * s).reshape(-1, 1, 2), True)
        img = np.hstack(views)
        img = cv2.copyMakeBorder(img, 44, 22, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        both = found[0] is not None and found[1] is not None
        if self.mode == "record":
            top = (f"RECORD  pairs {len(self.pairs)}   board L {'ok' if found[0] is not None else '--'} "
                   f"R {'ok' if found[1] is not None else '--'}   auto {'ON' if self.auto else 'off'}")
            keys = "SPACE capture  A auto  U undo  C calibrate  E auto exp  +/- exposure  Q quit"
            color = GREEN if both else YELLOW
        else:
            row = ""
            if both and self.rect is not None:
                row = f"   board row error {self.rect.row_error(found[0], found[1]):.2f} px"
            top = ("CHECK  rectified" + (f"   stereo error {self.calib['rms_stereo']:.2f} px" if self.calib else "")
                   + row + ("   SAVED" if self.saved else ""))
            keys = ("S save  R record more  Q quit" if not self.args.check else "Q quit")
            color = GREEN
        cv2.putText(img, top, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        exp = f"exposure {self.cam.exposure_us:.0f} us  gain {self.cam.gain:.1f}  AE {'on' if self.ae_on else 'off'}"
        msg = self.msg if time.monotonic() - self.msg_t < 3.0 else ""
        cv2.putText(img, f"{exp}   {msg}", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    RED if msg.startswith(("the board", "corners")) else GREY, 1, cv2.LINE_AA)
        cv2.putText(img, keys, (8, img.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1, cv2.LINE_AA)
        return img

    def _guides(self, view, s):
        """Lines along the rectified epipolar direction: matching points lie on the same line."""
        h, w = view.shape[:2]
        horizontal = self.rect is None or abs(self.rect.disparity_direction[0]) >= abs(self.rect.disparity_direction[1])
        step = 24
        for p in range(step // 2, h if horizontal else w, step):
            a, b = ((0, p), (w - 1, p)) if horizontal else ((p, 0), (p, h - 1))
            cv2.line(view, a, b, (90, 60, 160), 1)

    # ---- keys

    def key(self, k, pair, found):
        if k in (ord("q"), 27):
            return False
        if self.mode == "record":
            if k == ord(" "):
                self.capture(pair.left, pair.right, found)
            elif k == ord("a"):
                self.auto = not self.auto
                self.say(f"automatic capture {'on: hold the board still in new places' if self.auto else 'off'}")
            elif k == ord("u"):
                self.undo()
            elif k == ord("c"):
                self.calibrate()
        elif not self.args.check:
            if k == ord("s"):
                self.save()
            elif k == ord("r"):
                self.mode = "record"
                self.say("recording - C calibrates again with all pairs")
        if k == ord("e") and self.ae is not None:
            self.ae_on = not self.ae_on
            self.say(f"auto exposure {'on' if self.ae_on else 'off'}")
        elif k in (ord("+"), ord("="), ord("-")):
            self.ae_on = False
            self.cam.set_exposure(self.cam.exposure_us * (1.25 if k != ord("-") else 0.8))
        return True


def window_open():
    """A missing window's property is -1, or (Qt, all windows gone) an error."""
    try:
        return cv2.getWindowProperty(WIN, cv2.WND_PROP_AUTOSIZE) >= 0
    except cv2.error:
        return False


def stop_camera_services():
    active = [s for s in SERVICES if subprocess.run(["systemctl", "is-active", "--quiet", s]).returncode == 0]
    if not active:
        return []
    names = " and ".join(active)
    answer = input(f"{names} {'is' if len(active) == 1 else 'are'} using the cameras. Stop while calibrating "
                   f"(started again at the end)? [Y/n] ").strip().lower()
    if answer not in ("", "y", "yes"):
        sys.exit(f"the cameras are busy - stop it first: sudo systemctl stop {' '.join(active)}")
    subprocess.run(["sudo", "systemctl", "stop", *active], check=True)
    return active


def run_camera(args, board):
    from rb3_stereo.stereo import AutoExposure, StereoCamera
    if not os.environ.get("DISPLAY"):
        sys.exit("No X display: log in with X forwarding and run it again\n"
                 "  WSL / Linux:  ssh -Y -C ubuntu@192.168.0.172\n"
                 "  Windows:      MobaXterm (has an X server), or VcXsrv + PuTTY with X11 forwarding\n"
                 "(--images DIR calibrates pairs saved earlier without a window)")
    if args.check and not Path(args.out).exists():
        sys.exit(f"no calibration at {args.out} - calibrate first")
    stopped = stop_camera_services()
    w, h = args.resolution
    pin(LITTLE_CORES)                                # capture threads + v4l2-ctl inherit this
    cam = session = None
    try:
        cam = StereoCamera(w, h, 30, left=args.left, flip=args.flip, exposure_us=2000, gain=1.0,
                           calibration=args.out if args.check else None).start()
        pin(BIG_CORES)
        if args.check and cam.rectifier is None:
            sys.exit(f"{args.out} does not match the camera settings (left/flip)")
        ae = AutoExposure(target=args.ae_target, max_exposure_us=8000)
        with ThreadPoolExecutor(2) as pool:
            session = Session(args, cam, ae, board, pool)
            cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
            period, t_view, shown = 1.0 / args.view_fps, 0.0, False
            while True:
                pair = cam.read_pair(timeout=1.0)
                if pair is None:
                    print("no frames for 1 s")
                    continue
                if session.ae_on:
                    ae.update(cam, pair.left, pair.right)
                now = time.monotonic()
                if now - t_view < period:
                    continue
                t_view = now
                left, right = pair.left, pair.right
                if session.mode == "check" and not args.check:   # (--check: the camera rectifies)
                    left, right = session.rect.rectify(left, right)
                found = session.preview(left, right)
                if session.mode == "record" and session.auto:
                    session.capture(pair.left, pair.right, found, auto=True)
                session.prev = found[0]
                cv2.imshow(WIN, session.render(left, right, found))
                k = cv2.waitKey(1) & 0xFF
                if shown and not window_open():
                    break                                # closed with the window's X button
                shown = True
                if k != 255 and not session.key(k, pair, found):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if cam is not None:
            cam.stop()
        cv2.destroyAllWindows()
        if stopped:
            print(f"starting {' '.join(stopped)} again")
            subprocess.run(["sudo", "systemctl", "start", *stopped])
    if session is not None and session.pairs:
        print(f"{len(session.pairs)} pairs in {session.save_dir}")
        if session.calib is not None and not session.saved:
            print(f"the calibration was NOT saved; to redo it: {sys.argv[0]} --images {session.save_dir}")


def run_images(args, board):
    folder = Path(args.images).expanduser()
    meta_file = folder / "meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    if meta and (tuple(meta["squares"]) != board.squares or abs(meta["square_m"] - board.square_m) > 1e-9):
        print(f"using the board recorded in {meta_file}: {meta['squares']} squares of {meta['square_m'] * 1000:g} mm")
        board = cal.Board(meta["squares"], meta["square_m"])
    lefts, rights = sorted(folder.glob("left_*.png")), sorted(folder.glob("right_*.png"))
    if not lefts or len(lefts) != len(rights):
        sys.exit(f"{folder}: need matching left_*.png / right_*.png ({len(lefts)} / {len(rights)})")
    pin(BIG_CORES)
    left_points, right_points, size = [], [], None
    with ThreadPoolExecutor(2) as pool:
        for lf, rf in zip(lefts, rights):
            left, right = (cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in (lf, rf))
            size = (left.shape[1], left.shape[0])
            cl, cr = find_pair(board, pool, left, right, fast=False)
            if cl is None or cr is None:
                print(f"  {lf.name}: board not found in {'both views' if cl is None and cr is None else 'one view'}"
                      " - skipped")
                continue
            left_points.append(cl)
            right_points.append(cr)
    if len(left_points) < args.min_pairs:
        sys.exit(f"only {len(left_points)} usable pairs (need {args.min_pairs})")
    calib, rect = calibrate_pairs(board, size, left_points, right_points)
    print(report(calib, rect, len(left_points)))
    if args.dry_run:
        return
    left_cam, flip = meta.get("left_camera", args.left), meta.get("flip", args.flip)
    written = cal.save(args.out, calib, board, left_cam, flip)
    print(f"saved {', '.join(map(str, written))}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    size = lambda s: tuple(int(x) for x in s.lower().split("x"))
    p.add_argument("--squares", type=size, default=(9, 7), help="board squares, columns x rows (default 9x7)")
    p.add_argument("--square-mm", type=float, default=29.0, help="square size in mm (default 29)")
    p.add_argument("--resolution", type=size, default=(1280, 800), choices=[(1280, 800), (640, 400)],
                   help="1280x800 (default, more accurate) or 640x400")
    p.add_argument("--left", default="cam0a", choices=("cam0a", "cam0b"), help="left camera (as the node)")
    p.add_argument("--flip", action="store_true", help="images turned 180 deg (as the node's flip)")
    p.add_argument("--out", default=str(cal.default_path()), help="calibration file (default: %(default)s)")
    p.add_argument("--save-dir", help="where captured pairs go (default ~/rb3_calibration/<res>-<time>)")
    p.add_argument("--images", help="calibrate pairs saved earlier in this folder instead of using the camera")
    p.add_argument("--dry-run", action="store_true", help="with --images: report only, don't save")
    p.add_argument("--check", action="store_true", help="live rectified view with the saved calibration")
    p.add_argument("--min-pairs", type=int, default=12)
    p.add_argument("--ae-target", type=float, default=110.0, help="auto exposure: mean brightness 0-255")
    p.add_argument("--view-width", type=int, default=800, help="window width in px (X forwarding bandwidth)")
    p.add_argument("--view-fps", type=float, default=6.0, help="window refresh rate (X forwarding bandwidth)")
    return p.parse_args(argv)


def main(args):
    board = cal.Board(args.squares, args.square_mm / 1000)
    print(f"board {board.squares[0]}x{board.squares[1]} squares of {args.square_mm:g} mm "
          f"({board.size[0]}x{board.size[1]} inner corners)")
    if args.images:
        run_images(args, board)
    else:
        run_camera(args, board)


if __name__ == "__main__":
    main(parse_args())
