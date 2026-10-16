#!/usr/bin/env python3
"""axis_check: where do the IMU, the camera and the estimators think forward, right and up are?

Hold the drone in your hands (props OFF, disarmed) with the cameras 0.5-1.5 m above a textured
floor and follow the prompts: hold still, tilt the right side down, tilt the nose up, turn right,
slide forward, slide right, lift up. Every source is recorded at the same time, and the report
shows for each physical motion which of the source's OWN axes moved and with what sign. A
flipped or swapped axis shows up as the one cell that disagrees with the rest.

    python3 axis_check.py record              on the drone, ROS 2 sourced, ROS_DOMAIN_ID=42
    python3 axis_check.py analyze RUN.json    anywhere with numpy: the report for a recording
    python3 axis_check.py selftest            no hardware: checks this tool on simulated data

Read-only: it subscribes and never publishes. Sources, each optional:
    /drone/imu               gyro + accel as published (rb3/imu_topic.py), and the same remapped
                             the way the control loop reads it (the signs in Rb3ReadSend.py)
    /drone/attitude          the flight controller's roll/pitch/yaw (Betaflight)
    /stereo/left/image_raw   image motion measured here with OpenCV: shift, rotation and zoom of
                             the left image, independent of the team's estimator
    /vo/status               the control loop's camera velocity (vx, vy, vz, yaw_rate), its
                             world-frame camera velocity and the fused velocity (main.py running)
Attitude is also recomputed from the recorded IMU with the control loop's own IMU class
(sensors/IMU/IMU.py) and with drone_gui's filter (drone_gui/agent.py), so a recording can be
re-analysed after either of them changes.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CONTROLL = os.path.join(REPO, "GNNSDenaiedDrone", "controll")
READER = os.path.join(CONTROLL, "communication", "Rb3ReadSend.py")
IMU_CLASS = os.path.join(CONTROLL, "sensors", "IMU", "IMU.py")
CAMERA_CONFIG = os.path.join(CONTROLL, "sensors", "visual", "config", "cameraConfig.py")
GUI_AGENT = os.path.join(REPO, "drone_gui", "drone_gui", "agent.py")

LEAD = 1.0                  # s of stillness at the start of every step: the baseline
MOTION = 3.0                # s given for the motion
HOLD = 1.0                  # s of stillness at the end
STILL = 3.0                 # length of the first step, "hold still"
DEFAULT_FOCAL = 319.32      # px at 640 wide (cameraConfig STEREO_CAM_FOCAL_PX)
DEFAULT_DEPTH = 1.0         # m, the camera height assumed when /vo/status has no depth
DEFAULT_SIGNS = {"gyro": [1.0, -1.0, -1.0], "accel": [1.0, -1.0, -1.0]}

# The drone's body axes as unit vectors in FRD (x forward, y right, z down). Rotations follow the
# right-hand rule: a right roll turns about +forward, nose up about +right, a right turn about +down.
F, R, D = np.eye(3)
STEPS = [
    # name, prompt, kind, physical direction of the rotation / movement (FRD)
    ("still", "Hold the drone still and level", "still", None),
    ("roll_right", "Tilt the RIGHT side DOWN 20-30 deg, then hold", "rot", F),
    ("pitch_up", "Tilt the NOSE UP 20-30 deg, then hold", "rot", R),
    ("yaw_right", "Turn the nose RIGHT (clockwise seen from above) 45-90 deg, then hold", "rot", D),
    ("forward", "Slide FORWARD (towards the nose) 30-50 cm, keep it level, then stop", "move", F),
    ("right", "Slide to the RIGHT 30-50 cm, keep it level, then stop", "move", R),
    ("up", "Lift straight UP 30-50 cm, keep it level, then stop", "move", -D),
]
EXPECT = {name: direction for name, _, _, direction in STEPS if direction is not None}
ROT_STEPS = [name for name, _, kind, _ in STEPS if kind == "rot"]
MOVE_STEPS = [name for name, _, kind, _ in STEPS if kind == "move"]

# Below these a source counts as not having responded at all.
FLOOR = {"rad": math.radians(4), "m": 0.04, "m/s": 0.05, "deg": 4.0}


# --- the team's code, loaded from this checkout ------------------------------------------------

def load_module(name, path):
    """Import a file by path (no package imports, so no ROS); None if it is missing or broken."""
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:                        # noqa: BLE001 - a broken file must not stop us
        print("note: could not load %s: %s" % (os.path.relpath(path, REPO), exc))
        return None
    return module


def reader_signs(path=READER):
    """The sign vectors Rb3ReadSend multiplies /drone/imu's gyro and accel by, read from its source."""
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None
    signs = {}
    for key, name in (("gyro", "_gyro_sign"), ("accel", "_accel_sign")):
        match = re.search(r"%s\s*=\s*np\.array\(\s*\[([^\]]*)\]" % name, text)
        if not match:
            return None
        signs[key] = [float(v) for v in match.group(1).split(",")]
    return signs


def camera_config():
    """(focal px, resolution, cam-from-body rows) as cameraConfig.py's STEREO_CAM assumes them.

    cameraConfig describes the rig in the simulator's body frame, taken here as x forward, y left,
    z up, with camera axes image-right, image-up, backwards. The rows returned are the OpenCV
    camera axes (x image right, y image down, z the optical axis) as FRD body vectors."""
    module = load_module("axis_check_camera_config", CAMERA_CONFIG)
    if module is None or not hasattr(module, "STEREO_CAM"):
        return DEFAULT_FOCAL, (640, 400), None
    cam = module.STEREO_CAM
    axes = np.asarray(cam.camera_axes(), float)
    flu_to_frd = np.diag([1.0, -1.0, -1.0])
    right, up, back = (flu_to_frd @ axes[:, i] for i in range(3))
    return float(cam.focal_px), tuple(cam.res), np.array([right, -up, -back])


def estimator_dt():
    """(the dt VisulaEst starts with, whether anything in the control loop calls set_dt)."""
    try:
        with open(os.path.join(CONTROLL, "sensors", "visual", "visualEst.py"), encoding="utf-8") as handle:
            match = re.search(r"self\.dt\s*=\s*([0-9.eE+-]+)", handle.read())
    except OSError:
        return None, None
    called = False
    for root, _, files in os.walk(CONTROLL):
        for name in files:
            if name.endswith(".py"):
                with open(os.path.join(root, name), encoding="utf-8", errors="replace") as handle:
                    called = called or bool(re.search(r"\.set_dt\(", handle.read()))
    return (float(match.group(1)) if match else None), called


def repo_commit():
    try:
        out = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# --- image motion, measured here and not by the team's estimator --------------------------------

def measure_motion(prev, curr, scale=1.0):
    """How `curr` moved relative to `prev`: shift of the image centre (du, dv in full-resolution
    px, u right, v down), rotation (rad, from +u towards +v) and log zoom, from tracked corners.
    None when the image has too little texture."""
    import cv2
    points = cv2.goodFeaturesToTrack(prev, maxCorners=300, qualityLevel=0.01, minDistance=7)
    if points is None or len(points) < 12:
        return None
    moved, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, points, None, winSize=(21, 21),
                                                maxLevel=3)
    if moved is None:
        return None
    ok = status.ravel() == 1
    if ok.sum() < 12:
        return None
    m, inliers = cv2.estimateAffinePartial2D(points[ok], moved[ok], method=cv2.RANSAC,
                                             ransacReprojThreshold=1.5)
    if m is None or inliers is None or int(inliers.sum()) < 10:
        return None
    h, w = prev.shape[:2]
    centre = np.array([w / 2.0, h / 2.0])
    du, dv = (m[:, :2] @ centre + m[:, 2] - centre) / scale
    return (float(du), float(dv), float(math.atan2(m[1, 0], m[0, 0])),
            float(math.log(math.hypot(m[0, 0], m[1, 0]))), int(inliers.sum()))


class ImageMotion:
    """Frame-to-frame motion of a grey image stream, on a downscaled copy."""

    def __init__(self, scale=0.5):
        self.scale = scale
        self.prev = None

    def step(self, gray):
        import cv2
        small = gray if self.scale == 1 else cv2.resize(
            gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        prev, self.prev = self.prev, small
        return None if prev is None else measure_motion(prev, small, self.scale)


# --- the recording, as arrays ------------------------------------------------------------------

def _array(rows, columns):
    if not rows:
        return np.empty((0, columns))
    return np.array([list(row)[:columns] for row in rows], dtype=float)   # None -> NaN


def _rotation(v):
    """Rotation matrix of a rotation vector (Rodrigues)."""
    angle = float(np.linalg.norm(v))
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], float)
    if angle < 1e-12:
        return np.eye(3) + k
    k /= angle
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Run:
    """One recording: rows of each stream as arrays, plus the step windows."""

    # /vo/status columns after the receive time
    VO_BODY, VO_YAW, VO_WORLD, VO_FUSED, VO_DEPTH, VO_FPS = slice(1, 4), 4, slice(5, 8), slice(8, 11), 11, 13

    def __init__(self, rec):
        self.meta = rec.get("meta") or {}
        self.steps = {step["name"]: step for step in rec.get("steps") or []}
        self.imu = _array(rec.get("imu"), 8)       # t, stamp, gyro xyz, accel xyz (published)
        self.fc = _array(rec.get("fc"), 4)         # t, roll, pitch, yaw [rad]
        self.vo = _array(rec.get("vo"), 14)        # t, body v, yaw_rate, world v, fused v, depth, tracked, fps
        self.image = _array(rec.get("image"), 6)   # t, du, dv, dtheta, dlog_scale, inliers
        signs = self.meta.get("reader_signs") or reader_signs() or DEFAULT_SIGNS
        self.signs = {k: np.asarray(v, float) for k, v in signs.items()}
        width = (self.meta.get("image_size") or [640, 400])[0]
        self.focal = float(self.meta.get("focal_px_640") or DEFAULT_FOCAL) * width / 640.0
        self._attitudes = None

    # --- windows ---
    def window(self, t, name, part):
        step = self.steps[name]
        lo, hi = {"base": (step["t0"], step["go"] - 0.1),
                  "motion": (step["go"] - 0.1, step["t1"]),
                  "end": (step["t1"] - 0.6, step["t1"]),
                  "all": (step["t0"], step["t1"])}[part]
        return (t >= lo) & (t <= hi)

    def has(self, name):
        return name in self.steps

    # --- IMU in either frame ---
    def imu_frame(self, frame):
        gyro, accel = self.imu[:, 2:5], self.imu[:, 5:8]
        if frame == "ctrl":
            gyro, accel = gyro * self.signs["gyro"], accel * self.signs["accel"]
        return gyro, accel

    def turn(self, frame, name):
        """Rotation over a step [rad], the gyro integrated with the baseline's bias removed."""
        gyro, _ = self.imu_frame(frame)
        t, stamp = self.imu[:, 0], self.imu[:, 1]
        base, idx = self.window(t, name, "base"), np.flatnonzero(self.window(t, name, "motion"))
        if len(idx) < 2:
            return None
        bias = np.nanmean(gyro[base], axis=0) if base.any() else np.zeros(3)
        dt = np.clip(np.diff(stamp[idx]), 0.0, 0.05)
        return np.nansum((gyro[idx[1:]] - bias) * dt[:, None], axis=0)

    def push(self, frame, name):
        """Peak velocity over a movement step [m/s] from the accelerometer: readings turned back
        into the starting attitude with the gyro, minus the starting reading (gravity), integrated
        once. Its sign is the direction the accelerometer says the drone went."""
        gyro, accel = self.imu_frame(frame)
        t, stamp = self.imu[:, 0], self.imu[:, 1]
        base, idx = self.window(t, name, "base"), np.flatnonzero(self.window(t, name, "motion"))
        if len(idx) < 2 or not base.any():
            return None
        bias, a0 = np.nanmean(gyro[base], axis=0), np.nanmean(accel[base], axis=0)
        turn, velocity, best = np.eye(3), np.zeros(3), np.zeros(3)
        for prev, k in zip(idx[:-1], idx[1:]):
            dt = min(max(stamp[k] - stamp[prev], 0.0), 0.05)
            turn = turn @ _rotation((gyro[k] - bias) * dt)
            if np.all(np.isfinite(accel[k])):
                velocity = velocity + (turn @ accel[k] - a0) * dt
            if np.linalg.norm(velocity) > np.linalg.norm(best):
                best = velocity.copy()
        return best

    def gravity(self, frame):
        """Mean accelerometer reading over the still step."""
        if not self.has("still"):
            return None
        _, accel = self.imu_frame(frame)
        mask = self.window(self.imu[:, 0], "still", "all")
        return np.nanmean(accel[mask], axis=0) if mask.any() else None

    # --- camera image ---
    def depth(self, name):
        if len(self.vo):
            values = self.vo[self.window(self.vo[:, 0], name, "all"), self.VO_DEPTH]
            values = values[np.isfinite(values) & (values > 0)]
            if len(values):
                return float(np.median(values)), True
        return float(self.meta.get("depth_m") or DEFAULT_DEPTH), False

    def camera(self, name):
        """Camera motion over a step in OpenCV camera axes (x image right, y image down, z the
        optical axis): rotation [rad] for rotation steps, translation [m] for movement steps.
        Returns (vector, share of frames that had enough texture)."""
        if not len(self.image):
            return None, 0.0
        rows = self.image[self.window(self.image[:, 0], name, "motion")]
        if not len(rows):
            return None, 0.0
        good = np.all(np.isfinite(rows[:, 1:5]), axis=1)
        coverage = float(good.mean())
        if good.sum() < 3:
            return None, coverage
        du, dv, dtheta, dzoom = rows[good, 1:5].sum(axis=0)
        f = self.focal
        if dict((n, k) for n, _, k, _ in STEPS)[name] == "rot":
            return np.array([dv / f, -du / f, -dtheta]), coverage
        z, _ = self.depth(name)
        return np.array([-du * z / f, -dv * z / f, dzoom * z]), coverage

    # --- /vo/status ---
    def vo_integral(self, columns, name):
        """A /vo/status velocity integrated over a step: displacement (or angle for yaw_rate).
        None if there were no messages, NaN components never had a value."""
        if not len(self.vo):
            return None
        rows = self.vo[self.window(self.vo[:, 0], name, "motion")]
        if len(rows) < 2:
            return None
        values = rows[:, columns]
        dt = np.clip(np.diff(rows[:, 0]), 0.0, 0.5)
        dt = dt[:, None] if values.ndim == 2 else dt
        total = np.nansum(values[1:] * dt, axis=0)
        seen = np.isfinite(values).any(axis=0)
        return np.where(seen, total, np.nan)

    def vo_move(self, columns, name):
        """A /vo/status velocity's displacement over a movement step, for the axis tables: an
        axis that never had a value counts as 0, except vz on the lift (no value = no answer)."""
        total = self.vo_integral(columns, name)
        if total is None or (name == "up" and not np.isfinite(total[2])):
            return None
        return np.nan_to_num(total)

    # --- attitude from three estimators ---
    def attitudes(self):
        """{name: (t, [roll, pitch, yaw] rad)} for the control loop's IMU class, drone_gui's
        filter and the flight controller."""
        if self._attitudes is not None:
            return self._attitudes
        out = {}
        t, stamp = self.imu[:, 0], self.imu[:, 1]
        gyro_pub, accel_pub = self.imu_frame("pub")
        gyro_ctrl, accel_ctrl = self.imu_frame("ctrl")

        module = load_module("axis_check_imu", IMU_CLASS) if len(t) else None
        if module is not None and hasattr(module, "IMU"):
            imu = module.IMU()
            start = 0
            if self.has("still"):
                still = np.flatnonzero(self.window(t, "still", "all"))
                if len(still):
                    start = int(still[0])
                    imu.start_calibration(len(still))
            else:
                imu.start_calibration(min(200, len(t)))
            rpy = np.full((len(t), 3), np.nan)
            accel = accel_ctrl[start].copy()
            for k in range(start, len(t)):
                if np.all(np.isfinite(accel_ctrl[k])):
                    accel = accel_ctrl[k]
                dt = min(max(stamp[k] - stamp[k - 1], 1e-4), 0.05) if k > start else 0.008
                imu.update(accel, gyro_ctrl[k], dt)
                if not imu.is_calibrating():
                    rpy[k] = imu.get_attitude()
            out["ctrl"] = (t, rpy)

        agent = load_module("axis_check_gui_agent", GUI_AGENT) if len(t) else None
        if agent is not None and hasattr(agent, "AttitudeFilter"):
            fil = agent.AttitudeFilter()
            rpy = np.full((len(t), 3), np.nan)
            for k in range(len(t)):
                accel = accel_pub[k] if np.all(np.isfinite(accel_pub[k])) else None
                fil.update(float(stamp[k]), list(gyro_pub[k]), None if accel is None else list(accel))
                if fil.q is not None:
                    rpy[k] = agent.euler_from_quaternion(fil.q)
            out["gui"] = (t, rpy)

        if len(self.fc):
            out["fc"] = (self.fc[:, 0], self.fc[:, 1:4])
        self._attitudes = out
        return out

    def attitude_change(self, source, name):
        """[d roll, d pitch, d yaw] in degrees over a step: end hold minus start baseline."""
        series = self.attitudes().get(source)
        if series is None:
            return None
        t, rpy = series
        base, end = self.window(t, name, "base"), self.window(t, name, "end")
        base &= np.all(np.isfinite(rpy), axis=1)
        end &= np.all(np.isfinite(rpy), axis=1)
        if not base.any() or not end.any():
            return None

        def mean(rows):
            return np.array([rows[:, 0].mean(), rows[:, 1].mean(),
                             math.atan2(np.sin(rows[:, 2]).mean(), np.cos(rows[:, 2]).mean())])
        change = mean(rpy[end]) - mean(rpy[base])
        change[2] = _wrap(change[2])
        return np.degrees(change)

    def rates(self):
        out = {}
        for key, rows in (("imu", self.imu), ("fc", self.fc), ("vo", self.vo), ("image", self.image)):
            if len(rows) > 1 and rows[-1, 0] > rows[0, 0]:
                out[key] = (len(rows) - 1) / (rows[-1, 0] - rows[0, 0])
        return out


# --- axes and frames ---------------------------------------------------------------------------

def axis_of(vector, floor):
    """(signed unit axis, share of the vector on it) of the strongest component; None when the
    vector is missing, not finite or below floor."""
    if vector is None:
        return None
    v = np.asarray(vector, float)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        v = np.where(np.isfinite(v), v, 0.0) if v.shape == (3,) else None
        if v is None:
            return None
    norm = float(np.linalg.norm(v))
    if norm <= floor:
        return None
    i = int(np.argmax(np.abs(v)))
    unit = np.zeros(3)
    unit[i] = math.copysign(1.0, v[i])
    return unit, abs(v[i]) / norm


def axis_name(unit, names="xyz"):
    i = int(np.argmax(np.abs(unit)))
    return ("+" if unit[i] > 0 else "-") + names[i]


def cell(vector, floor, scale=1.0, unit="", names="xyz"):
    """'+x 24' style: the strongest axis and its size, '?' when it is not clearly one axis."""
    if vector is None:
        return "no data"
    hit = axis_of(vector, floor)
    if hit is None:
        return "- (%s%s)" % (_number(float(np.linalg.norm(np.nan_to_num(vector))) * scale), unit)
    value = float(np.dot(np.nan_to_num(vector), hit[0])) * scale
    return "%s %s%s%s" % (axis_name(hit[0], names), _number(value), unit, "" if hit[1] >= 0.7 else " ?")


def _number(value):
    return "%.0f" % value if abs(value) >= 10 else "%.2f" % value if abs(value) < 1 else "%.1f" % value


def frame_from(responses, floor):
    """Source-from-body matrix: column j = where body axis j (FRD) shows up in the source, from
    {step: response}. A column stays NaN when its step is missing or did not respond."""
    frame = np.full((3, 3), np.nan)
    for name, vector in responses.items():
        hit = axis_of(vector, floor)
        if hit is None:
            continue
        expected = EXPECT[name]
        j = int(np.argmax(np.abs(expected)))
        frame[:, j] = hit[0] * np.sign(expected[j])
    return frame


PHYSICAL = (("forward", "back"), ("right", "left"), ("down", "up"))


def describe(frame, names="xyz"):
    """'x = forward, y = right, z = down' for a source-from-body matrix."""
    parts = []
    for i in range(3):
        row = frame[i]
        if not np.all(np.isfinite(row)) or np.count_nonzero(row) != 1:
            hits = [j for j in range(3) if np.isfinite(frame[i, j]) and frame[i, j] != 0]
            if len(hits) == 1:
                j = hits[0]
                parts.append("%s = %s" % (names[i], PHYSICAL[j][0 if frame[i, j] > 0 else 1]))
            else:
                parts.append("%s = ?" % names[i])
            continue
        j = int(np.flatnonzero(row)[0])
        parts.append("%s = %s" % (names[i], PHYSICAL[j][0 if row[j] > 0 else 1]))
    return ", ".join(parts)


def complete(frame):
    """Every body axis measured, and each landing on a different source axis."""
    return bool(np.all(np.isfinite(frame)) and np.all(np.abs(frame).sum(0) == 1)
                and np.all(np.abs(frame).sum(1) == 1))


def compare(frame, reference):
    """Per body axis: 'same', 'flipped', 'swapped' or None (not measured)."""
    out = []
    for j in range(3):
        a, b = frame[:, j], reference[:, j]
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            out.append(None)
        elif np.array_equal(a, b):
            out.append("same")
        elif np.array_equal(a, -b):
            out.append("flipped")
        else:
            out.append("swapped")
    return out


def frame_cells(frame, names="xyz"):
    """forward / right / up columns of the summary table."""
    out = []
    for j, sign in ((0, 1), (1, 1), (2, -1)):
        column = frame[:, j]
        out.append(axis_name(column * sign, names) if np.all(np.isfinite(column)) else "?")
    return out


# --- the analysis ------------------------------------------------------------------------------

def analyze(rec):
    """(report text, findings dict) for a recording."""
    run = Run(rec)
    lines, findings, problems = [], {}, []
    out = lines.append
    meta = run.meta
    rates = run.rates()
    out("AXIS CHECK  %s  (repo %s)" % (meta.get("created", "?"), meta.get("commit") or "?"))
    out("streams: " + " | ".join(
        "%s %s" % (label, ("%.0f Hz" % rates[key]) if key in rates else "none")
        for key, label in (("imu", "/drone/imu"), ("fc", "FC attitude"), ("image", "camera"),
                           ("vo", "/vo/status"))))
    if len(run.image):
        found = np.all(np.isfinite(run.image[:, 1:5]), axis=1).mean()
        out("camera frames with enough texture to measure: %.0f%%" % (100 * found))
    out("control loop reads /drone/imu with gyro x %s, accel x %s (Rb3ReadSend.py)"
        % (run.signs["gyro"].tolist(), run.signs["accel"].tolist()))
    missing = [name for name, _, _, _ in STEPS if name not in run.steps]
    if missing:
        out("steps not recorded: " + ", ".join(missing))
    out("")

    # --- per step ---
    att_sources = (("ctrl", "control loop IMU.py"), ("gui", "drone_gui filter"), ("fc", "FC (Betaflight)"))
    out("PER STEP - IMU and attitude. Each cell is the source's own axis that moved, and how much:")
    out("  rotations in deg (gyro integrated), movements as peak velocity in m/s (accel), 'still' as")
    out("  the gravity reading in m/s^2. '-' = no response, '?' = not clearly a single axis.")
    head = "  %-11s %-15s %-15s %-20s %-17s %-15s" % (
        "step", "imu published", "imu ctrl loop", *[label for _, label in att_sources])
    out(head)
    names_rpy = ("roll", "pitch", "yaw")
    for name, _, kind, _ in STEPS:
        if name not in run.steps:
            continue
        if kind == "still":
            cells = [cell(run.gravity(f), 1.0, 1, "") for f in ("pub", "ctrl")] + ["", "", ""]
        elif kind == "rot":
            cells = [cell(run.turn(f, name), FLOOR["rad"], 180 / math.pi) for f in ("pub", "ctrl")]
            for source, _ in att_sources:
                change = run.attitude_change(source, name)
                if change is None:
                    cells.append("no data")
                    continue
                hit = axis_of(change, FLOOR["deg"])
                cells.append("-" if hit is None else "%s %+.0f%s" % (
                    names_rpy[int(np.argmax(np.abs(hit[0])))],
                    change[int(np.argmax(np.abs(hit[0])))], "" if hit[1] >= 0.7 else " ?"))
        else:
            cells = [cell(run.push(f, name), FLOOR["m/s"]) for f in ("pub", "ctrl")] + ["", "", ""]
        out("  %-11s %-15s %-15s %-20s %-17s %-15s" % (name, *cells))
    out("")

    out("PER STEP - camera and estimator. Rotations in deg, movements in m (velocities integrated).")
    out("  camera image: x = image right, y = image down, z = optical axis (OpenCV), measured here.")
    out("  vo body = vx, vy, vz (and yaw_rate) on /vo/status; during tilts it should stay ~0.")
    out("  %-11s %-17s %-17s %-17s %-17s" % ("step", "camera image", "vo body", "vo world", "fused"))
    for name, _, kind, _ in STEPS:
        if name not in run.steps or kind == "still":
            continue
        vector, coverage = run.camera(name)
        if kind == "rot":
            cam = cell(vector, FLOOR["rad"], 180 / math.pi)
            body = run.vo_integral(Run.VO_BODY, name)
            if name == "yaw_right":
                yaw = run.vo_integral(Run.VO_YAW, name)
                vo_body = "no data" if yaw is None else "yaw_rate %+.0f" % math.degrees(np.nan_to_num(yaw))
            else:
                vo_body = "no data" if body is None else "moved %.2f m" % np.linalg.norm(np.nan_to_num(body))
            others = [run.vo_integral(cols, name) for cols in (Run.VO_WORLD, Run.VO_FUSED)]
            others = ["no data" if v is None else "moved %.2f m" % np.linalg.norm(np.nan_to_num(v))
                      for v in others]
        else:
            cam = cell(vector, FLOOR["m"])
            vo_body = cell(run.vo_move(Run.VO_BODY, name), FLOOR["m"])
            others = [cell(run.vo_move(cols, name), FLOOR["m"]) for cols in (Run.VO_WORLD, Run.VO_FUSED)]
            if not len(run.vo):
                vo_body, others = "no data", ["no data", "no data"]
            elif name == "up":
                vo_body, *others = [text if run.vo_move(cols, name) is not None else "no vz"
                                    for text, cols in zip([vo_body] + others,
                                                          (Run.VO_BODY, Run.VO_WORLD, Run.VO_FUSED))]
        if vector is not None and coverage < 0.6:
            cam += " (%d%% fr)" % (100 * coverage)
        out("  %-11s %-17s %-17s %-17s %-17s" % (name, cam, vo_body, *others))
    out("")

    # --- frames ---
    rot_pub = frame_from({n: run.turn("pub", n) for n in ROT_STEPS if run.has(n)}, FLOOR["rad"])
    rot_ctrl = frame_from({n: run.turn("ctrl", n) for n in ROT_STEPS if run.has(n)}, FLOOR["rad"])
    acc_pub = frame_from({n: run.push("pub", n) for n in MOVE_STEPS if run.has(n)}, FLOOR["m/s"])
    acc_ctrl = frame_from({n: run.push("ctrl", n) for n in MOVE_STEPS if run.has(n)}, FLOOR["m/s"])
    cam_rot = frame_from({n: run.camera(n)[0] for n in ROT_STEPS if run.has(n)}, FLOOR["rad"])
    cam_move = frame_from({n: run.camera(n)[0] for n in MOVE_STEPS if run.has(n)}, FLOOR["m"])
    vo_body = frame_from({n: run.vo_move(Run.VO_BODY, n) for n in MOVE_STEPS if run.has(n)}, FLOOR["m"])
    vo_world = frame_from({n: run.vo_move(Run.VO_WORLD, n) for n in MOVE_STEPS if run.has(n)}, FLOOR["m"])
    fused = frame_from({n: run.vo_move(Run.VO_FUSED, n) for n in MOVE_STEPS if run.has(n)}, FLOOR["m"])
    _, _, cam_config = camera_config()

    out("WHERE EACH SOURCE PUTS FORWARD / RIGHT / UP (its own axis; for gyros and camera rotation:")
    out("  forward = axis of a right roll, right = axis of nose-up, up = minus the axis of a right turn)")
    out("  %-30s %-8s %-8s %-8s %s" % ("source", "forward", "right", "up", "so its axes are"))
    table = [("imu gyro, as published", rot_pub, "xyz"), ("imu gyro, control loop", rot_ctrl, "xyz"),
             ("imu accel, as published", acc_pub, "xyz"), ("imu accel, control loop", acc_ctrl, "xyz"),
             ("camera image, rotating", cam_rot, "xyz"), ("camera image, moving", cam_move, "xyz"),
             ("vo body velocity", vo_body, "xyz"), ("vo world velocity", vo_world, "xyz"),
             ("fused velocity", fused, "xyz")]
    if cam_config is not None:
        table.append(("camera as cameraConfig.py says", cam_config, "xyz"))
    for label, frame, names in table:
        if np.all(np.isnan(frame)):
            continue
        out("  %-30s %-8s %-8s %-8s %s" % (label, *frame_cells(frame, names), describe(frame, names)))
    out("")

    # --- findings ---
    out("FINDINGS")

    def ok(text):
        out("  ok  " + text)

    def bad(text):
        out("  !!  " + text)
        problems.append(text)

    def note(text):
        out("  ..  " + text)

    reference = rot_ctrl
    findings["imu_ctrl_frame"] = describe(rot_ctrl) if complete(rot_ctrl) else None
    if complete(rot_ctrl):
        det = round(float(np.linalg.det(rot_ctrl)))
        findings["imu_ctrl_det"] = det
        if det > 0:
            ok("control-loop gyro is a right-handed frame: %s" % describe(rot_ctrl))
        else:
            bad("control-loop gyro is MIRRORED (left-handed): %s - one gyro sign in "
                "Rb3ReadSend/imu_topic is wrong" % describe(rot_ctrl))
        if complete(rot_pub) and round(float(np.linalg.det(rot_pub))) < 0:
            bad("/drone/imu gyro as published is mirrored: %s" % describe(rot_pub))
    else:
        quiet = [n for n in ROT_STEPS if run.has(n) and axis_of(run.turn("ctrl", n), FLOOR["rad"]) is None]
        note("the gyro did not register %s, so the reference frame is incomplete - redo %s"
             % (", ".join(quiet) or "every rotation", "them" if len(quiet) != 1 else "it"))

    gravity = run.gravity("ctrl")
    if gravity is not None and complete(rot_ctrl):
        hit = axis_of(gravity, 5.0)
        if hit is not None:
            up = -rot_ctrl[:, 2]
            findings["gravity"] = "up" if np.array_equal(hit[0], up) else \
                "down" if np.array_equal(hit[0], -up) else "tilted"
            where = {"up": "the UP axis (standard: an accelerometer reads +g upwards at rest)",
                     "down": "the DOWN axis (so it reads gravity itself, not the specific force)",
                     "tilted": "a horizontal axis - was it level?"}[findings["gravity"]]
            (ok if findings["gravity"] == "up" else note)(
                "at rest the control-loop accel reads %+.2f on %s, i.e. %s"
                % (float(np.dot(gravity, hit[0])), axis_name(hit[0]), where))

    if complete(acc_ctrl) and complete(rot_ctrl):
        verdict = compare(acc_ctrl, rot_ctrl)
        findings["accel_vs_gyro"] = ("same" if verdict == ["same"] * 3 else
                                     "negated" if verdict == ["flipped"] * 3 else "mixed")
        if findings["accel_vs_gyro"] == "same":
            ok("accelerometer and gyro agree on every axis (accel is the specific force)")
        elif findings["accel_vs_gyro"] == "negated":
            bad("accelerometer is NEGATED relative to the gyro on all three axes: pushing forward "
                "reads backwards. FKF.predict does a = R(att)*accel - (0,0,9.81); with this accel "
                "it cancels gravity at rest but integrates every acceleration with the wrong sign")
        else:
            bad("accelerometer and gyro disagree on some axes: %s"
                % ", ".join("%s %s" % (axis, v) for axis, v in zip(("forward", "right", "down"), verdict)))

    if complete(cam_rot) and complete(cam_move):
        agree = compare(cam_move, cam_rot) == ["same"] * 3
        findings["camera_rot_vs_move"] = agree
        (ok if agree else note)("camera: rotations and movements give %s mounting"
                                % ("the same" if agree else "DIFFERENT (redo the steps more cleanly)"))
    camera_frame = cam_move if complete(cam_move) else cam_rot if complete(cam_rot) else None
    if camera_frame is not None:
        findings["camera_mount"] = describe(camera_frame)
        if cam_config is not None:
            same = compare(camera_frame, cam_config) == ["same"] * 3
            findings["camera_vs_config"] = "same" if same else "differs"
            if same:
                ok("camera mounting matches cameraConfig.py: %s" % describe(camera_frame))
            else:
                bad("camera mounting is NOT what cameraConfig.py assumes: measured %s, config %s"
                    % (describe(camera_frame), describe(cam_config)))

    # estimator body velocity against the control-loop IMU frame
    vo_checks = {}
    if not np.all(np.isnan(vo_body)) and np.all(np.isfinite(reference)):
        for j, (direction, verdict) in enumerate(zip(("forward", "right", "up"),
                                                     compare(vo_body, reference))):
            if verdict is None:
                continue
            vo_checks[direction] = verdict
            sign = -1 if direction == "up" else 1
            says = axis_name(vo_body[:, j] * sign)
            imu = axis_name(reference[:, j] * sign)
            if verdict == "same":
                ok("vo body velocity, moving %s: %s, same as the control-loop IMU" % (direction, says))
            else:
                bad("vo body velocity, moving %s: %s but the control-loop IMU says %s (%s)"
                    % (direction, says, imu, verdict))
    findings["vo_vs_imu"] = vo_checks

    if run.has("yaw_right") and len(run.vo):
        yaw = run.vo_integral(Run.VO_YAW, "yaw_right")
        gyro = run.turn("ctrl", "yaw_right")
        if yaw is not None and np.isfinite(yaw) and gyro is not None and abs(gyro[2]) > FLOOR["rad"]:
            ratio = float(yaw / gyro[2])
            findings["vo_yaw"] = {"sign": "same" if ratio > 0 else "opposite", "ratio": abs(ratio)}
            text = ("vo yaw_rate over the right turn: %+.0f deg, control-loop gyro z: %+.0f deg"
                    % (math.degrees(yaw), math.degrees(gyro[2])))
            if abs(ratio) < 0.1:
                bad(text + " - yaw_rate barely moves")
            elif ratio < 0:
                bad(text + " - OPPOSITE sign%s" % ("" if 0.7 < abs(ratio) < 1.4 else ", and %.1fx the size" % abs(ratio)))
            elif not 0.7 < ratio < 1.4:
                bad(text + " - right sign but %.1fx the size" % ratio)
            else:
                ok(text)

    scales = []
    for name in ("forward", "right"):
        cam, _ = run.camera(name) if run.has(name) else (None, 0)
        body = run.vo_integral(Run.VO_BODY, name) if run.has(name) else None
        if cam is not None and body is not None and np.linalg.norm(cam[:2]) > FLOOR["m"]:
            scales.append(np.linalg.norm(np.nan_to_num(body[:2])) / np.linalg.norm(cam[:2]))
    if scales:
        scale = float(np.median(scales))
        findings["vo_scale"] = scale
        depth, measured = run.depth("forward" if run.has("forward") else "right")
        text = ("vo body distance is %.2fx what the camera image shows (depth %.2f m%s, focal %.0f px)"
                % (scale, depth, "" if measured else " assumed", run.focal))
        fps = run.vo[:, Run.VO_FPS] if len(run.vo) else np.empty(0)
        fps = fps[np.isfinite(fps) & (fps > 0)]
        fixed, called = estimator_dt()
        if 0.7 < scale < 1.4:
            ok(text)
        elif len(fps) and fixed and called is False:
            frame_dt = 1.0 / float(np.median(fps))
            bad(text + "; the estimator processes a frame every %.2f s but VisulaEst.dt stays %g s "
                "(nothing calls set_dt), which alone makes its velocities %.0fx too big"
                % (frame_dt, fixed, frame_dt / fixed))
        else:
            bad(text)

    leaks = []
    for name in ("roll_right", "pitch_up"):
        if not run.has(name):
            continue
        body = run.vo_integral(Run.VO_BODY, name)
        gyro = run.turn("ctrl", name)
        if body is None or gyro is None or np.linalg.norm(gyro) < FLOOR["rad"]:
            continue
        depth, _ = run.depth(name)
        leaks.append(np.linalg.norm(np.nan_to_num(body[:2])) / (depth * np.linalg.norm(gyro)))
    if leaks:
        leak = float(np.median(leaks)) / findings.get("vo_scale", 1.0)
        findings["vo_leak"] = leak
        text = ("tilting without moving shows up in vo body velocity at %.1fx depth x angle%s"
                % (leak, " (after dividing out the %.1fx scale)" % findings["vo_scale"]
                   if "vo_scale" in findings else ""))
        if leak < 0.4:
            ok(text + " - the gyro compensation works")
        elif leak < 1.5:
            bad(text + " - as if the gyro compensation were missing or far too weak")
        else:
            bad(text + " - about 2x means the gyro compensation has the WRONG SIGN")

    for key, label, frame in (("vo_world", "vo world velocity", vo_world), ("fused", "fused velocity", fused)):
        if np.all(np.isnan(frame)) or not np.all(np.isfinite(reference[:2, :2])):
            continue
        world = {}
        body_xy = reference[:2, :2]
        if np.all(np.isfinite(frame[:2, :2])) and abs(np.linalg.det(body_xy)) > 0.5:
            m = frame[:2, :2] @ np.linalg.inv(body_xy)
            det = float(np.linalg.det(m))
            if det > 0.5:
                angle = math.degrees(math.atan2(m[1, 0], m[0, 0]))
                world["map"] = ("rotation", angle)
                ok("%s: world x/y = the IMU's body x/y turned by %.0f deg (a proper rotation)"
                   % (label, angle))
            elif det < -0.5:
                world["map"] = ("mirrored", None)
                bad("%s: body -> world is MIRRORED: one horizontal axis or the yaw sign is flipped "
                    "between the drone's real motion and this world frame" % label)
            else:
                note("%s: forward and right land on the same world axis - redo the slides" % label)
        if np.all(np.isfinite(frame[:, 2])):
            up = axis_name(-frame[:, 2])
            world["up"] = up
            if key == "fused":       # FKF.update_baro corrects vz and pz with the baro: up is +
                (ok if up == "+z" else bad)("%s: lifting up shows as %s%s" % (
                    label, up, "" if up == "+z" else " - but FKF's barometer correction counts up as +z"))
            else:
                note("%s: lifting up shows as %s" % (label, up))
        findings[key] = world

    for source, label in att_sources:
        signs = {}
        for name in ROT_STEPS:
            # judged only where the gyro saw the rotation: no motion is not a sign error
            if not run.has(name) or axis_of(run.turn("ctrl", name), FLOOR["rad"]) is None:
                continue
            change = run.attitude_change(source, name)
            if change is None:
                continue
            hit = axis_of(change, FLOOR["deg"])
            if hit is None:
                signs[name] = "none"
                continue
            i = int(np.argmax(np.abs(hit[0])))
            signs[name] = "%s%s" % (names_rpy[i], "+" if change[i] > 0 else "-")
        if not signs:
            continue
        findings.setdefault("attitude", {})[source] = signs
        expected = {"roll_right": "roll+", "pitch_up": "pitch+", "yaw_right": "yaw+"}
        wrong = ["%s gives %s" % (name, got) for name, got in signs.items() if got != expected[name]]
        text = "%s attitude: %s" % (label, ", ".join("%s %s" % (n, s) for n, s in signs.items()))
        if source == "fc":
            note(text + " (Betaflight's own convention)")
        elif wrong:
            bad(text + " - aerospace convention is roll+ / pitch+ / yaw+ for these")
        else:
            ok(text + " (aerospace convention)")

    if not problems:
        out("  no problems found")
    findings["problems"] = problems
    return "\n".join(lines), findings


def quick_look(rec, name):
    """One line about a just-recorded step, so a bad motion can be redone at once."""
    run = Run(rec)
    kind = dict((n, k) for n, _, k, _ in STEPS)[name]
    parts = ["imu %d, camera %d, vo %d samples" % tuple(
        int(rows[run.window(rows[:, 0], name, "all")].shape[0]) if len(rows) else 0
        for rows in (run.imu, run.image, run.vo))]
    if kind == "still":
        parts.append("gravity on control-loop %s" % cell(run.gravity("ctrl"), 1.0))
    elif kind == "rot":
        parts.append("gyro %s" % cell(run.turn("ctrl", name), FLOOR["rad"], 180 / math.pi, " deg"))
        parts.append("camera %s" % cell(run.camera(name)[0], FLOOR["rad"], 180 / math.pi, " deg"))
    else:
        parts.append("accel %s" % cell(run.push("ctrl", name), FLOOR["m/s"], 1.0, " m/s"))
        parts.append("camera %s" % cell(run.camera(name)[0], FLOOR["m"], 1.0, " m"))
        parts.append("vo %s" % (cell(run.vo_move(Run.VO_BODY, name), FLOOR["m"], 1.0, " m")
                                  if len(run.vo) else "no data"))
    return " | ".join(parts)


# --- recording on the drone --------------------------------------------------------------------

def record(args):
    try:
        import rclpy
        from geometry_msgs.msg import Vector3Stamped
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import Image, Imu
        from std_msgs.msg import String
    except ImportError as exc:
        sys.exit("record needs ROS 2 (%s): source /opt/ros/jazzy/setup.bash and export "
                 "ROS_DOMAIN_ID=42 first" % exc)
    try:
        import cv2
    except ImportError:
        cv2 = None
        print("note: no OpenCV, so no camera image measurements")

    focal, _, _ = camera_config()
    rec = {"meta": {"created": datetime.datetime.now().isoformat(timespec="seconds"),
                    "commit": repo_commit(), "reader_signs": reader_signs(),
                    "focal_px_640": focal, "depth_m": args.depth,
                    "topics": {"imu": args.imu, "fc": args.fc, "vo": args.vo, "image": args.image},
                    "timing": {"lead": LEAD, "motion": MOTION, "hold": HOLD, "still": STILL}},
           "imu": [], "fc": [], "vo": [], "image": [], "steps": []}
    lock = threading.Lock()
    start = time.monotonic()

    def now():
        return round(time.monotonic() - start, 4)

    def add(key, row):
        with lock:
            rec[key].append(row)

    def on_imu(m):
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        accel = ([None] * 3 if m.linear_acceleration_covariance[0] < 0 else
                 [m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
        add("imu", [now(), stamp, m.angular_velocity.x, m.angular_velocity.y,
                    m.angular_velocity.z] + accel)

    def on_fc(m):
        add("fc", [now(), m.vector.x, m.vector.y, m.vector.z])

    def triple(value):
        return list(value)[:3] if isinstance(value, list) and len(value) >= 3 else [None] * 3

    def on_vo(m):
        try:
            status = json.loads(m.data)
        except ValueError:
            return
        if isinstance(status, dict):
            add("vo", [now(), status.get("vx"), status.get("vy"), status.get("vz"),
                       status.get("yaw_rate")] + triple(status.get("visual_velocity"))
                + triple(status.get("velocity")) + [status.get("depth_m"), status.get("tracked"),
                                                    status.get("fps")])

    latest = {"image": None, "t": 0.0, "seq": 0}
    ready = threading.Condition()
    running = [True]

    def on_image(m):
        channels = {"mono8": 1, "8uc1": 1, "bgr8": 3, "rgb8": 3}.get(m.encoding.lower())
        if channels is None:
            return
        try:
            data = np.frombuffer(m.data, np.uint8)
        except TypeError:
            data = np.asarray(m.data, np.uint8)
        image = data[:m.height * m.step].reshape(m.height, m.step)[:, :m.width * channels]
        if channels == 3:
            image = cv2.cvtColor(image.reshape(m.height, m.width, 3), cv2.COLOR_BGR2GRAY)
        with ready:
            latest.update(image=image.copy(), t=now(), seq=latest["seq"] + 1)
            rec["meta"]["image_size"] = [int(m.width), int(m.height)]
            ready.notify()

    def image_worker():
        # Frames are measured against the previous MEASURED frame, so one skipped while the
        # worker is busy only makes that step longer, never wrong.
        motion = ImageMotion(args.image_scale)
        seen = 0
        while True:
            with ready:
                ready.wait_for(lambda: latest["seq"] != seen or not running[0], timeout=0.5)
                if not running[0]:
                    return
                if latest["seq"] == seen:
                    continue
                seen, image, t = latest["seq"], latest["image"], latest["t"]
            result = motion.step(image)
            add("image", [t] + (list(result) if result else [None] * 5))

    rclpy.init()
    node = rclpy.create_node("axis_check")
    node.create_subscription(Imu, args.imu, on_imu,
                             QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT))
    node.create_subscription(Vector3Stamped, args.fc, on_fc, 10)
    node.create_subscription(String, args.vo, on_vo, 10)
    if cv2 is not None and args.image:
        node.create_subscription(Image, args.image, on_image, qos_profile_sensor_data)
        threading.Thread(target=image_worker, daemon=True, name="image").start()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True, name="ros").start()

    def stop():
        running[0] = False
        with ready:
            ready.notify_all()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()

    print("Listening for 3 s to see which streams are there ...")
    time.sleep(3.0)
    with lock:
        counts = {key: len(rec[key]) for key in ("imu", "fc", "vo", "image")}
    for key, label in (("imu", args.imu), ("fc", args.fc), ("image", args.image), ("vo", args.vo)):
        print("  %-26s %s" % (label, "%.0f Hz" % (counts[key] / 3.0) if counts[key] else "NOTHING"))
    if not counts["imu"]:
        stop()
        sys.exit("No /drone/imu: start_rb3.py (or the drone-rb3 unit) must be running.")
    if not counts["vo"]:
        print("  -> no /vo/status, so the estimator rows stay empty. To test the estimator too, run\n"
              "     main.py (keep the drone DISARMED, props off) and start this again.")
    print("\nHold the drone with the cameras 0.5-1.5 m above a textured floor, props OFF.\n"
          "Each step: press Enter, keep it still until GO, do the motion, hold until it says done.")

    try:
        for i, (name, prompt, kind, _) in enumerate(STEPS):
            while True:
                input("\n[%d/%d] %s.\n      Press Enter, keep it still until GO ... "
                      % (i + 1, len(STEPS), prompt))
                t0 = now()
                if kind == "still":
                    print("      recording, keep it still ...", flush=True)
                    time.sleep(STILL)
                else:
                    time.sleep(LEAD)
                    print("      GO", flush=True)
                    time.sleep(MOTION)
                    print("      hold it ...", flush=True)
                    time.sleep(HOLD)
                print("      done", flush=True)
                step = {"name": name, "t0": t0, "go": round(t0 + LEAD, 4), "t1": now()}
                with lock:
                    snapshot = {key: list(value) if isinstance(value, list) else value
                                for key, value in rec.items()}
                snapshot["steps"] = [step]
                print("      " + quick_look(snapshot, name))
                answer = input("      %sEnter for the next step, r + Enter to redo this one: "
                               % ("" if kind == "still" else "Bring it back to the start, then "))
                if answer.strip().lower() != "r":
                    rec["steps"].append(step)
                    break
    except (KeyboardInterrupt, EOFError):
        print("\nstopped early: saving what was recorded")
    finally:
        stop()

    path = args.out or "axis_check_%s.json" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rec, handle)
    print("\nsaved %s  (re-run the report any time: python3 axis_check.py analyze %s)\n" % (path, path))
    print(analyze(rec)[0])


# --- self-test on simulated data ---------------------------------------------------------------

def _minimum_jerk(x):
    x = min(max(x, 0.0), 1.0)
    return x ** 3 * (10 - 15 * x + 6 * x * x)


def _rotation_vector(m):
    angle = math.acos(min(1.0, max(-1.0, (np.trace(m) - 1) / 2)))
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    return axis / 2 if angle < 1e-9 else axis * angle / (2 * math.sin(angle))


def simulate(truth, seed=0):
    """A recording of the seven steps for a drone whose sensors are mounted as `truth` says.

    truth: imu_raw (raw-from-body), accel_sign (+1: the raw accel reads the specific force),
    cam (OpenCV cam-from-body rows), vo (estimator body velocity = gain * vo @ the true velocity
    in the control-loop frame), vo_gain, vo_leak (0 = perfect gyro compensation, 1 = none,
    2 = wrong sign), yaw_sign, world (world-from-vo body), fused (fused-from-world), fc_signs."""
    rng = np.random.default_rng(seed)
    height, g, f = 1.0, np.array([0.0, 0.0, 9.81]), DEFAULT_FOCAL
    published = np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, 1.0])   # rb3/imu_topic.py
    to_ctrl = np.diag(DEFAULT_SIGNS["gyro"]) @ published[0] @ truth["imu_raw"]
    amplitude = {"roll_right": math.radians(25), "pitch_up": math.radians(25),
                 "yaw_right": math.radians(60), "forward": 0.4, "right": 0.4, "up": 0.4}

    segments, steps, t = [], [], 0.5
    for name, _, kind, direction in STEPS:
        t0 = t
        if kind == "still":
            steps.append({"name": name, "t0": t0, "go": t0 + LEAD, "t1": t0 + STILL})
            t = t0 + STILL + 0.5
            continue
        go, t1 = t0 + LEAD, t0 + LEAD + MOTION + HOLD
        segments.append((go + 0.2, go + 2.2, kind, direction, amplitude[name]))
        segments.append((t1 + 0.3, t1 + 2.3, kind, direction, -amplitude[name]))
        steps.append({"name": name, "t0": t0, "go": go, "t1": t1})
        t = t1 + 3.0
    end = t

    def pose(time_s):
        rot, pos = np.zeros(3), np.zeros(3)
        for a, b, kind, direction, amp in segments:
            if time_s > a:
                value = amp * _minimum_jerk((time_s - a) / (b - a))
                if kind == "rot":
                    rot += direction * value
                else:
                    pos += direction * value
        return _rotation(rot), pos

    h = 1e-3
    imu, fc, image, vo = [], [], [], []
    bias = np.array([0.0177, 0.0073, -0.0126])
    for time_s in np.arange(0.0, end, 1 / 125.0):
        rot, pos = pose(time_s)
        rot2, _ = pose(time_s + h)
        omega = _rotation_vector(rot.T @ rot2) / h
        accel_world = (pose(time_s + h)[1] - 2 * pos + pose(time_s - h)[1]) / h ** 2
        specific = rot.T @ (accel_world - g)
        gyro = published[0] @ (truth["imu_raw"] @ omega + bias + rng.normal(0, 0.003, 3))
        accel = published[1] @ (truth["accel_sign"] * truth["imu_raw"] @ specific
                                + rng.normal(0, 0.05, 3))
        imu.append([time_s + 0.002, time_s, *gyro, *accel])
    for time_s in np.arange(0.0, end, 0.15):
        rot, _ = pose(time_s)
        roll = math.atan2(rot[2, 1], rot[2, 2])
        pitch = -math.asin(max(-1.0, min(1.0, rot[2, 0])))
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        fc.append([time_s + 0.01, *(np.array([roll, pitch, yaw]) * truth.get("fc_signs", 1))])
    frames = np.arange(0.0, end, 1 / 30.0)
    for a, b in zip(frames[:-1], frames[1:]):
        rot_a, pos_a = pose(a)
        rot_b, pos_b = pose(b)
        turn = truth["cam"] @ _rotation_vector(rot_a.T @ rot_b)
        move = truth["cam"] @ (rot_a.T @ (pos_b - pos_a))
        z = height - pos_a[2]
        du = -f * (move[0] / z + turn[1]) + rng.normal(0, 0.05)
        dv = -f * move[1] / z + f * turn[0] + rng.normal(0, 0.05)
        image.append([b + 0.01, du, dv, -turn[2], move[2] / z, 120])
    for time_s in np.arange(0.0, end, 0.1):
        rot, pos = pose(time_s)
        velocity = rot.T @ (pose(time_s + h)[1] - pose(time_s - h)[1]) / (2 * h)
        omega = _rotation_vector(rot.T @ pose(time_s + h)[0]) / h
        z = height - pos[2]
        apparent = z * np.cross(omega, D)
        estimate = truth["vo_gain"] * truth["vo"] @ (to_ctrl @ (velocity + truth["vo_leak"] * apparent))
        yaw_rate = truth["vo_gain"] * truth["yaw_sign"] * (to_ctrl @ omega)[2]
        world = truth["world"] @ estimate
        fused = truth.get("fused", np.eye(3)) @ world
        vo.append([time_s + 0.005, *estimate, yaw_rate, *world, *fused, z, 150, 10.0])
    return {"meta": {"created": "simulated", "commit": None, "reader_signs": DEFAULT_SIGNS,
                     "focal_px_640": f, "image_size": [640, 400]},
            "imu": imu, "fc": fc, "vo": vo, "image": image, "steps": steps}


def _check_image_motion():
    """measure_motion against a known warp of a random texture."""
    import cv2
    rng = np.random.default_rng(3)
    texture = cv2.GaussianBlur((rng.random((400, 640)) * 255).astype(np.uint8), (0, 0), 2.0)
    texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX)
    angle, zoom, shift = 2.0, 1.02, (5.0, -3.0)
    m = cv2.getRotationMatrix2D((320.0, 200.0), angle, zoom)
    m[:, 2] += shift
    moved = cv2.warpAffine(texture, m, (640, 400), borderMode=cv2.BORDER_REFLECT)
    du, dv, theta, log_zoom, _ = measure_motion(texture, moved)
    # getRotationMatrix2D turns counter-clockwise on screen: from +u towards -v, so negative here
    assert abs(du - shift[0]) < 0.3 and abs(dv - shift[1]) < 0.3, (du, dv)
    assert abs(theta + math.radians(angle)) < 0.002, theta
    assert abs(log_zoom - math.log(zoom)) < 0.002, log_zoom
    return "image motion: shift %.2f/%.2f px, rotation %.2f deg, zoom %.4f (expected 5/-3, -2, 1.02)" % (
        du, dv, math.degrees(theta), math.exp(log_zoom))


def selftest(args):
    _, _, config_cam = camera_config()
    config_cam = np.eye(3) if config_cam is None else config_cam
    # The raw IMU taken as FRD with an Android-style accel reproduces what /drone/imu really
    # reads at rest (accel z = -9.8 published, +9.8 in the control loop). The fused velocity's z
    # is up, as FKF's barometer correction makes it.
    good = {"imu_raw": np.eye(3), "accel_sign": 1.0, "cam": config_cam, "vo": np.eye(3),
            "vo_gain": 1.0, "vo_leak": 0.0, "yaw_sign": 1.0, "world": np.eye(3),
            "fused": np.diag([1.0, 1.0, -1.0])}
    # camera turned 90 deg about its optical axis (image right = body right), vo y flipped (and
    # so the world velocity mirrored), 10x scale, gyro compensation with the wrong sign,
    # yaw_rate negated
    broken = dict(good, cam=np.array([R, -F, D]), vo=np.diag([1.0, -1.0, 1.0]), vo_gain=10.0,
                  vo_leak=2.0, yaw_sign=-1.0)
    failures = []

    def expect(label, actual, wanted):
        if actual != wanted:
            failures.append("%s: got %r, expected %r" % (label, actual, wanted))

    try:
        print(_check_image_motion())
    except ImportError:
        print("image motion: skipped, no OpenCV")
    except AssertionError as exc:
        failures.append("image motion off: %s" % exc)

    for title, truth in (("consistent drone (raw IMU = FRD, accel as published today)", good),
                         ("broken drone (injected flips)", broken)):
        text, found = analyze(simulate(truth))
        if args.verbose:
            print("\n=== %s ===\n%s" % (title, text))
        expect(title + " imu frame", found.get("imu_ctrl_frame"), "x = forward, y = right, z = down")
        expect(title + " accel", found.get("accel_vs_gyro"), "negated")
        expect(title + " gravity", found.get("gravity"), "down")
        if truth is good:
            expect(title + " camera", found.get("camera_vs_config"), "same")
            expect(title + " vo", found.get("vo_vs_imu"), {"forward": "same", "right": "same", "up": "same"})
            expect(title + " yaw", (found.get("vo_yaw") or {}).get("sign"), "same")
            expect(title + " scale", round(found.get("vo_scale", 0), 1), 1.0)
            expect(title + " leak", found.get("vo_leak", 9) < 0.2, True)
            expect(title + " world", (found.get("vo_world") or {}).get("map", ("?",))[0], "rotation")
            expect(title + " attitude", (found.get("attitude") or {}).get("ctrl"),
                   {"roll_right": "roll+", "pitch_up": "pitch+", "yaw_right": "yaw+"})
            expect(title + " gui attitude", (found.get("attitude") or {}).get("gui"),
                   {"roll_right": "roll+", "pitch_up": "pitch+", "yaw_right": "yaw+"})
            expect(title + " problems", len(found["problems"]), 1)    # only the negated accel
        else:
            expect(title + " camera", found.get("camera_vs_config"), "differs")
            expect(title + " camera mount", found.get("camera_mount"), "x = right, y = back, z = down")
            expect(title + " vo", found.get("vo_vs_imu"), {"forward": "same", "right": "flipped", "up": "same"})
            expect(title + " yaw", (found.get("vo_yaw") or {}).get("sign"), "opposite")
            expect(title + " scale", round(found.get("vo_scale", 0)), 10)
            expect(title + " leak", round(found.get("vo_leak", 0)), 2)
            expect(title + " world", (found.get("vo_world") or {}).get("map", ("?",))[0], "mirrored")
        print("%-62s %s" % (title, "ok" if not any(f.startswith(title) for f in failures) else "FAILED"))

    if failures:
        print("\n".join(["", "SELFTEST FAILED:"] + failures))
        return 1
    print("selftest passed")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="\n".join(__doc__.splitlines()[2:]))
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="guided recording on the drone (needs ROS 2)")
    rec.add_argument("--out", help="output file (default axis_check_<date>_<time>.json)")
    rec.add_argument("--imu", default="/drone/imu")
    rec.add_argument("--fc", default="/drone/attitude")
    rec.add_argument("--vo", default="/vo/status")
    rec.add_argument("--image", default="/stereo/left/image_raw", help="'' to skip the camera")
    rec.add_argument("--image-scale", type=float, default=0.5, help="downscale before tracking")
    rec.add_argument("--depth", type=float, default=DEFAULT_DEPTH,
                     help="camera height above the floor [m] if /vo/status has no depth")
    ana = sub.add_parser("analyze", help="print the report for a recording")
    ana.add_argument("file")
    test = sub.add_parser("selftest", help="check this tool on simulated recordings")
    test.add_argument("-v", "--verbose", action="store_true", help="print the simulated reports")
    args = parser.parse_args(argv)

    if args.command == "record":
        record(args)
        return 0
    if args.command == "analyze":
        with open(args.file, encoding="utf-8") as handle:
            print(analyze(json.load(handle))[0])
        return 0
    return selftest(args)


if __name__ == "__main__":
    sys.exit(main())
