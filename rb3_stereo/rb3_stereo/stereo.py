"""Synchronized OV9282 stereo pair on the RB3 Gen 2 (CAM0A + CAM0B), no ROS dependency.

    from rb3_stereo.stereo import StereoCamera
    cam = StereoCamera(640, 400, fps=60).start()
    pair = cam.read_pair()            # pair.left, pair.right (uint8 HxW), pair.t_left/t_right
    cam.set_exposure(exposure_us=3000, gain=2.0)
    cam.stop()

Sync ("software genlock"): both sensors run from the same reference clock, so their frame periods are
identical and the offset only needs setting once. align() makes CAM0B's frames a few lines longer or
shorter for a counted number of frames until both end within ~30 us of each other; a background
thread repeats this every few seconds against the ~2 ppm difference between the two MCLKs.
"""
import collections
import threading
import time
from dataclasses import dataclass

import numpy as np

from .sensor import (EXPOSURE_REGS, VTS_REG, Capture, Pipelines, SensorA, SensorB, cam0b_i2c_bus,
                     check_setup)

MODES = {(640, 400): (30, 60, 120), (1280, 800): (30, 60)}   # 1 MIPI lane per camera


@dataclass
class StereoPair:
    left: np.ndarray        # uint8 HxW
    right: np.ndarray
    t_left: float           # CLOCK_MONOTONIC seconds, end of frame readout
    t_right: float
    index: int              # running pair counter


class StereoCamera:
    def __init__(self, width=640, height=400, fps=120, left="cam0a", flip=False, genlock=True,
                 genlock_every=5.0, genlock_tol_us=30, exposure_us=2000, gain=1.0, log=print):
        if fps not in MODES.get((width, height), ()):
            raise ValueError(f"unsupported mode {width}x{height}@{fps}; modes: "
                             + ", ".join(f"{w}x{h}@{f}" for (w, h), fs in MODES.items() for f in fs))
        if left not in ("cam0a", "cam0b"):
            raise ValueError("left must be 'cam0a' or 'cam0b'")
        self.w, self.h, self.fps, self.left_cam, self.flip = width, height, fps, left, flip
        self.genlock, self.genlock_every, self.genlock_tol = genlock, genlock_every, genlock_tol_us * 1e-6
        self.exposure_us, self.gain = exposure_us, gain
        self.log = log
        self.debug = False
        self.offset = None                               # last measured B-A offset (s)
        self._lock = threading.RLock()                   # CAM0B timing registers
        self._running = False
        self.cap_a = self.cap_b = None

    # ---------------------------------------------------------------- start / stop

    def start(self):
        check_setup()
        p = Pipelines()
        vid_a, vid_b = p.setup(self.w, self.h)
        self.a = SensorA(p.subdev, p.i2c_a)
        for name in ("horizontal_flip", "vertical_flip"):         # driver default is 1/1
            self.a.ctrl(name, int(not self.flip))
        self._timing_a(self.fps)
        self._exposure_a(self.exposure_us, self.gain)

        self.cap_a = Capture(vid_a, self.w, self.h)
        self.a.wait_streaming()
        self.b = SensorB(cam0b_i2c_bus())
        self.b.clone_from(self.a)
        self.cap_b = Capture(vid_b, self.w, self.h)
        time.sleep(0.2)                                  # CSIPHY/CSID/VFE are set up at STREAMON
        self.b.stream(True)
        self._last = [-1, -1]
        self._pairs = 0
        self.pair_dt = collections.deque(maxlen=240)     # CAM0B-CAM0A frame-end time of recent pairs
        self._running = True
        if self.read_pair(timeout=3.0) is None:
            self.stop()
            raise RuntimeError("no frames from both cameras (vision mezzanine DIP2-1 must be OFF)")
        if self.genlock:
            self.align()
            threading.Thread(target=self._keep_aligned, daemon=True).start()
        return self

    def stop(self):
        self._running = False
        try:
            if self.cap_b:
                self.b.stream(False)
        except OSError:
            pass
        for cap in (self.cap_b, self.cap_a):
            if cap:
                cap.close()
        self.cap_a = self.cap_b = None

    def restart(self, **changes):
        """Stop, apply changed settings (width, height, fps, left, flip, ...) and start again."""
        self.stop()
        for k, v in changes.items():
            setattr(self, {"width": "w", "height": "h", "left": "left_cam"}.get(k, k), v)
        if self.fps not in MODES.get((self.w, self.h), ()):
            raise ValueError(f"unsupported mode {self.w}x{self.h}@{self.fps}")
        return self.start()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---------------------------------------------------------------- frames

    def read_pair(self, timeout=1.0):
        """Next stereo pair (frames from the same frame period of both sensors) or None on timeout."""
        deadline = time.monotonic() + timeout
        caps = (self.cap_a, self.cap_b)
        while self._running:
            got = []
            for i, cap in enumerate(caps):
                f = cap.wait_newer(self._last[i], max(0.0, deadline - time.monotonic()))
                if f is None:
                    return None
                got.append(f)
            ts = [cap.stamp(idx) for cap, (idx, _) in zip(caps, got)]
            if None in ts:
                if time.monotonic() > deadline:
                    return None
                continue
            d = ts[1] - ts[0]
            if abs(d) > 0.5 / self.fps:                  # different frame periods: drop the older one
                self._last[0 if d > 0 else 1] = got[0 if d > 0 else 1][0]
                continue
            self._last = [got[0][0], got[1][0]]
            self._pairs += 1
            self.pair_dt.append(d)
            (fa, fb), (ta, tb) = (got[0][1], got[1][1]), ts
            if self.left_cam == "cam0a":
                return StereoPair(fa, fb, ta, tb, self._pairs)
            return StereoPair(fb, fa, tb, ta, self._pairs)
        return None

    def pair_offset(self):
        """Median CAM0B-minus-CAM0A frame-end time (s) of the recent pairs."""
        d = sorted(self.pair_dt)
        return d[len(d) // 2] if d else None

    @property
    def readout_s(self):
        """Time from the first to the last image line (frame end minus readout = exposure end)."""
        return self.h * self.line_s

    # ---------------------------------------------------------------- timing / exposure

    def _timing_a(self, fps):
        hb_min = self.a.ctrl_range("horizontal_blanking")[0]
        self.a.ctrl("horizontal_blanking", hb_min)
        self.pixel_rate = self.a.ctrl("pixel_rate")
        hts = self.w + hb_min
        vb = self.a.ctrl("vertical_blanking", self.pixel_rate // (hts * fps) - self.h)
        self.line_s = hts / self.pixel_rate
        self.vts = self.h + vb

    def _exposure_a(self, exposure_us, gain):
        lines = max(1, round(exposure_us * 1e-6 / self.line_s))
        lines = min(lines, self.vts - 12)
        g = int(min(255, max(16, round(gain * 16))))
        self.a.ctrl("exposure", lines)
        self.a.ctrl("analogue_gain", g)
        self.exposure_us, self.gain = lines * self.line_s * 1e6, g / 16

    def set_fps(self, fps):
        if fps not in MODES[(self.w, self.h)]:
            raise ValueError(f"{fps} fps not available at {self.w}x{self.h}")
        with self._lock:
            self._timing_a(fps)
            self.fps = fps
            self.b.copy_regs(self.a, (VTS_REG, VTS_REG + 1))
            if self.exposure_us * 1e-6 > (self.vts - 12) * self.line_s:
                self.set_exposure(self.exposure_us, self.gain)
        if self.genlock:
            self.align()

    def set_exposure(self, exposure_us=None, gain=None):
        """Exposure (us, limited by the frame time) and analogue gain (1.0 .. 15.9) on both sensors."""
        with self._lock:
            self._exposure_a(self.exposure_us if exposure_us is None else exposure_us,
                             self.gain if gain is None else gain)
            self.b.copy_regs(self.a, EXPOSURE_REGS)
        return self.exposure_us, self.gain

    # ---------------------------------------------------------------- genlock

    def frame_offset(self, secs=0.2, since=None):
        """Mean CAM0B-minus-CAM0A frame-end time (s) of the frames ending in [since, since + secs]."""
        since = time.monotonic() if since is None else since
        t_end = since + secs
        deadline = time.monotonic() + secs + 2.0
        ra, rb = self.cap_a.recent, self.cap_b.recent
        while time.monotonic() < deadline and not (ra and rb and ra[-1] > t_end and rb[-1] > t_end):
            time.sleep(0.005)
        ta = [t for t in list(ra) if since <= t <= t_end]
        tb = [t for t in list(rb) if since <= t <= t_end]
        if len(ta) < 3 or len(tb) < 3:
            return None
        period = (ta[-1] - ta[0]) / (len(ta) - 1)
        d = sorted((t - min(ta, key=lambda u: abs(t - u)) + period / 2) % period - period / 2 for t in tb)
        return d[len(d) // 2]                            # median: robust to interrupt-latency outliers

    def align(self, timeout=10.0, only_above=None):
        """Bring CAM0B's frames within genlock_tol of CAM0A's. Frames only get longer or shorter by a
        few lines, never cut short, so the capture pipelines are not disturbed.
        only_above: do nothing unless the offset exceeds this (s) - hysteresis for the background loop."""
        with self._lock:
            vts = self.b.vts()
            max_short = max(0, vts - self.h - 30)        # keep frame longer than readout + margin
            period = 1.0 / self.fps
            off = self.frame_offset()
            if off is not None and only_above is not None and abs(off) <= only_above:
                self.offset = off
                return off
            t_end = time.monotonic() + timeout
            try:
                for _ in range(25):
                    if off is None or abs(off) <= self.genlock_tol or time.monotonic() > t_end:
                        break
                    err = off / (period / vts)           # lines CAM0B is late (+) or early (-)
                    k = max(1, min(50, round(abs(err) / 15)))
                    if err > 0:
                        k = min(k, max_short) or 1
                    # a VTS change takes effect ~1 frame late and stays ~1 frame longer than written
                    frames = 1 if abs(err) <= 4 else max(1, round(abs(err) / k) - 1)
                    self.b.set_vts(vts - k if err > 0 else vts + k)
                    time.sleep(frames * period)
                    self.b.set_vts(vts)
                    new = self.frame_offset(since=time.monotonic() + 2 * period)
                    if self.debug:
                        self.log(f"align: {off * 1e6:+.0f} us -> {'-' if err > 0 else '+'}{k} lines x {frames} "
                                 f"frames -> {(new or 0) * 1e6:+.0f} us")
                    # the smallest nudge moves the offset by ~10-80 us (coarser at 120 fps): once it
                    # crosses zero the resolution limit is reached - stop instead of dithering around 0
                    crossed = new is not None and abs(err) <= 4 and (new > 0) != (off > 0)
                    off = new
                    if crossed:
                        break
            finally:
                self.b.set_vts(vts)
            self.offset = off
        return off

    def _keep_aligned(self):
        while self._running:
            time.sleep(self.genlock_every)
            if not self._running:
                return
            try:
                self.align(only_above=2 * self.genlock_tol)
            except Exception as e:                       # pipeline restarting etc.
                self.log(f"genlock: {e}")


class AutoExposure:
    """Software auto exposure (the raw sensor path has no ISP): steer the mean brightness of both
    images to `target` (0-255), exposure first up to max_exposure_us (motion blur), then gain."""

    def __init__(self, target=100, max_exposure_us=4000, max_gain=8.0, min_exposure_us=30, rate_hz=10):
        self.target, self.max_exposure_us, self.max_gain = target, max_exposure_us, max_gain
        self.min_exposure_us, self.period = min_exposure_us, 1.0 / rate_hz
        self.mean = None
        self._t = 0.0

    def update(self, cam, left, right):
        now = time.monotonic()
        if now - self._t < self.period:
            return False
        self._t = now
        self.mean = (float(left[::8, ::8].mean()) + float(right[::8, ::8].mean())) / 2
        ratio = self.target / max(self.mean, 1.0)
        if 0.93 < ratio < 1.07:
            return False
        total = cam.exposure_us * cam.gain * min(2.0, max(0.5, ratio)) ** 0.7
        exp = min(self.max_exposure_us, max(self.min_exposure_us, total))
        gain = min(self.max_gain, max(1.0, total / exp))
        if abs(exp - cam.exposure_us) / cam.exposure_us < 0.02 and abs(gain - cam.gain) < 0.07:
            return False
        cam.set_exposure(exp, gain)
        return True
