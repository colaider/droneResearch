"""Position tracks: the drone's x/y/z (and attitude) over time, one file per flight run.

The Position card draws the path in a top-down and a 3D view, for the live run or any earlier
one. A run is what the control loop flies between a start and a stop, so a new track begins when
positions come back after a gap, when the control loop's stage goes back to WARMUP (a restart
of drone-gnss-control), or when the page asks for one ("new track").

Sampled here from the link's latest values rather than inside a transport, so it works the same
over SSH and DDS. The SSH agent re-sends the latest value on every tick, so repeats are dropped.

Files: <state_dir>/tracks/<YYYYmmdd-HHMMSS>.jsonl - a header line, then one sample per line:
    [t, x, y, z, roll, pitch, yaw, mode, cam_x, cam_y, cam_z, cam_vx, cam_vy, cam_vz]
t is seconds since the start of the run, x/y/z in metres, attitude in radians (null when the
flight controller is not publishing it), mode the control loop's stage. cam_* is the camera's
own estimate from /vo/status (visual_position, and visual_velocity in the world frame); an
axis is null while the camera has no measurement for it. Tracks recorded before the camera
fields existed simply have shorter rows.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time

SAMPLE_HZ = 10.0
FRESH_SECONDS = 1.0         # a position older than this is not the drone's position any more
GAP_SECONDS = 5.0           # no positions for this long ends the run
REPEAT_SECONDS = 1.0        # an unchanged sample is still recorded this often, so time advances
TRACK_ID = re.compile(r"^\d{8}-\d{6}(-\d+)?$")


def _optional(values, length=3):
    """Like _vector, but per axis: non-finite or missing entries become None."""
    out = []
    for i in range(length):
        try:
            v = float(values[i])
        except (TypeError, ValueError, IndexError):
            v = None
        out.append(round(v, 3) if v is not None and math.isfinite(v) else None)
    return out


def _vector(values, length=3):
    """The first `length` entries as floats, or None unless all are finite numbers."""
    if not isinstance(values, (list, tuple)) or len(values) < length:
        return None
    try:
        out = [float(v) for v in values[:length]]
    except (TypeError, ValueError):
        return None
    return out if all(math.isfinite(v) for v in out) else None


class TrackLog:
    def __init__(self, state_dir, ros, on_event=None):
        self.dir = os.path.join(state_dir, "tracks")
        os.makedirs(self.dir, exist_ok=True)
        self.ros = ros
        self.on_event = on_event or (lambda *a, **k: None)
        self.lock = threading.Lock()
        self.live = None                # {"id", "start", "source", "samples": [...]}
        self._last_sample_at = 0.0      # monotonic
        self._last_mode = None
        self._split = False
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="tracks")

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop.set()

    # --- recording --------------------------------------------------------
    def _current(self):
        """(position, source, mode, camera) from the freshest source, or None. camera is
        [x, y, z, vx, vy, vz] from the camera alone, None per axis where it has nothing."""
        odom = self.ros.get("odom", FRESH_SECONDS)
        vo = self.ros.get("vo_status", FRESH_SECONDS)
        vo = vo if isinstance(vo, dict) else {}
        mode = vo.get("mode")
        camera = _optional(vo.get("visual_position")) + _optional(vo.get("visual_velocity"))
        position = _vector(odom.get("position")) if isinstance(odom, dict) else None
        if position is not None:
            return position, "odom", mode, camera
        position = _vector(vo.get("position"))
        if position is not None:
            return position, "vo", mode, camera
        return None

    def _run(self):
        period = 1.0 / SAMPLE_HZ
        while not self._stop.wait(period):
            try:
                self.sample()
            except Exception as exc:                    # noqa: BLE001 - keep recording
                self.on_event("position track: %s: %s" % (type(exc).__name__, exc), "error")
                self._stop.wait(5.0)

    def sample(self, now=None):
        now = now if now is not None else time.monotonic()
        current = self._current()
        if current is None:
            return
        position, source, mode, camera = current
        attitude = _vector(self.ros.get("attitude", FRESH_SECONDS)) or [None, None, None]

        with self.lock:
            restarted = mode == "WARMUP" and self._last_mode not in (None, "WARMUP")
            if (self.live is None or self._split or restarted
                    or now - self._last_sample_at > GAP_SECONDS):
                self._begin(source)
            self._last_mode = mode
            live = self.live
            t = round(time.time() - live["start"], 2)
            row = [t] + [round(v, 3) for v in position] + \
                  [None if a is None else round(a, 4) for a in attitude] + [mode] + camera
            last = live["samples"][-1] if live["samples"] else None
            if last is not None and last[1:] == row[1:] and now - self._last_sample_at < REPEAT_SECONDS:
                return
            live["samples"].append(row)
            self._last_sample_at = now
            self._append(live["id"], row)

    def _begin(self, source):
        start = time.time()
        track_id = base = time.strftime("%Y%m%d-%H%M%S", time.localtime(start))
        n = 1
        while os.path.exists(os.path.join(self.dir, track_id + ".jsonl")):
            n += 1
            track_id = "%s-%d" % (base, n)
        self.live = {"id": track_id, "start": start, "source": source, "samples": []}
        self._split = False
        self._last_mode = None
        self._append(track_id, {"start": start, "source": source,
                                "fields": ["t", "x", "y", "z", "roll", "pitch", "yaw", "mode",
                                           "cam_x", "cam_y", "cam_z",
                                           "cam_vx", "cam_vy", "cam_vz"]})
        self.on_event("position track %s started (%s)" % (track_id, source))

    def _append(self, track_id, entry):
        try:
            with open(os.path.join(self.dir, track_id + ".jsonl"), "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        except OSError:
            pass                # a full or read-only disk must not take the panel down

    def split(self):
        """End the live track; the next position starts a new one."""
        with self.lock:
            self._split = True

    # --- reading ----------------------------------------------------------
    def status(self):
        with self.lock:
            if self.live is None:
                return {"live": None, "samples": 0}
            return {"live": self.live["id"], "samples": len(self.live["samples"]),
                    "split": self._split}

    def list(self):
        tracks = []
        for name in sorted(os.listdir(self.dir), reverse=True):
            track_id = name[:-6]
            if not name.endswith(".jsonl") or not TRACK_ID.match(track_id):
                continue
            header, count, last_t = self._summary(os.path.join(self.dir, name))
            if header is None:
                continue
            tracks.append({"id": track_id, "start": header.get("start"),
                           "source": header.get("source"), "samples": count,
                           "seconds": last_t})
        return {"live": self.status()["live"], "tracks": tracks}

    @staticmethod
    def _summary(path):
        header, count, last_t = None, 0, 0.0
        try:
            with open(path, encoding="utf-8") as handle:
                for i, line in enumerate(handle):
                    if i == 0:
                        header = json.loads(line)
                        continue
                    count += 1
                    last = line
                if count:
                    last_t = json.loads(last)[0]
        except (OSError, ValueError, IndexError):
            pass
        return header, count, last_t

    def get(self, track_id, start_index=0):
        """Samples from start_index on. The live track comes from memory, others from disk."""
        if not TRACK_ID.match(track_id or ""):
            return None
        with self.lock:
            if self.live is not None and self.live["id"] == track_id:
                samples = self.live["samples"]
                return {"id": track_id, "start": self.live["start"],
                        "source": self.live["source"], "live": not self._split,
                        "total": len(samples), "from": start_index,
                        "samples": samples[start_index:]}
        path = os.path.join(self.dir, track_id + ".jsonl")
        header, samples = None, []
        try:
            with open(path, encoding="utf-8") as handle:
                for i, line in enumerate(handle):
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    if i == 0:
                        header = entry
                    else:
                        samples.append(entry)
        except OSError:
            return None
        header = header or {}
        return {"id": track_id, "start": header.get("start"), "source": header.get("source"),
                "live": False, "total": len(samples), "from": start_index,
                "samples": samples[start_index:]}
