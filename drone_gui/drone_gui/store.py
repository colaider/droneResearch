"""The panel's view of the drone, independent of how the data arrived.

Both transports (ssh_link and ros_link) push the same value names in here, so the server and
the page do not care which one is in use.
"""
from __future__ import annotations

import collections
import math
import threading
import time

STALE_AFTER = 2.0

# Value name -> the plot series it feeds, and how to get the number out of the value.
SERIES_SOURCES = {
    "altitude": ("altitude", lambda v: v),
    "baro_altitude": ("baro_altitude", lambda v: v),
    "battery": ("battery", lambda v: v[0]),
    "attitude_roll": ("attitude", lambda v: math.degrees(v[0])),
    "attitude_pitch": ("attitude", lambda v: math.degrees(v[1])),
}


class Series:
    """A fixed-span ring buffer of (t, value) for the plots."""

    def __init__(self, span):
        self.span = span
        self.points = collections.deque()

    def add(self, value, now=None):
        now = now if now is not None else time.monotonic()
        self.points.append((now, float(value)))
        cutoff = now - self.span
        while self.points and self.points[0][0] < cutoff:
            self.points.popleft()

    def xy(self, origin=None):
        """Seconds-ago on x (negative, newest at 0) and value on y."""
        if not self.points:
            return [], []
        origin = origin if origin is not None else time.monotonic()
        return ([round(t - origin, 2) for t, _ in self.points], [v for _, v in self.points])


class Frame:
    """The latest JPEG of one camera topic."""

    def __init__(self):
        self.jpeg = None
        self.stamp = 0.0
        self.count = 0
        self.arrivals = collections.deque(maxlen=20)


class Store:
    def __init__(self, history_seconds=120.0, on_event=None):
        self.lock = threading.Lock()
        self.on_event = on_event or (lambda *a, **k: None)
        self.values = {}                    # name -> (value, monotonic stamp)
        self.counts = collections.Counter()
        self.frames = {}                    # camera topic -> Frame
        self.watchers = collections.Counter()   # camera topic -> open streams showing it
        self.started = time.monotonic()
        self.series = {name: Series(history_seconds) for name in
                       ("altitude", "baro_altitude", "battery", "roll", "pitch")}
        self._last_armed = None
        self._last_frames_sent = None
        self.transport_note = ""

    # --- writing ----------------------------------------------------------
    def apply(self, values):
        """One tick of telemetry. Names whose value is None are simply not refreshed."""
        now = time.monotonic()
        with self.lock:
            for name, value in values.items():
                if value is None:
                    continue
                self.values[name] = (value, now)
                self.counts[name] += 1
            for series_name, (value_name, extract) in (
                    ("altitude", SERIES_SOURCES["altitude"]),
                    ("baro_altitude", SERIES_SOURCES["baro_altitude"]),
                    ("battery", SERIES_SOURCES["battery"]),
                    ("roll", SERIES_SOURCES["attitude_roll"]),
                    ("pitch", SERIES_SOURCES["attitude_pitch"])):
                value = values.get(value_name)
                if value is None:
                    continue
                try:
                    self.series[series_name].add(extract(value), now)
                except (TypeError, IndexError, ValueError):
                    continue

        # Events worth keeping: these are the things you want to know about after the fact.
        armed = values.get("armed")
        if armed is not None:
            if self._last_armed is not None and armed != self._last_armed:
                self.on_event("ARMED" if armed else "disarmed",
                              "warning" if armed else "info")
            self._last_armed = armed

        status = values.get("status")
        if isinstance(status, dict):
            frames = status.get("frames_sent")
            # The counter only climbs, so a drop means the node was restarted.
            if isinstance(frames, int):
                if self._last_frames_sent is not None and frames < self._last_frames_sent:
                    self.on_event("flight controller node restarted", "warning")
                self._last_frames_sent = frames

    def apply_frame(self, topic, jpeg):
        now = time.monotonic()
        with self.lock:
            frame = self.frames.get(topic)
            if frame is None:
                frame = self.frames[topic] = Frame()
            frame.jpeg, frame.stamp = jpeg, now
            frame.count += 1
            frame.arrivals.append(now)

    # --- which camera topics an open page is showing ----------------------
    def watch(self, topic):
        """One more stream shows `topic`. True if that changed the set of watched topics."""
        with self.lock:
            self.watchers[topic] += 1
            return self.watchers[topic] == 1

    def unwatch(self, topic):
        with self.lock:
            self.watchers[topic] -= 1
            if self.watchers[topic] > 0:
                return False
            del self.watchers[topic]
            return True

    def watched(self):
        with self.lock:
            return sorted(self.watchers)

    # --- reading ----------------------------------------------------------
    def get(self, name, max_age=STALE_AFTER):
        with self.lock:
            entry = self.values.get(name)
        if entry is None:
            return None
        value, stamp = entry
        if max_age is not None and (time.monotonic() - stamp) > max_age:
            return None
        return value

    def age(self, name):
        with self.lock:
            entry = self.values.get(name)
        return None if entry is None else time.monotonic() - entry[1]

    def rates(self):
        elapsed = max(time.monotonic() - self.started, 1e-3)
        with self.lock:
            return {name: count / elapsed for name, count in self.counts.items()}

    def latest_frame(self, topic, max_age=5.0):
        with self.lock:
            frame = self.frames.get(topic)
            if frame is None or frame.jpeg is None:
                return None
            if max_age is not None and (time.monotonic() - frame.stamp) > max_age:
                return None
            return frame.jpeg

    def frame_stats(self):
        """Per camera topic: seconds since the last frame, frames received, and the rate they
        arrive at here (after the agent's throttling, so at most its FRAME_HZ)."""
        now = time.monotonic()
        with self.lock:
            out = {}
            for topic, frame in self.frames.items():
                times = frame.arrivals
                fps = 0.0
                if len(times) > 1 and now - times[-1] < 2.0 and times[-1] > times[0]:
                    fps = (len(times) - 1) / (times[-1] - times[0])
                out[topic] = {"age": now - frame.stamp, "frames": frame.count, "fps": fps}
            return out

    def plot(self, *names):
        """Each series carries its own x: they run at very different rates (25 Hz baro,
        6.7 Hz attitude, 1.3 Hz battery), so a shared index axis would misalign them."""
        origin = time.monotonic()
        with self.lock:
            out = {}
            for name in names:
                xs, ys = self.series[name].xy(origin)
                out[name] = {"x": xs, "y": ys}
        return out
