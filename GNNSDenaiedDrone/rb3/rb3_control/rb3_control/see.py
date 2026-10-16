"""Read the RB3 Gen 2's on-board sensors through Qualcomm's SEE (sensor core) stack.

The ICM-42688-P IMU, ICP-10111 barometer and AKM magnetometer on the vision mezzanine are
wired to the SSC sensor core, not to the applications processor, so they never appear as
Linux IIO devices and no device-tree change can make them appear - the buses are not the
apps processor's to begin with. The supported path is Qualcomm's SEE stack:

    sscrpcd.service  ->  ADSP sensor PD (msm/adsp/sensor_pd)  ->  physical sensors
    see_workhorse    ->  streams one sensor as JSON event blocks on stdout

Requirements on the board (all from the ubuntu-qcom-iot/qcom-ppa PPA):
    qcom-sensors-core qcom-sensors-services qcom-sensors-registry qcom-sensors-utils
    qcom-sensors-test-apps       <- provides see_workhorse and ssc_sensor_info
and passwordless sudo: see_workhorse needs root to reach the sensor PD (it silently
produces no samples as a normal user).

Confirm what the sensor core actually offers with `sudo ssc_sensor_info`; on the hackathon
board that is accel, gyro, pressure, mag, sensor_temperature plus QTI's fused algorithms
(rotation vector, gravity, tilt, ...), which are reachable the same way.

Units as delivered by SEE: accel m/s^2, gyro rad/s, pressure hPa, mag uT.
"""
from __future__ import annotations

import collections
from math import isfinite
import re
import shutil
import subprocess
import threading
import time

BINARY = "see_workhorse"

# SSC timestamps are QTimer ticks, not seconds.
TICK_HZ = 19_200_000.0

# Rates each sensor accepts, from `ssc_sensor_info` on the RB3 Gen 2. Asking for anything
# else is not an error - the stream just stays silent, which is far harder to diagnose, so
# requests are snapped down to a supported rate instead.
RATES = {
    "accel": (12.5, 25.0, 50.0, 100.0, 200.0, 500.0),
    "gyro": (12.5, 25.0, 50.0, 100.0, 200.0, 500.0),
    "pressure": (1.0, 5.0, 10.0, 15.0, 20.0, 25.0),
    "mag": (1.0, 10.0, 20.0, 50.0, 100.0),
}

_DATA = re.compile(r'"data"\s*:\s*\[([^\]]*)\]')
_TIMESTAMP = re.compile(r'"timestamp"\s*:\s*(\d+)')

# Seconds to wait before respawning a stream that died (duration elapsed, ADSP restart, ...).
RESPAWN_DELAY = 1.0


class SscClockMapper:
    """Map SSC capture seconds onto host monotonic time from paired SEE events.

    Host event times include delivery latency. A low-percentile offset estimate
    tracks the low-latency edge of those observations instead of their average.
    """

    def __init__(self, capacity=256, quantile=0.1):
        if (isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0
                or not isfinite(quantile) or not 0.0 <= quantile <= 1.0):
            raise ValueError("capacity must be positive and quantile must be in [0, 1]")
        self._offsets = collections.deque(maxlen=capacity)
        self.quantile = float(quantile)

    def observe(self, host_monotonic_s, ssc_s):
        if host_monotonic_s is None or ssc_s is None:
            return
        host_monotonic_s, ssc_s = float(host_monotonic_s), float(ssc_s)
        if not (isfinite(host_monotonic_s) and isfinite(ssc_s)):
            return
        self._offsets.append(host_monotonic_s - ssc_s)

    def to_monotonic(self, ssc_s):
        if not self._offsets:
            return None
        offsets = sorted(self._offsets)
        index = int(round((len(offsets) - 1) * self.quantile))
        return float(ssc_s) + offsets[index]


def available() -> bool:
    return shutil.which(BINARY) is not None


def clamp_rate(sensor: str, rate: float) -> float:
    """Largest supported rate not above `rate` (or the lowest supported one)."""
    allowed = RATES.get(sensor)
    if not allowed:
        return rate
    below = [r for r in allowed if r <= rate]
    return max(below) if below else min(allowed)


def altitude_from_pressure(hpa: float, reference_hpa: float) -> float:
    """Barometric altitude [m] above the level where the pressure is `reference_hpa`."""
    if hpa <= 0.0 or reference_hpa <= 0.0:
        return 0.0
    return 44330.0 * (1.0 - (hpa / reference_hpa) ** 0.1902632)


class SeeStream(threading.Thread):
    """One `see_workhorse` process, parsed into (host_time, ssc_time, values) samples.

    Samples are both queued (drain() - nothing is dropped between drains, up to queue_len)
    and kept as `latest` (peek() - for merging a slow sensor into a fast one). The process
    is respawned if it exits, so an ADSP restart or a finished -duration is not fatal.
    """

    def __init__(self, sensor: str, rate: float, queue_len: int = 1024, sudo: bool = True,
                 duration: int = 86400, logger=None):
        super().__init__(daemon=True, name="see-" + sensor)
        self.sensor = sensor
        self.rate = clamp_rate(sensor, rate)
        self.sudo = sudo
        self.duration = duration
        self.log = logger
        self.samples: collections.deque = collections.deque(maxlen=queue_len)
        self.lock = threading.Lock()
        self.latest = None
        self.count = 0
        self.restarts = 0
        self.running = False
        self.proc = None

    @property
    def command(self):
        cmd = [BINARY, "-sensor=%s" % self.sensor, "-sample_rate=%g" % self.rate,
               "-display_events=1", "-duration=%d" % self.duration]
        return (["sudo", "-n"] + cmd) if self.sudo else cmd

    def start(self):
        self.running = True
        super().start()

    def stop(self):
        self.running = False
        proc = self.proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()

    def run(self):
        while self.running:
            try:
                self.proc = subprocess.Popen(
                    self.command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    text=True, bufsize=1)
            except FileNotFoundError:
                self._warn("%s not found - install qcom-sensors-test-apps" % BINARY)
                return
            self._consume(self.proc)
            if not self.running:
                return
            self.restarts += 1
            self._warn("%s stream exited (%d samples), restarting" % (self.sensor, self.count))
            time.sleep(RESPAWN_DELAY)

    def _consume(self, proc):
        buf = []
        for line in proc.stdout:
            if not self.running:
                return
            buf.append(line)
            # One event block ends with the "Time Elapsed" trailer see_workhorse prints.
            if '"Time Elapsed"' in line:
                self._parse("".join(buf))
                buf = []
            elif len(buf) > 4000:          # malformed output: do not grow without bound
                buf = []
        proc.wait()

    def _warn(self, message: str):
        if self.log is not None:
            self.log.warn(message)
        else:
            print("[see] " + message)

    def _parse(self, block: str):
        host = time.monotonic()
        # A block holds one or more events; configuration/attribute events carry no "data".
        for event in block.split('"msg_id"')[1:]:
            data = _DATA.search(event)
            if not data:
                continue
            try:
                values = [float(v) for v in data.group(1).split(",") if v.strip()]
            except ValueError:
                continue
            if not values:
                continue
            stamp = _TIMESTAMP.search(event)
            sample = (host, int(stamp.group(1)) / TICK_HZ if stamp else None, values)
            with self.lock:
                self.samples.append(sample)
                self.latest = sample
                self.count += 1

    # --- consumer side ----------------------------------------------------
    def drain(self):
        """Every sample since the last drain, oldest first."""
        with self.lock:
            out = list(self.samples)
            self.samples.clear()
        return out

    def peek(self):
        with self.lock:
            return self.latest

    def alive(self, timeout: float = 1.0) -> bool:
        sample = self.peek()
        return sample is not None and (time.monotonic() - sample[0]) < timeout

    def info(self):
        return {"rate_hz": self.rate, "samples": self.count, "restarts": self.restarts,
                "alive": self.alive()}


class SensorSet:
    """The SEE streams the node wants, keyed by sensor name. A rate of 0 disables one."""

    def __init__(self, rates: dict, sudo: bool = True, logger=None):
        self.streams = {name: SeeStream(name, rate, sudo=sudo, logger=logger)
                        for name, rate in rates.items() if rate and rate > 0}
        self.log = logger

    def start(self):
        for stream in self.streams.values():
            stream.start()
        if self.log is not None:
            self.log.info("SEE streams: " + ", ".join(
                "%s @ %g Hz" % (name, s.rate) for name, s in sorted(self.streams.items())))

    def stop(self):
        for stream in self.streams.values():
            stream.stop()

    def get(self, name):
        return self.streams.get(name)

    def info(self):
        return {name: stream.info() for name, stream in self.streams.items()}


def main(argv=None):
    """Standalone check: stream every sensor for a few seconds and print what arrives."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="print RB3 sensor-core (SEE) samples")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--no-sudo", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if not available():
        raise SystemExit("%s not found. sudo apt install qcom-sensors-test-apps" % BINARY)

    sensors = SensorSet({"accel": 200.0, "gyro": 200.0, "pressure": 25.0, "mag": 50.0},
                        sudo=not args.no_sudo)
    sensors.start()
    reference = None
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            time.sleep(0.25)
            parts = []
            for name in ("accel", "gyro", "mag", "pressure"):
                sample = sensors.get(name).peek()
                if sample is None:
                    parts.append("%s --" % name)
                    continue
                values = sample[2]
                if name == "pressure":
                    reference = reference or values[0]
                    parts.append("pressure %8.2f hPa (%+5.2f m)"
                                 % (values[0], altitude_from_pressure(values[0], reference)))
                else:
                    parts.append("%s " % name + " ".join("%+8.3f" % v for v in values[:3]))
            print("\r" + " | ".join(parts), end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        for name, info in sorted(sensors.info().items()):
            print("%-9s %s" % (name, info))
        sensors.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
