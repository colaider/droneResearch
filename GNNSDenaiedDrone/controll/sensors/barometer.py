"""Barometer: altitude (zeroed at calibration) + vertical velocity from altitude differences.

Pure, no ROS. Calibrated alongside the IMU: it averages the resting altitude over the calibration
window and subtracts it, so altitude reads 0 at the start. Vertical velocity is a smoothed
derivative of that zeroed altitude. Fed into the FKF (baro vz + pz), not the IMU.
"""


class Barometer:
    def __init__(self, vel_ema=0.8):
        self.vel_ema = vel_ema
        self._offset = 0.0          # resting altitude subtracted to zero the start
        self._alt = 0.0             # zeroed altitude [m]
        self._vel = 0.0             # vertical velocity [m/s]
        self._prev_alt = None
        self._calib = None

    # ---------------- update ----------------
    def update(self, raw_altitude, dt):
        raw = float(raw_altitude)
        if self._calib is not None:
            self._accumulate_calibration(raw)
            return                              # calibrating: hold altitude/velocity at 0
        self._alt = raw - self._offset
        if self._prev_alt is not None and dt > 0:
            raw_vel = (self._alt - self._prev_alt) / dt
            b = self.vel_ema
            self._vel = b * self._vel + (1 - b) * raw_vel
        self._prev_alt = self._alt

    # ---------------- calibration (called together with the IMU) ----------------
    def start_calibration(self, n=400):
        self._calib = {"sum": 0.0, "count": 0, "n": int(n)}

    def _accumulate_calibration(self, raw):
        c = self._calib
        c["sum"] += raw
        c["count"] += 1
        if c["count"] < c["n"]:
            return
        self._offset = c["sum"] / c["count"]    # resting altitude -> 0 at start
        self._alt = 0.0
        self._vel = 0.0
        self._prev_alt = 0.0
        self._calib = None

    def is_calibrating(self):
        return self._calib is not None

    # ---------------- getters ----------------
    def get_altitude(self):
        return self._alt            # zeroed altitude [m]

    def get_velocity(self):
        return self._vel            # vertical velocity [m/s]
