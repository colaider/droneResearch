"""State machine (GNNS style, blueprint: swarm StateMachine) -- thin sequencer, no math.

Stages:  WARMUP -> CALIBRATE -> (FIND_HOVER | HOVER)
It only decides WHAT runs this tick and the transitions; the math lives in Sensors (bias) and
Controller (commands), the I/O in ReaderSender. Drive it from the loop: sensors.update() then
sm.step().

FIND_HOVER is an opt-in bench routine: hold the craft level and ramp ONLY throttle slowly up
until the barometer shows it lifting, then report that throttle (~hover) and cut. Use the printed
value as hover_throttle / the throttle floor in the controller. Props ON, drone free to lift
(tether or let it leave the ground a few cm); the radio + MSP OVERRIDE gate still applies.
"""
import time

import numpy as np


class StateMachine:
    def __init__(self, sensors, controller, reader, altitude=1.0,
                 warmup_steps=200, calib_samples=200, send=False, find_hover=False,
                 hover_start=0.20, hover_ramp=0.03, hover_max=0.70,
                 lift_alt=0.08, lift_hold=15):
        self.sensors = sensors
        self.controller = controller
        self.reader = reader
        self.altitude = altitude
        self.warmup_steps = warmup_steps
        self.calib_samples = calib_samples
        self.send = send                    # False = compute only (safe); True = actually command

        # FIND_HOVER tuning
        self.find_hover = find_hover        # True = run the hover-throttle search instead of HOVER
        self.hover_start = hover_start      # throttle to start the ramp at [0..1]
        self.hover_ramp = hover_ramp        # ramp rate [throttle / second] (dt-independent, wall clock)
        self.hover_max = hover_max          # give up / safety cap [0..1]
        self.lift_alt = lift_alt            # baro climb that counts as "lifting" [m]
        self.lift_hold = lift_hold          # consecutive ticks above lift_alt (beats baro noise)
        self.hover_throttle = None          # result: throttle at which lift was detected

        self.stage = "WARMUP"
        self.setpoint = np.array([0.0, 0.0, altitude, 0.0])
        self.last_cmd = None
        self._i = 0

        # FIND_HOVER state
        self._hover_t0 = None
        self._ground_alt = 0.0
        self._throttle = 0.0
        self._lift_count = 0

    def step(self):
        if self.stage == "WARMUP":
            if self._i >= self.warmup_steps:            # attitude settled -> measure bias
                self.sensors.start_calibration(self.calib_samples)
                self.stage = "CALIBRATE"

        elif self.stage == "CALIBRATE":
            if not self.sensors.is_calibrating():       # bias locked, world frame set
                self.controller.reset()
                # if self.find_hover:
                #     self.stage = "FIND_HOVER"
                # else:
                #     self.setpoint = np.array([0.0, 0.0, self.altitude, 0.0])   # hover at origin
                #     self.stage = "HOVER"

        elif self.stage == "FIND_HOVER":
            self._step_find_hover()

        elif self.stage == "HOVER":
            self.last_cmd = self.controller.update(self.sensors, self.setpoint)
            if self.send:
                self.reader.send(self.last_cmd)         # radio + MSP OVERRIDE still gate the FC

        elif self.stage == "DONE":
            self.last_cmd = np.array([0.0, 0.0, 0.0, 0.0])   # throttle cut, hold neutral
            if self.send:
                self.reader.send(self.last_cmd)

        self._i += 1
        return self.stage

    def _step_find_hover(self):
        now = time.monotonic()
        if self._hover_t0 is None:                      # first tick: anchor the ground level
            self._hover_t0 = now
            self._ground_alt = self.sensors.get_baro_position()
            print(f"[FIND_HOVER] start: ground={self._ground_alt:+.2f} m, "
                  f"ramp {self.hover_ramp:.3f}/s from {self.hover_start:.2f} to {self.hover_max:.2f}",
                  flush=True)

        # level command, throttle ramped on wall clock (dt-independent)
        self._throttle = min(self.hover_max, self.hover_start + self.hover_ramp * (now - self._hover_t0))
        self.last_cmd = np.array([0.0, 0.0, 0.0, self._throttle])
        if self.send:
            self.reader.send(self.last_cmd)

        # lift detection: baro climb above ground, sustained for lift_hold ticks (vs noise)
        climb = self.sensors.get_baro_position() - self._ground_alt
        self._lift_count = self._lift_count + 1 if climb > self.lift_alt else 0

        if self._lift_count >= self.lift_hold:
            self.hover_throttle = self._throttle
            print(f"[FIND_HOVER] LIFT at throttle={self._throttle:.3f} (climb {climb:+.2f} m). "
                  f"Use hover_throttle ~ {self._throttle:.3f}. Cutting throttle.", flush=True)
            self._cut()
        elif self._throttle >= self.hover_max:
            print(f"[FIND_HOVER] reached max throttle {self.hover_max:.2f} with no lift "
                  f"(climb {climb:+.2f} m) -- aborting. Check props/arming or raise hover_max.",
                  flush=True)
            self._cut()

    def _cut(self):
        self.last_cmd = np.array([0.0, 0.0, 0.0, 0.0])
        if self.send:
            self.reader.send(self.last_cmd)
        self.stage = "DONE"
