"""State machine (GNNS style, blueprint: swarm StateMachine) -- thin sequencer, no math.

Stages:  WARMUP -> CALIBRATE -> HOVER
It only decides WHAT runs this tick and the transitions; the math lives in Sensors (bias) and
Controller (commands), the I/O in ReaderSender. Drive it from the loop: sensors.update() then
sm.step().
"""
import numpy as np


class StateMachine:
    def __init__(self, sensors, controller, reader, altitude=1.0,
                 warmup_steps=200, calib_samples=200, send=False):
        self.sensors = sensors
        self.controller = controller
        self.reader = reader
        self.altitude = altitude
        self.warmup_steps = warmup_steps
        self.calib_samples = calib_samples
        self.send = send                    # False = compute only (safe); True = actually command

        self.stage = "WARMUP"
        self.setpoint = 0.3
        self.last_cmd = None
        self._i = 0

    def step(self):
        if self.stage == "WARMUP":
            if self._i >= self.warmup_steps:            # attitude settled -> measure bias
                self.sensors.start_calibration(self.calib_samples)
                self.stage = "CALIBRATE"

        elif self.stage == "CALIBRATE":
            if not self.sensors.is_calibrating():       # bias locked, world frame set
                self.setpoint = np.array([0.0, 0.0, self.altitude, 0.0])   # hover at origin
                self.controller.reset()
                self.stage = "HOVER"

        elif self.stage == "HOVER":
            self.last_cmd = self.controller.update(self.sensors, self.setpoint)
            if self.send:
                self.reader.send(self.last_cmd)         # radio + MSP OVERRIDE still gate the FC

        self._i += 1
        return self.stage
