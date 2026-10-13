import threading
import time
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor

from communication.Rb3ReadSend import ReaderSender
from sensors.sensors import Sensors
from controllers.controller import Controller
from controllers.state_machine import StateMachine

WARMUP_STEPS = 200      # ~2 s: let the attitude estimate settle BEFORE measuring bias
CALIB_SAMPLES = 200     # ~2 s: average the resting accel+gyro bias


class ControlLoop:
    def __init__(self, dt=0.01, ns="/drone", altitude=1.0, send=True):
        self.dt = dt
        rclpy.init()
        self.node = rclpy.create_node("control_loop")
        self.reader = ReaderSender(node=self.node, ns=ns)
        self.sensors = Sensors()
        self.controller = Controller(dt=dt)
        self.sm = StateMachine(self.sensors, self.controller, self.reader,
                               altitude=altitude, warmup_steps=WARMUP_STEPS,
                               calib_samples=CALIB_SAMPLES, send=send)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin.start()


    def step(self, dt):
        self.reader.read()                                  # latch only; executor spins in background
        self.sensors.update(self.reader.get_accel(), self.reader.get_gyro(), self.reader.get_baro(), dt)
        self.sm.step()                                      # WARMUP -> CALIBRATE -> HOVER


    def run(self):
        next_t = time.monotonic()
        last = None
        i = 0
        try:
            while rclpy.ok():
                now = time.monotonic()
                dt = self.dt if last is None else max(1e-4, now - last)
                last = now

                self.step(dt)            # read -> sensors.update -> state machine

                if i % 50 == 0:
                    cmd = self.sm.last_cmd
                    print(f"[{i}] {self.sm.stage:9s} dt={dt*1e3:4.1f}ms "
                          f"att={np.round(self.sensors.get_attitude(),3)} "
                          f"baro={self.sensors.get_baro_position():+.2f}m "
                          f"cmd={None if cmd is None else np.round(cmd,3)}"
                          f"pos={self.sensors.get_acc_position():+.4f}", flush=True)
                i += 1

                next_t += self.dt
                time.sleep(max(0.0, next_t - time.monotonic()))
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        # Neutral stop: make the last thing on the wire zero tilt / zero throttle, so a Ctrl-C
        # does not leave the FC holding a non-zero command for the command_timeout window.
        try:
            self.reader.send([0.0, 0.0, 0.0, 0.0])
            time.sleep(0.05)
        except Exception:
            pass
        self._executor.shutdown()
        self._spin.join(timeout=1.0)
        if rclpy.ok():
            self.node.destroy_node()
            rclpy.shutdown()

