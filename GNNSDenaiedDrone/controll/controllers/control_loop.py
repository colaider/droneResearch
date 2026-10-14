import threading
import time
import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor

from communication.EstimatePublisher import EstimatePublisher
from communication.Rb3ReadSend import ReaderSender
from sensors.sensors import Sensors
from controllers.controller import Controller
from controllers.state_machine import StateMachine

WARMUP_STEPS = 200      # ~2 s: let the attitude estimate settle BEFORE measuring bias
CALIB_SAMPLES = 400     # ~4 s: longer resting average -> cleaner gyro bias (root of the drift)


class ControlLoop:
    def __init__(self, dt=0.01, ns="/drone", altitude=1.0, send=True,
                 show_camera=False):
        self.dt = dt
        self.show_camera = show_camera
        self._camera_window_open = show_camera
        self._last_camera_sequence = None
        self._camera_window_name = "Drone stereo camera"
        rclpy.init()
        self.node = rclpy.create_node("control_loop")
        self.reader = ReaderSender(node=self.node, ns=ns)
        self.sensors = Sensors(node=self.node, async_visual_processing=True)
        self.estimates = EstimatePublisher(self.node)     # /vo/status, for drone_gui
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
        self.estimates.publish(self.sensors, self.sm.stage)


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
                self._update_camera_view()

                if i % 2 == 0:
                    cmd = self.sm.last_cmd
                    visual_estimator = self.sensors.visual_estimator
                    stereo_depth = visual_estimator.depth_median
                    stereo_depth_text = (
                        f"{stereo_depth:.2f}m"
                        if np.isfinite(stereo_depth)
                        else "unavailable"
                    )
                    print(f"[{i}] {self.sm.stage:9s} dt={dt*1e3:4.1f}ms "
                          f"att={np.round(self.sensors.get_attitude(),3)} "
                          f"baro={self.sensors.get_baro_position():+.2f}m "
                          f"cmd={None if cmd is None else np.round(cmd,3)}"
                          f"pos={self.sensors.get_acc_position()}", 
                          f"cmd pos = {self.controller.posCMD}"
                          f"acc={self.sensors.get_acceleration()} "
                          f"visual_vel={np.round(self.sensors.get_visual_velocity(), 3)} "
                          f"stereo_depth={stereo_depth_text} "
                          f"depth_points={len(visual_estimator.depth_points)} "
                          f"depth_valid={visual_estimator.depth_valid_frac:.0%}",
                          flush=True)
                i += 1

                next_t += self.dt
                time.sleep(max(0.0, next_t - time.monotonic()))
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def _update_camera_view(self):
        if not self._camera_window_open:
            return

        frame = self.sensors.latest_camera_frame
        if frame is not None and frame.sequence_id != self._last_camera_sequence:
            preview = cv2.hconcat([frame.left, frame.right])
            cv2.imshow(self._camera_window_name, preview)
            self._last_camera_sequence = frame.sequence_id

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(self._camera_window_name)
            self._camera_window_open = False


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
        self.sensors.close()
        self.estimates.close()
        if self.show_camera:
            cv2.destroyAllWindows()
        if rclpy.ok():
            self.node.destroy_node()
            rclpy.shutdown()
