import threading
import time
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor

from communication.Rb3ReadSend import ReaderSender
from sensors.sensors import Sensors


class ControlLoop:
    def __init__(self, dt=0.01, ns="/drone"):
        self.dt = dt
        rclpy.init()
        self.node = rclpy.create_node("control_loop")
        self.reader = ReaderSender(node=self.node, ns=ns)   
        self.sensors = Sensors()                            

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin.start()


    def step(self, dt):
        self.reader.read()                                  
        self.sensors.update(self.reader.get_accel(), self.reader.get_gyro(), self.reader.get_baro(), dt)


    def run(self):
        next_t = time.monotonic()
        last = None
        i = 0
        try:
            while rclpy.ok():
                now = time.monotonic()
                dt = self.dt if last is None else max(1e-4, now - last)
                last = now
                self.step(dt)

                if i == 200: self.sensors.start_calibration(200)


                if i % 50 == 0:                             # ~2 Hz heartbeat: see the data update
                    # print(f"[{i}] dt={dt*1e3:4.1f}ms  att={np.round(self.sensors.get_attitude(),3)}"
                    #       f"  baro={self.sensors.get_baro_position():+.2f}m"
                    #       f"  baro_v={self.sensors.get_baro_velocity():+.2f}"
                    #       f"  acc_pos={np.round(self.sensors.get_acc_position(),2)}", flush=True)
                    pass
                i += 1

                next_t += self.dt
                time.sleep(max(0.0, next_t - time.monotonic()))
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        self._executor.shutdown()
        self._spin.join(timeout=1.0)
        if rclpy.ok():
            self.node.destroy_node()
            rclpy.shutdown()

