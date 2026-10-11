"""A REAL drone struct backed by ROS 2 topics - no simulator, no camera.

It has the same method names the swarm controllers call on `self.drone`, but the numbers come
from the RB3's `/drone` topics instead of Genesis. This is all the POSITION controller
(`DronePositionCTRL`) needs - no visual controller, no FKF, no frames.

How the data flows:
  * We only SUBSCRIBE here: gyro + accel (`/drone/imu`), attitude (`/drone/attitude`),
    barometer (`/drone/baro_altitude`), mag (`/drone/mag`). The latest value of each is kept.
  * The struct itself does NOT integrate anything. `DroneCTRL.step()` is what turns the raw
    gyro/accel into attitude, velocity and position (it calls `imu.read()` and `get_vel()` and
    does the integration). So position/velocity are "calculated from the IMU" - just inside the
    controller, exactly as in the sim. The struct only feeds the sensors.

Interface used by DroneCTRL / DronePositionCTRL (and matched here):
    imu.read() -> reading with .lin_acc / .ang_vel / .mag, each exposing .numpy()
    get_pos()  -> [x, y, z]      (x,y unknown on the real drone -> 0; z from the barometer)
    get_vel()  -> [vx, vy, vz]   (no direct velocity source -> 0; step() leans on IMU instead)
    get_quat() -> [w, x, y, z]   (from the FC attitude; not needed by the position controller)
    set_propellers_rpm(rpms)     (stored, not actuated - there is no motor path here)

Minimal use (no runner needed):
    import rclpy
    from realCameraTest.realDroneStruct import RealDroneStruct
    # (import the controller only after installing the genesis stub; that's the runner's job)
    d = RealDroneStruct()                 # creates its own node + background spin
    # ctrl = DronePositionCTRL(d, dt=0.01)
    # while True:
    #     ctrl.step()                     # reads imu.read()+get_vel(), integrates state
    #     U = ctrl.position_control(setpoint)   # [roll, pitch, yaw_rate, throttle]
    d.close()
"""


import threading
import numpy as np


# --- tiny shims so imu.read() looks like the sim's Genesis sensor reading ------------------

class _Vec:
    """Mimics a Genesis sensor field: `.numpy()` returns the array."""
    def __init__(self, a):
        self._a = np.asarray(a, dtype=float)

    def numpy(self):
        return self._a


class _Reading:
    def __init__(self, lin_acc, ang_vel, mag):
        self.lin_acc = _Vec(lin_acc)
        self.ang_vel = _Vec(ang_vel)
        self.mag = _Vec(mag)


class _IMU:
    def __init__(self, struct):
        self._s = struct

    def read(self):
        return _Reading(self._s.accel, self._s.gyro, self._s.mag)


# --- the real drone struct -----------------------------------------------------------------

class RealDroneStruct:

    def __init__(self, drone_ns="/drone", node=None):
        import rclpy
        from geometry_msgs.msg import Vector3Stamped
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Imu, MagneticField
        from std_msgs.msg import Float32

        self.imu = _IMU(self)

        # Latest sensor values, overwritten by the ROS callbacks.
        self.accel = np.array([0.0, 0.0, 9.81])   # specific force at rest (gravity, body z up)
        self.gyro = np.zeros(3)                    # [wx, wy, wz] rad/s
        self.mag = np.zeros(3)                     # [mx, my, mz]
        self.att = np.zeros(3)                     # [roll, pitch, yaw] rad, from the FC
        self.baro_z = 0.0                          # altitude above startup pressure level [m]
        self.last_rpm = np.zeros(4)                # whatever the controller asked for (not sent)

        # Own node + spin thread unless the caller supplies a node.
        self._owns_node = node is None
        if self._owns_node and not rclpy.ok():
            rclpy.init()
        self.node = node or rclpy.create_node("real_drone_struct")

        ns = drone_ns.rstrip("/")
        self.node.create_subscription(Imu, ns + "/imu", self._on_imu, qos_profile_sensor_data)
        self.node.create_subscription(MagneticField, ns + "/mag", self._on_mag, qos_profile_sensor_data)
        self.node.create_subscription(Vector3Stamped, ns + "/attitude", self._on_att, qos_profile_sensor_data)
        self.node.create_subscription(Float32, ns + "/baro_altitude", self._on_baro, 10)

        self._spinning = False
        self._thread = None
        if self._owns_node:
            from rclpy.executors import SingleThreadedExecutor
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self.node)
            self._spinning = True
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()

    # --- ROS callbacks: just store the latest value ---
    def _on_imu(self, m):
        self.accel = np.array([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
        self.gyro = np.array([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])

    def _on_mag(self, m):
        self.mag = np.array([m.magnetic_field.x, m.magnetic_field.y, m.magnetic_field.z])

    def _on_att(self, m):
        self.att = np.array([m.vector.x, m.vector.y, m.vector.z])

    def _on_baro(self, m):
        self.baro_z = float(m.data)

    def _spin(self):
        import rclpy
        while self._spinning and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.05)

    # --- the DroneStruct interface the controllers call ---
    def get_pos(self):
        # No absolute x/y on the real drone; z from the barometer.
        return np.array([0.0, 0.0, self.baro_z])

    def get_vel(self):
        # No GPS/mocap velocity. Returning zero makes DroneCTRL.step lean on IMU integration
        # (its complementary filter blends this at only 5%).
        return np.zeros(3)

    def get_quat(self):
        return _euler_to_quat(self.att[0], self.att[1], self.att[2])

    def set_propellers_rpm(self, rpms):
        # No motor path here: we are only exercising the position controller's math. Store the
        # request so a caller can inspect it; a real runner would convert U to /drone/cmd_attitude.
        self.last_rpm = np.asarray(rpms, dtype=float)

    def close(self):
        import rclpy
        self._spinning = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._owns_node:
            try:
                self._executor.shutdown()
            except Exception:
                pass
            try:
                self.node.destroy_node()
            except Exception:
                pass
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass


def _euler_to_quat(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])
