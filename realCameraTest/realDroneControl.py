"""Middle layer: fly the REAL drone with the swarm visual controller + state machine.

It glues three things together WITHOUT changing anything in `swarm/`:

    rb3_stereo (cameras)  ->  swarm VisulaAcEst / DroneVisulaCTRL / StateMachine  ->  rb3_control (FC)
         stereo pair                 vision + fusion + control                      /drone/cmd_attitude

Data flow (closed loop, per tick):
  1. rb3_stereo delivers a stereo image. It is SEPARATED into (left, right) and sent to the
     visual processor (`frame_processor.processing([left, right], idx)`).
  2. The RB3's own IMU (/drone/imu) feeds the controller's state estimator; the visual estimator
     adds body velocity; the FKF fuses them.
  3. The swarm StateMachine runs startup -> calibrate -> hover (the TRAJECTORY stage is skipped).
  4. The controller's output U = [roll, pitch, yaw_rate, throttle] is published to
     /drone/cmd_attitude. Betaflight runs its own inner loop + mixer. We never touch motors/RPM.

Everything swarm-side is reused unchanged: `RB3DroneBridge` mimics the sim's `DroneStruct`
interface, and `AttitudeDroneCTRL` only *overrides* `lowLevelControl` to emit cmd_attitude.

=====================================================================================
DANGERS (read before flying) -- this commands a real aircraft:
  * Radio stays the authority. rb3_control only streams sticks while the radio has ARMED the
    drone AND the MSP OVERRIDE switch is on; dropping either hands control back in ~100 ms. Keep
    a finger on the override switch. This middle layer never arms/disarms.
  * DRY RUN IS THE DEFAULT. Without --fly it computes and logs commands but publishes NOTHING, so
    the FC keeps following the radio. Bench-test first (props off) and watch `sticks_fc` follow
    `sticks_sent` in /drone/status, per rb3_control/README.md bring-up order.
  * hover_throttle (0..1) is a SIM value by default and is almost certainly wrong for your craft.
    Measure it in a manual hover and pass --hover-throttle; too high = it leaps up.
  * Sign/axis conventions between swarm roll/pitch and Betaflight are UNVERIFIED. Confirm on the
    bench (tilt the drone, watch the sticks) before props on. --max-tilt-deg and --throttle-cap
    hard-clamp the output regardless.
  * No absolute position: horizontal hold is dead-reckoned from vision+IMU and WILL drift. z comes
    from the barometer. Treat position hold as loose. Fly only with room and a spotter.
  * calibrate() samples the gyro while blocking; keep the drone still and level during startup.
  * Vision needs texture/light and calibrated stereo for metric depth; uncalibrated => scale is
    assumed (--height). Bad velocity => bad commands. Start with --fly off and sanity-check the
    logged vx/vy/depth first.
=====================================================================================

Run on the drone (ROS 2 sourced, rb3_control + rb3_stereo running):
    python3 realCameraTest/realDroneControl.py                 # DRY RUN: logs only, publishes nothing
    python3 realCameraTest/realDroneControl.py --hover-throttle 0.55 --altitude 1.0 --fly
Keys: Ctrl-C stops (publishing stops instantly -> radio takes over).
"""
import argparse
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _ensure_genesis_stub():
    """swarm/droneClasses/droneStruct.py does `from genesis.utils.misc import tensor_to_array`
    at import time. That is a simulation-only helper (converts Genesis GPU tensors to numpy) and is
    never called on the real drone -- but importing the controller still loads droneStruct, so the
    `genesis` package must be importable. On the drone Genesis is not installed, so provide a tiny
    stub that satisfies the import without touching swarm. If Genesis *is* installed, use the real one.
    """
    try:
        import genesis  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    g = types.ModuleType("genesis")
    utils = types.ModuleType("genesis.utils")
    misc = types.ModuleType("genesis.utils.misc")
    misc.tensor_to_array = lambda x: np.asarray(x)   # only needed by sim's get_two_frames (unused here)
    utils.misc = misc
    g.utils = utils
    sys.modules["genesis"] = g
    sys.modules["genesis.utils"] = utils
    sys.modules["genesis.utils.misc"] = misc


_ensure_genesis_stub()

from swarm.compVision.visualAccEst import VisulaAcEst
from swarm.droneClasses.droneVisulaControler import DroneVisulaCTRL
from swarm.droneClasses.stateMachine import StateMachine


# --------------------------------------------------------------------------- frame separation

def split_stereo(frame):
    """Return (left, right) from whatever the stereo source hands over.

    The frame may already be a pair, or a single combined image that must be split:
      * [left, right] / (left, right)        -> used as is
      * ndarray (2, H, W[, C])               -> frame[0], frame[1]
      * ndarray (H, 2W[, C]) side-by-side    -> left half | right half
    """
    if isinstance(frame, (list, tuple)):
        if len(frame) != 2:
            raise ValueError(f"expected 2 frames, got {len(frame)}")
        return frame[0], frame[1]

    arr = np.asarray(frame)
    if arr.ndim >= 3 and arr.shape[0] == 2:            # stacked pair
        return arr[0], arr[1]
    if arr.ndim >= 2 and arr.shape[1] % 2 == 0:        # side-by-side single image
        half = arr.shape[1] // 2
        return arr[:, :half], arr[:, half:]
    raise ValueError(f"cannot separate stereo frame of shape {arr.shape}")


# --------------------------------------------------------------------------- IMU shim

class _Vec:
    """Mimics Genesis sensor fields: `.numpy()` returns the array."""
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
    def __init__(self, bridge):
        self._b = bridge
    def read(self):
        return _Reading(self._b.accel, self._b.gyro, self._b.mag)


# --------------------------------------------------------------------------- drone shim

class RB3DroneBridge:
    """Stands in for the simulation's DroneStruct, backed by RB3 ROS topics + rb3_stereo.

    Exposes exactly what swarm's DroneCTRL / DroneVisulaCTRL touch on `self.drone`:
      imu.read(), get_pos(), get_vel(), get_quat(), frame_processor, camera_step(),
      camera_show(), set_camera_dt(), set_propellers_rpm() (guarded - must never be reached).
    """
    def __init__(self, stereo, res, fov, baseline):
        self.imu = _IMU(self)
        self.frame_processor = VisulaAcEst(res, fov)
        self.frame_processor.camera_saperation = baseline
        self.steps_cam = 0
        self._stereo = stereo

        # latest sensor values, updated by ROS callbacks (background executor)
        self.accel = np.array([0.0, 0.0, 9.81])   # specific force at rest
        self.gyro = np.zeros(3)
        self.mag = np.zeros(3)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z)
        self.att = np.zeros(3)                       # measured [roll, pitch, yaw] from the FC [rad]
        self.baro_z = 0.0
        self._last_pair = None

    # --- sensors the controller reads ---
    def get_pos(self):
        # No absolute x/y on the real drone; z from the barometer. Used for logging / get_bar_z,
        # NOT for the control loop (that runs on the FKF's cam_pos).
        return np.array([0.0, 0.0, self.baro_z])

    def get_vel(self):
        # No GPS/mocap velocity. Returning zero makes the controller's complementary filter lean
        # on pure IMU integration (plus the visual velocity the FKF fuses in).
        return np.zeros(3)

    def get_quat(self):
        return self.quat.copy()

    # --- motors: must never be driven here (we publish attitude instead) ---
    def set_propellers_rpm(self, rpms):
        raise RuntimeError("RB3DroneBridge.set_propellers_rpm reached: lowLevelControl override "
                           "must publish cmd_attitude, not drive motors.")

    # --- camera: read a pair, SEPARATE it, run the visual processor ---
    def camera_step(self):
        frame = self._stereo.get_two_frames()
        if frame is None:               # timeout: reuse the last good pair so flow has two frames
            if self._last_pair is None:
                return self.frame_processor.current_frame
            left, right = self._last_pair
        else:
            left, right = split_stereo(frame)
            self._last_pair = (left, right)
        out = self.frame_processor.processing([left, right], self.steps_cam)
        self.steps_cam += 1
        return out

    def camera_show(self):
        pass

    def set_camera_dt(self, dt):
        self.frame_processor.set_dt(dt)


# --------------------------------------------------------------------------- attitude controller

class AttitudeDroneCTRL(DroneVisulaCTRL):
    """swarm visual controller whose low-level output is published as /drone/cmd_attitude
    (roll, pitch, yaw_rate, throttle) instead of being mixed to motor RPMs."""

    def __init__(self, drone, dt, camera_fps, publish_cmd,
                 hover_throttle=None, max_tilt_deg=15.0, throttle_cap=0.75, yaw_rate_cap=1.5):
        super().__init__(drone, dt, camera_fps)
        self._publish_cmd = publish_cmd
        self._max_tilt = np.radians(max_tilt_deg)
        self._throttle_cap = throttle_cap
        self._yaw_cap = yaw_rate_cap
        if hover_throttle is not None:
            self.hover_throttle = float(hover_throttle)   # real-craft value overrides the sim default

    def lowLevelControl(self, U):
        roll, pitch, yaw_rate, throttle = (float(x) for x in U)
        # Hard safety clamps, independent of the vision/control math.
        roll = float(np.clip(roll, -self._max_tilt, self._max_tilt))
        pitch = float(np.clip(pitch, -self._max_tilt, self._max_tilt))
        yaw_rate = float(np.clip(yaw_rate, -self._yaw_cap, self._yaw_cap))
        throttle = float(np.clip(throttle, 0.0, self._throttle_cap))
        self._publish_cmd(roll, pitch, yaw_rate, throttle)
        self.last_cmd = (roll, pitch, yaw_rate, throttle)


# --------------------------------------------------------------------------- ROS node + loop

class RealDroneNode:
    def __init__(self, args):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from geometry_msgs.msg import Twist, Vector3Stamped
        from sensor_msgs.msg import Imu
        from std_msgs.msg import Bool, Float32
        from rb3_stereo.client import StereoSubscriber

        self.args = args
        self._Twist = Twist
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("real_drone_control")
        drone_ns = args.drone_ns.rstrip("/")

        # sensors in
        self.node.create_subscription(Imu, f"{drone_ns}/imu", self._on_imu, qos_profile_sensor_data)
        self.node.create_subscription(Vector3Stamped, f"{drone_ns}/attitude", self._on_att, qos_profile_sensor_data)
        self.node.create_subscription(Float32, f"{drone_ns}/baro_altitude", self._on_baro, 10)
        self.node.create_subscription(Bool, f"{drone_ns}/armed", self._on_armed, 10)
        # commands out
        self.cmd_pub = self.node.create_publisher(Twist, f"{drone_ns}/cmd_attitude", 10)

        # stereo on our own node (shares the executor) so there is one spin thread
        self.stereo = StereoSubscriber(args.stereo_ns, bgr=True, node=self.node)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spinning = True
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        self.armed = False
        self._bridge = None

    def _spin(self):
        import rclpy
        while self._spinning and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.05)

    # --- sensor callbacks (fill the bridge's latest values) ---
    def _on_imu(self, m):
        if self._bridge is None:
            return
        self._bridge.accel = np.array([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
        self._bridge.gyro = np.array([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])

    def _on_att(self, m):
        if self._bridge is None:
            return
        self._bridge.att = np.array([m.vector.x, m.vector.y, m.vector.z])   # measured roll/pitch/yaw [rad]
        self._bridge.quat = _euler_to_quat(m.vector.x, m.vector.y, m.vector.z)

    def _on_baro(self, m):
        if self._bridge is not None:
            self._bridge.baro_z = float(m.data)

    def _on_armed(self, m):
        self.armed = bool(m.data)

    # --- command publishing ---
    def publish_cmd(self, roll, pitch, yaw_rate, throttle):
        if self.args.dry_run:
            return                      # DRY RUN: never command the FC
        if self.args.require_armed and not self.armed:
            return                      # extra belt-and-suspenders on top of rb3_control's gate
        msg = self._Twist()
        msg.angular.x = roll
        msg.angular.y = pitch
        msg.angular.z = yaw_rate
        msg.linear.z = throttle
        self.cmd_pub.publish(msg)

    def run(self):
        a = self.args
        # Wait for the camera_info so we start the estimator with the real res / fov / baseline.
        print(f"waiting for stereo camera_info on {a.stereo_ns}/ ... (up to {a.stereo_wait:.0f}s)", flush=True)
        try:
            self.stereo._wait_info(timeout=a.stereo_wait)
            res, fov, baseline = self.stereo.res, self.stereo.fov, self.stereo.camera_saperation
        except TimeoutError:
            self._report_no_stereo()
            self.close()
            return 1
        print(f"stereo: res={res} vfov={fov:.1f} deg baseline={baseline:.3f} m", flush=True)

        self._bridge = RB3DroneBridge(self.stereo, res, fov, baseline)

        # StateMachine builds a plain DroneVisulaCTRL; swap in the attitude-publishing controller.
        sm = StateMachine(self._bridge, dt=a.dt, camera_fps=a.fps)
        sm.controller = AttitudeDroneCTRL(
            self._bridge, a.dt, a.fps, self.publish_cmd,
            hover_throttle=a.hover_throttle, max_tilt_deg=a.max_tilt_deg,
            throttle_cap=a.throttle_cap, yaw_rate_cap=a.yaw_rate_cap)

        mode = "DRY RUN (publishing nothing)" if a.dry_run else "FLYING (publishing cmd_attitude)"
        cam_period = max(1, int(round((1.0 / a.fps) / a.dt)))   # control steps per camera frame
        print(f"=== {mode} === hover={a.altitude} m  hover_throttle={sm.controller.hover_throttle:.3f}  "
              f"tilt<= {a.max_tilt_deg} deg  throttle<= {a.throttle_cap}", flush=True)
        print(f"    control dt={a.dt:.4f}s ({1.0/a.dt:.0f} Hz)  camera fps={a.fps} "
              f"(camera_step every {cam_period} control steps)", flush=True)

        step, t0, next_t = 0, time.monotonic(), time.monotonic()
        try:
            while True:
                sim_time = time.monotonic() - t0

                sm.tick(step, sim_time)                 # IMU step + stereo (camera_step)

                if sm.stage == 0:                       # startup + gyro calibration
                    if sm.startup(step, sim_time):
                        sm.stage = 1
                else:                                   # hover only -- TRAJECTORY stage skipped
                    sm.hower(a.altitude)

                if step % 50 == 0:
                    fp = self._bridge.frame_processor
                    # camera_velocity = [vx, vy, z_estimated, yaw_rate]  (index 2 is the vision DEPTH
                    # scale, not a vertical velocity -- the controller replaces vz with IMU vz).
                    vx, vy, z_est, yaw = (float(x) for x in fp.camera_velocity)
                    cam = getattr(sm.controller, "cam_pos", None)
                    cmd = getattr(sm.controller, "last_cmd", None)

                    # --- camera: are any points actually detected / used this frame? ---
                    n_pts = int(len(fp.point_layers))                               # features tracked
                    n_vel = int(fp.velocity_point_mask.sum()) if fp.velocity_point_mask.size else 0
                    depth_pct = fp.depth_valid_frac * 100.0                          # % with stereo depth

                    # --- depth source check (the barometer question) ---
                    # Horizontal velocity scale comes ONLY from per-point stereo depth. When stereo
                    # drops out the swarm estimator falls back to IMU dead-reckoning for vx/vy -- the
                    # baro CANNOT scale horizontal flow from here (it only knows altitude). Altitude
                    # itself is always baro+IMU driven (vision vz is discarded), so z stays covered.
                    stereo_ok = np.isfinite(fp.depth_median) and fp.depth_valid_frac > 0.0
                    depth_src = ("stereo" if stereo_ok
                                 else "LOST -> vx/vy=IMU dead-reckoning (baro still holds altitude)")

                    # --- altitude actually used by the controller ---
                    baro_alt = self._bridge.baro_z              # RB3 baro altitude (BIASED: rel. to start-up level)
                    ctrl_alt = None if cam is None else float(cam[2])   # fused z the loop closes on

                    # --- motor bias: we publish cmd_attitude; Betaflight does the mixing ---
                    if cmd is None:
                        bias = "cmd=None (no attitude output yet -> nothing biasing motors)"
                    else:
                        r, pt, yr, th = cmd
                        bias = (f"cmd roll={r:+.3f} pitch={pt:+.3f} yaw_rate={yr:+.3f} throttle={th:.3f}"
                                f"  (FC mixes this to the 4 motors)")

                    # measured attitude coming back from the FC AFTER the bias is applied
                    mroll, mpitch, myaw = self._bridge.att
                    meas = (f"meas roll={mroll:+.3f} pitch={mpitch:+.3f} yaw={myaw:+.3f} rad")

                    print(f"[{step}] stage={sm.stage} armed={self.armed} "
                          f"dt={a.dt:.4f}s({1.0/a.dt:.0f}Hz) fps={a.fps} | "
                          f"points={n_pts} vel_used={n_vel} depth_valid={depth_pct:.0f}% "
                          f"depth_src={depth_src} | "
                          f"vx={vx:+.2f} vy={vy:+.2f} z_est={z_est:.2f} depth={fp.depth_median:.2f} | "
                          f"baro_alt={baro_alt:+.2f}m "
                          f"ctrl_alt={'n/a' if ctrl_alt is None else f'{ctrl_alt:+.2f}m'} "
                          f"setpoint={a.altitude:.2f}m | {bias} | {meas}", flush=True)

                step += 1
                next_t += a.dt
                time.sleep(max(0.0, next_t - time.monotonic()))
        except KeyboardInterrupt:
            print("\nstopping -> radio takes over", flush=True)
        finally:
            self.close()

    def _report_no_stereo(self):
        """No camera_info arrived: say why and what to check (instead of crashing)."""
        import os
        a = self.args
        try:
            topics = [n for n, _ in self.node.get_topic_names_and_types() if "stereo" in n]
        except Exception:
            topics = []
        print("\nERROR: no stereo camera_info received.", flush=True)
        print(f"  ROS_DOMAIN_ID here = {os.environ.get('ROS_DOMAIN_ID', '(unset -> 0)')}", flush=True)
        if topics:
            print("  stereo topics visible:", flush=True)
            for t in topics:
                print(f"    {t}", flush=True)
            print(f"  -> camera_info not seen under {a.stereo_ns}/. Check --stereo-ns matches the node.", flush=True)
        else:
            print("  No /stereo topics visible at all. Likely causes:", flush=True)
            print("   1. the camera node is not running   -> on the drone:  bash scripts/run_stereo.sh", flush=True)
            print("      (or:  ros2 run rb3_stereo stereo_camera)", flush=True)
            print("   2. ROS_DOMAIN_ID differs from the camera node (run_stereo.sh may set 42).", flush=True)
            print("      Match it, e.g.:  export ROS_DOMAIN_ID=42   then rerun this.", flush=True)
            print("   3. check with:  ros2 topic list | grep stereo", flush=True)

    def close(self):
        import rclpy
        self._spinning = False
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)
        try:
            self.stereo.close()
        except Exception:
            pass
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
                rclpy.shutdown()      # clean teardown -> avoids "Aborted (core dumped)" on exit
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Real-drone middle layer (swarm controller + RB3).")
    p.add_argument('--fly', dest='dry_run', action='store_false',
                   help='actually publish cmd_attitude (default: DRY RUN, publishes nothing)')
    p.add_argument('--altitude', type=float, default=1.0, help='hover altitude setpoint [m]')
    p.add_argument('--hover-throttle', type=float, default=None,
                   help='measured hover throttle 0..1 (default: swarm sim value -- almost certainly wrong)')
    p.add_argument('--max-tilt-deg', type=float, default=15.0, help='hard roll/pitch clamp [deg]')
    p.add_argument('--throttle-cap', type=float, default=0.75, help='hard throttle clamp 0..1')
    p.add_argument('--yaw-rate-cap', type=float, default=1.5, help='hard yaw-rate clamp [rad/s]')
    p.add_argument('--dt', type=float, default=0.01, help='control loop period [s]')
    p.add_argument('--fps', type=int, default=30, help='camera fps (for the camera step period)')
    p.add_argument('--stereo-ns', default='/stereo', help='rb3_stereo namespace')
    p.add_argument('--drone-ns', default='/drone', help='rb3_control namespace')
    p.add_argument('--stereo-wait', type=float, default=10.0, help='seconds to wait for camera_info before giving up')
    p.add_argument('--no-require-armed', dest='require_armed', action='store_false',
                   help='publish even when /drone/armed is false (rb3_control still gates)')
    p.set_defaults(dry_run=True, require_armed=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    node = RealDroneNode(args)
    return node.run() or 0


if __name__ == '__main__':
    raise SystemExit(main())
