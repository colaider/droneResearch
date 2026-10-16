import numpy as np
import threading
import time

from filters.FKF import FusionKalmanFilter
from sensors.IMU.IMU import IMU
from sensors.IMU.IMUBridge import IMUBridge
from sensors.barometer import Barometer
from sensors.visual.config.cameraConfig import STEREO_CAM


class Sensors:
    def __init__(self, baro_vel_ema=0.8, *, node=None, camera_ns="/stereo",
                 publish_processed_camera=True, async_visual_processing=False,
                 camera_config=None, image_filters=("clahe",),
                 convert_to_grayscale=True, filters_enabled=True,
                 filter_kernel_size=3,
                 max_stereo_skew_s=0.005, vision_backend=None):
        from sensors.visual import VisulaEst

        self.camera_receiver = None
        self.visual_frame_bridge = None
        self.node = node
        if node is not None:
            from communication.CameraStreamReceiver import CameraStreamReceiver
            from communication.VisualFrameBridge import VisualFrameBridge

            self.camera_receiver = CameraStreamReceiver(node, ns=camera_ns)
            if publish_processed_camera:
                self.visual_frame_bridge = VisualFrameBridge(node)

        self.imu = IMU()                  # accel + gyro estimator (de-biasing, attitude, body accel)
        self.sample = None                # latest fully timestamped ImuSample (accel+gyro+baro+t)
        self.imu_bridge = IMUBridge(capacity=50)

        # Fusion: the Kalman filter REPLACES the old low-pass blend. IMU accel drives the predict,
        # the camera velocity corrects it (with its timestamp for out-of-sequence handling).
        self.fkf = FusionKalmanFilter()
        self._pending_cam = None          # one unconsumed camera measurement: (v4_world, t_s, idx)

        # barometer: own class, calibrated with the IMU (zeroed at start), feeds the FKF
        self.barometer = Barometer(baro_vel_ema)
        self.latest_camera_frame = None
        self._visual_position = None
        self._visual_velocity_world = None     # latest camera velocity in world frame (nan axis = no measurement)
        self._last_visual_frame_idx = None
        self._visual_state_lock = threading.Lock()
        self._visual_condition = threading.Condition()
        self._pending_visual_frame = None
        self._stop_visual_worker = False
        self._visual_worker = None
        self.visual_estimator = VisulaEst(
            STEREO_CAM if camera_config is None else camera_config,
            image_filters=image_filters,
            convert_to_grayscale=convert_to_grayscale,
            filters_enabled=filters_enabled,
            filter_kernel_size=filter_kernel_size,
            max_stereo_skew_s=max_stereo_skew_s,
        )
        if async_visual_processing and self.camera_receiver is not None:
            self._visual_worker = threading.Thread(
                target=self._visual_worker_loop,
                name="visual-estimator",
                daemon=True,
            )
            self._visual_worker.start()

    # ---------------- update ----------------
    def update(self, accel, gyro, baro, dt, t=None, command_velocity=None):
        # Build the timestamped record: the IMU makes it (gyro+accel+stamp, baro empty), then
        # the Sensors facade fills the barometer -> one fully timestamped sample per update.
        self.sample = self.imu.build_sample(accel, gyro, t)
        self.sample.baro = float(baro)

        was_calibrating = self.imu.is_calibrating()
        self.imu.update(accel, gyro, dt)
        if t is not None and np.isfinite(t):
            timestamp_ns = int(round(float(t) * 1_000_000_000))
            timestamp_ns = self._ros_to_monotonic_ns(timestamp_ns)
            self.imu_bridge.capture(self.imu, timestamp_ns)
        if was_calibrating and not self.imu.is_calibrating():
            with self._visual_state_lock:
                self._visual_position = None
                self._visual_velocity_world = None
                self._last_visual_frame_idx = None
                self._pending_cam = None
            self._reset_fkf()


        self.barometer.update(float(baro), dt)   # altitude + vz; fused into the FKF below, not the IMU

        self._push_visual_sensor_sample(command_velocity)
        if self.camera_receiver is not None:
            frame = self.camera_receiver.get_frame()
            if frame is not None:
                self.latest_camera_frame = frame
                if self._visual_worker is None:
                    self.process_visual_frame(frame)
                else:
                    with self._visual_condition:
                        self._pending_visual_frame = frame
                        self._visual_condition.notify()

        # --- Kalman fusion: IMU accel drives predict, baro anchors vertical, camera corrects
        # horizontal (with its timestamp for out-of-sequence replay). Replaces the old low-pass. ---
        if not self.imu.is_calibrating():
            v_cam, cam_id = self._pop_camera_measurement()
            self.fkf.step(self.imu.get_body_accel(), dt, self.imu.get_attitude(), v_camera=v_cam, camera_id=cam_id)
            self.fkf.update_baro(self.barometer.get_altitude(), self.barometer.get_velocity())

    def _visual_worker_loop(self):
        while True:
            with self._visual_condition:
                while self._pending_visual_frame is None and not self._stop_visual_worker:
                    self._visual_condition.wait()
                if self._stop_visual_worker:
                    return
                frame = self._pending_visual_frame
                self._pending_visual_frame = None
            self.process_visual_frame(frame)

    def _push_visual_sensor_sample(self, command_velocity=None):
        estimator = self.visual_estimator
        estimator.drone_pos = self.get_fused_position()
        estimator.drone_vel = self.get_acc_velocity()
        estimator.drone_ang_vel = self.get_angular_velocity()
        estimator.imu_att = self.get_attitude()
        estimator.drone_cmd_vel = (
            np.zeros(4)
            if command_velocity is None
            else np.asarray(command_velocity, dtype=float).copy()
        )
        estimator.push_sensors()

    def process_visual_frame(self, frame, idx=None):
        """Process a stereo capture and integrate valid visual velocity into position."""
        imu_snapshots = ()
        frame_timestamp_ns = getattr(frame, "left_timestamp_ns", None)
        if frame_timestamp_ns is not None:
            monotonic_timestamp_ns = self._ros_to_monotonic_ns(frame_timestamp_ns)
            previous_frame = self.visual_estimator.buffer[-1] if self.visual_estimator.buffer else None
            previous_timestamp_ns = getattr(previous_frame, "left_timestamp_ns", None)
            if previous_timestamp_ns is not None:
                previous_monotonic_ns = self._ros_to_monotonic_ns(previous_timestamp_ns)
                imu_snapshots = self.imu_bridge.between(
                    previous_monotonic_ns, monotonic_timestamp_ns
                )
            if not imu_snapshots:
                closest = self.imu_bridge.closest(
                    monotonic_timestamp_ns, max_delta_ns=25_000_000
                )
                imu_snapshots = () if closest is None else (closest,)
        processed = self.visual_estimator.processing(frame, idx, imu_snapshots=imu_snapshots)
        if self.visual_frame_bridge is not None:
            self.visual_frame_bridge.publish( processed, self.visual_estimator.tracks,
            self.visual_estimator.point_depth, self.visual_estimator, )
        frame_idx = processed.idx

        with self._visual_state_lock:
            if frame_idx == self._last_visual_frame_idx:
                return processed

            self._last_visual_frame_idx = frame_idx
        estimator = self.visual_estimator

        if (np.count_nonzero(estimator.velocity_point_mask) < 2 or not np.isfinite(estimator.camera_velocity[:3]).all()):
            return processed

        imu_position = estimator.avg_pos.copy()
        dt = float(estimator.dt)

        roll, pitch, yaw = estimator.avg_att
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        body_to_world = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])

        has_visual_vertical_velocity = np.isfinite(estimator.depth_velocity_z)
        visual_velocity = estimator.camera_velocity[:3].copy()

        if not has_visual_vertical_velocity:
            visual_velocity[2] = 0.0

        visual_velocity_world = body_to_world @ visual_velocity

        held = visual_velocity_world.copy()
        if not has_visual_vertical_velocity: held[2] = np.nan

        yaw_rate = float(estimator.camera_velocity[3])
        v4 = np.array([visual_velocity_world[0], visual_velocity_world[1], visual_velocity_world[2], yaw_rate])

        with self._visual_state_lock:
            visual_position = (imu_position.copy() if self._visual_position is None
                               else self._visual_position.copy())
            visual_position += visual_velocity_world * dt

            self._visual_position = visual_position
            self._visual_velocity_world = held
            self._pending_cam = (v4, frame_idx)
        return processed

    def get_visual_velocity(self):
        """Return latest camera velocity [vx, vy, vz, yaw_rate]."""
        visula_velocity = self.visual_estimator.camera_velocity.copy()
        return visula_velocity

    def get_visual_position(self):
        """Return integrated visual position, or None until visual tracks are usable."""
        with self._visual_state_lock:
            return None if self._visual_position is None else self._visual_position.copy()

    def get_visual_velocity_world(self):
        """Return latest camera velocity in the world frame [vx, vy, vz] (nan axis = no
        measurement), or None until visual tracks are usable."""
        with self._visual_state_lock:
            return (None if self._visual_velocity_world is None
                    else self._visual_velocity_world.copy())

    def close(self):
        """Release optional camera-topic subscriptions."""
        if self._visual_worker is not None:
            with self._visual_condition:
                self._stop_visual_worker = True
                self._pending_visual_frame = None
                self._visual_condition.notify_all()
            self._visual_worker.join()
            self._visual_worker = None
        if self.camera_receiver is not None:
            self.camera_receiver.close()
            self.camera_receiver = None
        if self.visual_frame_bridge is not None:
            self.visual_frame_bridge.close()
            self.visual_frame_bridge = None

    # ---------------- calibration (IMU + barometer, zeroed together at start) ----------------
    def start_calibration(self, n=400):
        with self._visual_state_lock:
            self._visual_position = None
            self._visual_velocity_world = None
            self._last_visual_frame_idx = None
        self.imu.start_calibration(n)
        self.barometer.start_calibration(n)

    def is_calibrating(self):
        return self.imu.is_calibrating()

    def get_sample(self):
        return self.sample

    def get_imu_snapshots(self):
        """Return retained IMU state snapshots, timestamped in host monotonic time."""
        return self.imu_bridge.snapshots

    def _ros_to_monotonic_ns(self, timestamp_ns):
        if self.node is None:
            return timestamp_ns
        mono_before_ns = time.monotonic_ns()
        ros_now_ns = self.node.get_clock().now().nanoseconds
        mono_after_ns = time.monotonic_ns()
        mono_mid_ns = (mono_before_ns + mono_after_ns) // 2
        return int(timestamp_ns - (ros_now_ns - mono_mid_ns))

    # ---------------- forwarded IMU estimates ----------------
    def get_attitude(self):
        return self.imu.get_attitude()

    def get_acc_velocity(self):
        return self.imu.get_velocity()

    def get_acc_position(self):
        return self.imu.get_position()

    def get_acceleration(self):
        return self.imu.get_acceleration()

    def get_angular_velocity(self):
        return self.imu.get_angular_velocity()

    # ---------------- barometer ----------------
    def get_baro_position(self):
        return self.barometer.get_altitude()   # zeroed altitude [m]

    def get_baro_velocity(self):
        return self.barometer.get_velocity()   # vertical velocity from the barometer

    # ---------------- fused (Kalman filter, not a low-pass blend) ----------------
    def _pop_camera_measurement(self):
        """Take the one unconsumed camera measurement for the FKF: (v4_world, idx) or (None, None)."""
        with self._visual_state_lock:
            pending, self._pending_cam = self._pending_cam, None
        if pending is None:
            return None, None
        return pending                       # (v4, idx)

    def _reset_fkf(self):
        """Re-origin the filter at calibration end: zero velocity, position 0, yaw = current heading."""
        self.fkf.v[:] = 0.0
        self.fkf.reset_position(p=np.zeros(3), yaw=float(self.imu.get_attitude()[2]))

    def get_fused_position(self):
        return self.fkf.get_position()       # world [x, y, z] from the Kalman filter

    def get_fused_velocity(self):
        return self.fkf.get_velocity()       # world [vx, vy, vz] from the Kalman filter
