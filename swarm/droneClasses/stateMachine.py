from swarm.droneClasses.droneVisulaControler import DroneVisulaCTRL
import numpy as np


class StateMachine:
    def __init__(self, drone_entity, dt, camera_fps):
        self.controller = DroneVisulaCTRL(drone_entity, dt, camera_fps)
        self.target = None
        self.tol = 0.3
        self.dwell_need = 20            # ticks the drone must stay inside tol
        self._dwell = 0
        self.stage = 0

    def tick(self, step, sim_time):
        self.controller.camera_update_step(step, sim_time)

    def startup(self, step, sim_time):
        self.controller.calibrate()

        true = np.asarray(self.controller.get_position())
        cam  = getattr(self.controller, "cam_pos", None)
        imu  = self.controller.get_imu_pos()

        print(f"[startup step={step}] "
            f"true={np.round(true,3)}  "
            f"cam={None if cam is None else np.round(cam,3)}  "
            f"imu={np.round(imu,3)}  "
            f"calibrated={getattr(self.controller,'calibrated',False)}")

        return getattr(self.controller, "calibrated", False)

    def _reached(self):
        inside = np.linalg.norm(self.controller.cam_pos[:3] - self.target[:3]) < self.tol
        self._dwell = self._dwell + 1 if inside else 0
        return self._dwell >= self.dwell_need

    def hower(self, altitude=1):
        if self.target is None:
            self.target = self.controller.cam_pos.copy()
            self.target[2] = altitude
            self._dwell = 0             # reset on new target

        print("fused_z=%.3f  true_z=%.3f" % (self.controller.cam_pos[2], self.controller.get_position()[2]))
        self.controller.position_ctrl_fused(self.target)
        done = self._reached()
        if done:
            self.target = None; self._dwell = 0
        return done

    def heading_to_relative(self, cmd):
        if self.target is None: self.target = self.controller.cam_pos.copy() + np.asarray(cmd, float)

        self.controller.position_ctrl_fused(self.target)
        done = self._reached()
        if done: self.target = None

        return done


    def execute_trajectory(self, segments, durations):
        """
        segments  : list of per-segment coefficient arrays, shape (M, 4, 8) 
                    (M segments, 4 axes [x,y,z,yaw], 8 coeffs each)
        durations : list of per-segment durations T_i
        """
        t_abs = 0.0
        T_cum = np.cumsum([0] + list(durations))  # start time of each segment

        while t_abs < T_cum[-1]:
            # 1. Find which segment we're in and local time within it
            seg_idx = np.searchsorted(T_cum[1:], t_abs, side='right')
            tau = t_abs - T_cum[seg_idx]
            c = segments[seg_idx]            # (4, 8)

            # 2. Evaluate position, velocity, (optionally) accel at tau
            powers = np.array([tau**k for k in range(8)])
            d_powers = np.array([k * tau**(k-1) if k >= 1 else 0 for k in range(8)])
            dd_powers = np.array([k*(k-1) * tau**(k-2) if k >= 2 else 0 for k in range(8)])

            pos_ref = c @ powers              # (4,) — [x, y, z, yaw]
            vel_ref = c @ d_powers            # (4,) — [vx, vy, vz, yaw_rate]
            acc_ref = c @ dd_powers           # (4,) — [ax, ay, az, yaw_accel]

            # 3. Feed into the position controller
            self.position_control(
                setpoint=pos_ref,
                # if you add feedforward support:
                vel_ff=vel_ref,
                acc_ff=acc_ref,
            )
            self.scene.step()
            t_abs += self.dt



    def steping(self, step, sim_time):
        self.tick(step, sim_time)

        if self.stage == 0:
            status = self.startup(step, sim_time)
            if status: self.stage = 1

        elif self.stage == 1:
            err = np.linalg.norm(self.controller.cam_pos[:3] - self.target[:3]) if self.target is not None else -1
            status = self.hower(4)
            if status: self.stage = 2

        elif self.stage == 2:
            status = self.heading_to_relative(np.array([5,0,0,0]))
            if status == 1: self.stage = 3

        elif self.stage == 3:
            status = self.hower(3)

       