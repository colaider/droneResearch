from swarm.droneClasses.droneVisulaControler import DroneVisulaCTRL
from swarm.droneClasses.trajectoryBuilder import MINCO
import numpy as np


class StateMachine:
    def __init__(self, drone_entity, dt, camera_fps):
        self.controller = DroneVisulaCTRL(drone_entity, dt, camera_fps)
        self.stage = 0
        self.target = None

        # trajectory state
        self.minco = None
        self.traj_t_abs = 0.0
        self.traj_T_total = 0.0

    # ---------------- tick ----------------

    def tick(self, step, sim_time):
        self.controller.camera_update_step(step, sim_time)

    # ---------------- primitives ----------------

    def startup(self, step, sim_time):
        self.controller.calibrate()
        true = np.asarray(self.controller.get_position())
        cam  = getattr(self.controller, "cam_pos", None)
        print(f"[startup step={step}] true={np.round(true,3)}  "
              f"cam={None if cam is None else np.round(cam,3)}  "
              f"calibrated={getattr(self.controller,'calibrated',False)}")
        return getattr(self.controller, "calibrated", False)

    def hower(self, altitude):
        """
        Approach the hover altitude via a MINCO trajectory from the current
        pose, then hold at the target forever. Call every tick.
        """
        if self.target is None:
            start = self.controller.cam_pos.copy()
            self.target = start.copy()
            self.target[2] = altitude
            self.build_trajectory(np.vstack([start, self.target]), v_target=1.0)

        if self.traj_t_abs < self.traj_T_total:
            self.execute_trajectory()
        else:
            self.controller.position_ctrl_fused(self.target)

    # ---------------- trajectory core ----------------

    def build_trajectory(self, waypoints, v_target=1.0,
                         close_loop=False, dur_floor=1.0):
        wp = np.asarray(waypoints, float)
        if close_loop:
            wp = np.vstack([wp, wp[0:1]])
        seg_dist  = np.linalg.norm(np.diff(wp[:, :3], axis=0), axis=1)
        durations = np.maximum(seg_dist / v_target, dur_floor)

        self.minco = MINCO(wp, durations)
        self.minco.create_sub_trajectories()
        self.traj_T_total = float(np.sum(durations))
        self.traj_t_abs = 0.0
        print(f"[traj] built: {len(wp)} wp, T={self.traj_T_total:.2f}s, "
              f"cost={self.minco.calculate_control_effort():.3f}")

    def execute_trajectory(self):
        p_ref, _, _ = self._sample_minco(self.traj_t_abs)
        self.controller.position_ctrl_fused(p_ref)
        self.traj_t_abs += self.controller.dt

    def _sample_minco(self, t_abs):
        T_cum = np.concatenate([[0.0], np.cumsum(self.minco.durations)])
        seg_idx = int(min(np.searchsorted(T_cum[1:], t_abs, side='right'),
                          self.minco.M - 1))
        tau = t_abs - T_cum[seg_idx]
        powers    = np.array([tau**k                          for k in range(8)])
        d_powers  = np.array([k*tau**(k-1)   if k >= 1 else 0 for k in range(8)])
        dd_powers = np.array([k*(k-1)*tau**(k-2) if k >= 2 else 0 for k in range(8)])
        D = self.minco.D
        p, v, a = np.zeros(D), np.zeros(D), np.zeros(D)
        for d in range(D):
            c = self.minco.segments[d][seg_idx].c
            p[d] = c @ powers
            v[d] = c @ d_powers
            a[d] = c @ dd_powers
        return p, v, a

    # ---------------- state machine ----------------

    def steping(self, step, sim_time):
        self.tick(step, sim_time)

        if self.stage == 0:
            if self.startup(step, sim_time):
                self.stage = 1

        elif self.stage == 1:
            self.hower(4)
