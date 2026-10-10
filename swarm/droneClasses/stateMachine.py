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
        self.controller.position_ctrl_fused(self.target)
        done = self._reached()
        if done:
            print('AAAAAAA'); self.target = None; self._dwell = 0
        return done

    def heading_to_relative(self, cmd):
        if self.target is None: self.target = self.controller.cam_pos.copy() + np.asarray(cmd, float)

        self.controller.position_ctrl_fused(self.target)
        done = self._reached()
        if done: self.target = None

        return done


    def steping(self, step, sim_time):
        self.tick(step, sim_time)

        if self.stage == 0:
            status = self.startup(step, sim_time)
            if status: self.stage = 1

        elif self.stage == 1:
            err = np.linalg.norm(self.controller.cam_pos[:3] - self.target[:3]) if self.target is not None else -1
            status = self.hower(4)
            if step % 50 == 0:
                print(f"[stage1] err={err:.3f} tol={self.tol} dwell={self._dwell} status={status}")
            if status: self.stage = 2

        elif self.stage == 2:
            status = self.heading_to_relative(np.array([5,0,0,0]))
            if status == 1: self.stage = 3

        elif self.stage == 3:
            status = self.hower(3)

       