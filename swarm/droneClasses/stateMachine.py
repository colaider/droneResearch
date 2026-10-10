from swarm.droneClasses.droneVisulaControler import DroneVisulaCTRL
import numpy as np


class StateMachine:
    def __init__(self, drone_entity, dt, camera_fps):
        self.controller = DroneVisulaCTRL(drone_entity, dt, camera_fps)
        self.target = None
        self.tol = 0.1
        self.stage = 0

    def tick(self, step, sim_time):
        self.controller.camera_update_step(step, sim_time)

    def startup(self, step, sim_time):
        self.controller.calibrate()
        return getattr(self.controller, "calibrated", False)

    def _reached(self):
        return np.linalg.norm(self.controller.cam_pos[:3] - self.target[:3]) < self.tol

    def hower(self, altitude=1):
        if self.target is None:
            self.target = self.controller.cam_pos.copy()
            self.target[2] = altitude

        self.controller.position_ctrl_fused(self.target)
        done = self._reached()

        if done: self.target = None
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
            self.hower(3)


       