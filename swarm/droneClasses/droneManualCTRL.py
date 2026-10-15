import types

import cv2
import numpy as np
import genesis as gs
import genesis.utils.geom as gu

from swarm.droneClasses.dronePositionCTRL import DronePositionCTRL
from swarm.droneClasses.droneRealism import (
    KF,
    MotorLagFilter,
    aerodynamic_drag_force_body_frame,
    ground_effect_extra_thrust,
)


def _camera_show_with_key(self):
    """
    Replacement for DroneStruct.camera_show(): identical CV display behaviour, but
    returns the cv2.waitKey code instead of discarding it. Patched onto individual
    DroneStruct instances (see DroneManualCTRL.__init__) rather than edited in
    droneStruct.py, so the CV pipeline itself is untouched.
    """
    for i, frame in enumerate(self.postpocessed_frames):
        cv2.imshow(str(i), frame)
    return cv2.waitKey(1)


class DroneManualCTRL(DronePositionCTRL):
    """
    Keyboard-teleoperated position controller. Reads key codes from the same
    cv2 windows already opened by the stereo CV pipeline, so the drone must be
    flown with one of those windows focused.

    Keys: w/a/s/d move forward/left/back/right (relative to current yaw),
    r/f ascend/descend, q/e yaw left/right.
    """

    # Matches the custom_racer.urdf drone (kf=1.07e-8, 1.0kg operating mass), which has a
    # much higher thrust coefficient than the stock cf2x drone DroneCTRL's defaults assume.
    # hover_rpm = sqrt(m*g/(4*kf)); max_rpm is the ESTIMATED max RPM (~27500, not yet measured).
    CUSTOM_MAX_RPM = 27500
    CUSTOM_HOVER_RPM = 15146

    def __init__(self, drone_entity, dt: float = 0.01, start_z: float = 1.2):
        super().__init__(drone_entity, dt)

        # Patch this drone instance's camera_show so it returns the pressed key,
        # without modifying the DroneStruct class itself.
        drone_entity.camera_show = types.MethodType(_camera_show_with_key, drone_entity)

        # Override the RPM limits DroneCTRL.__setup_low_level_control_variables set for the
        # stock cf2x drone -- they'd overpower the real drone's motors many times over.
        self.max_rpm = self.CUSTOM_MAX_RPM
        self.hover_rpm = self.CUSTOM_HOVER_RPM
        self.hover_throttle = self.hover_rpm / self.max_rpm

        # Velocity-loop gains (self.kp/kd/ki, index order [vx, vy, vz, yaw]) are also
        # inherited from DroneCTRL's cf2x-tuned defaults (kp=[2,2,1,0.5]) and saturate
        # almost immediately against this drone's real thrust curve:
        #
        # Per-motor thrust sensitivity to throttle-fraction at hover, from T = kf*rpm^2
        # linearized around hover_rpm: dT/du = 2*kf*max_rpm*hover_rpm
        #   = 2 * 1.07e-8 * 27500 * 15146  ~= 8.91 N per unit throttle-fraction.
        #
        # compute_tilt() uses kp[0]/kp[1] to turn horizontal velocity error directly into
        # a target tilt angle (radians), clipped to max_tilt=15deg=0.2618 rad. At kp=2,
        # that clips at just 0.2618/2 = 0.13 m/s of velocity error -- any real movement
        # commands an instant step to max tilt (bang-bang), not a proportional response.
        #
        # lowLevelControl() uses kp[2] to turn vertical velocity error directly into a
        # throttle-fraction added equally to all 4 motors. Climb headroom above hover is
        # (1 - hover_throttle) ~= 0.449 (this drone's TWR=3.3 is much punchier than
        # cf2x's 2.25, so there's more headroom to blow through). At kp=1.0, that clips
        # at just 0.449/1.0 ~= 0.45 m/s of vertical velocity error.
        #
        # Retuned so each channel saturates around a few m/s of error instead of a
        # fraction of one; kd and ki[2] scaled down by the same factor as their kp to
        # keep the loops' relative damping/integral-time-constant unchanged. The roll/
        # pitch attitude loop (kp_att/kd_att) was checked separately (tau = 4*arm*(dT/du)*
        # cmd, then omega_n = sqrt(tau_per_unit/I), zeta = kd_att*tau_per_unit/(2*omega_n))
        # and comes out well-damped (zeta ~= 0.82 roll, ~= 1.06 pitch) against this
        # drone's real inertia already, so it is left as DroneCTRL set it.
        self.kp[0] = self.kp[1] = 0.10   # vx, vy: saturate ~2.5 m/s error instead of ~0.13 m/s
        self.kd[0] = self.kd[1] = 0.0005
        self.kp[2] = 0.22                # vz: saturate ~2 m/s error instead of ~0.45 m/s
        self.kd[2] = 0.0022
        self.ki[2] = 0.09

        self.setpoint = np.array([0.0, 0.0, start_z, 0.0])
        self.move_step = 0.05          # m per keypress, horizontal
        self.vert_step = 0.05          # m per keypress, vertical
        self.yaw_step = np.radians(3)  # rad per keypress
        self.min_z = 0.2

        self._motor_lag = MotorLagFilter(n_motors=4)
        self._prop_links = None  # lazily resolved on first set_propeller_rpm call (post scene.build())
        self._body_link = None

    def set_propeller_rpm(self, rpms):
        """
        Overrides DroneCTRL.set_propeller_rpm (see swarm/droneClasses/droneRealism.py):
        applies motor/ESC first-order response lag to the commanded RPM, adds each
        propeller's ground-effect thrust boost, and adds aerodynamic drag on the airframe
        -- all as extra external forces on top of the RPM-driven thrust Genesis already
        applies internally.
        """
        cmd_rpm = np.clip(rpms * self.max_rpm, 0, self.max_rpm)
        actual_rpm = self._motor_lag.step(cmd_rpm, self.dt)
        self.drone.set_propellers_rpm(actual_rpm)

        if self._prop_links is None:
            self._prop_links = [self.drone.get_link(f"prop{i}_link") for i in range(4)]
            self._body_link = self.drone.get_link("base_link")

        height_m = float(self.drone.get_pos()[2])
        for link, rpm in zip(self._prop_links, actual_rpm):
            base_thrust = KF * rpm ** 2
            extra = ground_effect_extra_thrust(base_thrust, height_m)
            if extra > 0.0:
                link.apply_external_force(
                    force=[0.0, 0.0, extra],
                    ref=gs.link_ref_frame.link_COM,
                    local=True,
                )

        world_vel = self.drone.get_vel().numpy()
        body_vel = gu.inv_transform_by_quat(world_vel, self.drone.get_quat().numpy())
        drag_force = aerodynamic_drag_force_body_frame(body_vel)
        self._body_link.apply_external_force(
            force=list(drag_force),
            ref=gs.link_ref_frame.link_COM,
            local=True,
        )

    def handle_key(self, key: int):
        """Nudge the setpoint based on a raw cv2.waitKey code (-1 means no key pressed)."""
        if key < 0:
            return

        yaw = self.setpoint[3]
        forward = np.array([np.cos(yaw), np.sin(yaw)])
        right = np.array([np.sin(yaw), -np.cos(yaw)])
        key_char = chr(key & 0xFF).lower()

        if key_char == 'w':
            self.setpoint[:2] += forward * self.move_step
        elif key_char == 's':
            self.setpoint[:2] -= forward * self.move_step
        elif key_char == 'a':
            self.setpoint[:2] -= right * self.move_step
        elif key_char == 'd':
            self.setpoint[:2] += right * self.move_step
        elif key_char == 'r':
            self.setpoint[2] += self.vert_step
        elif key_char == 'f':
            self.setpoint[2] = max(self.min_z, self.setpoint[2] - self.vert_step)
        elif key_char == 'q':
            self.setpoint[3] += self.yaw_step
        elif key_char == 'e':
            self.setpoint[3] -= self.yaw_step

    def fly(self):
        """Call once per step (after start() reports the hover phase is done) to track the current setpoint."""
        self.position_control(self.setpoint)
