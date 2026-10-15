"""
Three physical effects the base RPM->thrust model (T = kf*rpm^2, applied only by
DroneEntity.set_propellers_rpm) does not capture on its own:

  1. Motor/ESC response lag: commanded RPM is not reached instantly.
  2. Ground effect: a rotor closer than a few rotor-radii to a surface below it produces
     MORE thrust than the same RPM would in open air, because its downwash recirculates
     off the surface instead of dispersing freely.
  3. Aerodynamic drag: air resistance opposing the airframe's own motion, independent of
     the propellers entirely.

Same three confidence tiers as mpu6000Imu.py:
  SOURCED    -- a documented physical constant or relation.
  DERIVED    -- computed from a sourced relation via a stated formula.
  ESTIMATED  -- no spec exists for this specific hardware; a stated, flagged assumption.

Physical constants duplicated here from genesis_drones/robots/assets/drone_urdf/custom_racer.urdf
(kept separate on purpose rather than importing/parsing the URDF, per request to keep this
work self-contained in new files).
"""

import math

KF = 1.07e-8          # thrust coefficient, N per (rpm)^2 -- custom_racer.urdf <properties kf=...>
PROP_RADIUS_M = 0.065  # rotor radius, m -- custom_racer.urdf <properties prop_radius=...>

# ---- Motor/ESC lag ----
# ESTIMATED: no datasheet for the actual motor+ESC pair, so this is a representative time
# constant for small high-KV FPV racing motors (commonly-cited spin-up response for this
# class of hardware is on the order of 10-30 ms). Measure the real value with a bench step-
# response test (command a throttle step, time how long RPM takes to reach ~63% of the
# commanded change -- that elapsed time IS the time constant) and replace this.
MOTOR_TAU_S = 0.02


class MotorLagFilter:
    """
    First-order (single-pole) lag: rpm_dot = (rpm_cmd - rpm)/tau. DERIVED: exact discrete-time
    solution of that ODE over a step dt is rpm[t+dt] = rpm[t] + (rpm_cmd - rpm[t])*(1 - exp(-dt/tau)),
    which is what `.step()` applies (not a first-order Euler approximation of it).
    """

    def __init__(self, n_motors: int, tau_s: float = MOTOR_TAU_S):
        self.tau_s = tau_s
        self.rpm = [0.0] * n_motors

    def reset(self, rpm=0.0):
        self.rpm = [rpm] * len(self.rpm)

    def step(self, cmd_rpm, dt: float):
        alpha = 1.0 - math.exp(-dt / self.tau_s)
        self.rpm = [r + (c - r) * alpha for r, c in zip(self.rpm, cmd_rpm)]
        return self.rpm


# ---- Ground effect ----
# SOURCED: Cheeseman & Bennett (1955) in-ground-effect thrust augmentation, the standard
# rotorcraft-aerodynamics result for a rotor of radius R hovering at height z above a
# surface (still the reference relation in, e.g., Leishman's "Principles of Helicopter
# Aerodynamics", and the same relation gym-pybullet-drones implements for multirotors):
#
#   T_in_ground_effect / T_open_air = 1 / (1 - (R / (4*z))^2)
#
# Applied here with coefficient 1 (the theoretical form) rather than an empirically-fitted
# scale factor, since no wind-tunnel/bench measurement exists for this specific airframe.
#
# The relation has a singularity at z = R/4 and is only physically meaningful for z
# noticeably larger than that; two engineering clamps make it usable in a real-time sim:
#   - Z_MIN_M (DERIVED, R/3): floors the height used in the ratio so it never reaches the
#     singularity. At z=R/3, (R/4z)^2 = (3/4)^2 = 0.5625 -> a ~2.29x thrust ceiling, i.e. at
#     most ~129% extra thrust, however close the drone actually gets to the surface.
#   - Z_MAX_M (ESTIMATED, 4*R): beyond ~4 rotor radii (~2 rotor diameters) the augmentation
#     is under 0.4% and treated as negligible, matching the general aerodynamics convention
#     that ground effect fades out within about 2 rotor diameters of the surface.
Z_MIN_M = PROP_RADIUS_M / 3.0
Z_MAX_M = 4.0 * PROP_RADIUS_M


def ground_effect_extra_thrust(base_thrust_n: float, height_m: float) -> float:
    """
    Extra thrust (Newtons, on top of base_thrust_n) a single rotor produces from ground
    effect at `height_m` above the surface below it. Returns 0 beyond Z_MAX_M.
    """
    if height_m >= Z_MAX_M:
        return 0.0
    z = max(height_m, Z_MIN_M)
    ratio = (PROP_RADIUS_M / (4.0 * z)) ** 2
    augmentation = 1.0 / (1.0 - ratio) - 1.0
    return base_thrust_n * augmentation


# ---- Aerodynamic drag ----
# SOURCED: the standard quadratic drag equation, F = -0.5*rho*Cd*A*v*|v| (fluid dynamics,
# not specific to rotorcraft). rho = 1.225 kg/m^3 is the ISA standard atmosphere density at
# sea level, 15 deg C.
AIR_DENSITY_KG_M3 = 1.225

# DERIVED from custom_racer.urdf's measured geometry (motor half-spans 0.094m/0.074m,
# prop radius 0.065m, body height 0.07m):
#   - horizontal (xy) reference area: the side-profile silhouette, body height times the
#     average of the front-back and left-right motor-tip-to-motor-tip spans.
#   - vertical (z) reference area: the top-down silhouette, approximated as the ellipse
#     bounded by those same two spans (pi/4 * span_x * span_y is the area of an ellipse
#     with those spans as its major/minor axes).
_MOTOR_SPAN_X_M = 2 * 0.094 + 2 * PROP_RADIUS_M  # front-back, incl. propeller diameter
_MOTOR_SPAN_Y_M = 2 * 0.074 + 2 * PROP_RADIUS_M  # left-right, incl. propeller diameter
DRAG_AREA_XY_M2 = 0.07 * (_MOTOR_SPAN_X_M + _MOTOR_SPAN_Y_M) / 2.0   # ~= 0.0209 m^2
DRAG_AREA_Z_M2 = math.pi / 4.0 * _MOTOR_SPAN_X_M * _MOTOR_SPAN_Y_M    # ~= 0.0694 m^2

# ESTIMATED (no wind-tunnel data for this airframe): drag coefficients bracketed by two
# textbook reference values -- a streamlined body (Cd ~= 0.1-0.3) and a flat plate/disc
# broadside to the flow (Cd ~= 1.28, the classic fluid-dynamics figure for a flat plate
# perpendicular to flow). Moving sideways, the frame is a relatively open lattice of arms
# and motors (closer to streamlined); moving vertically, the rotor discs and arms present
# a much more solid-looking silhouette to the airflow (closer to the flat-plate figure).
DRAG_COEFF_XY = 0.7
DRAG_COEFF_Z = 1.0


def aerodynamic_drag_force_body_frame(body_vel_xyz):
    """
    Drag force (Newtons, body-frame xyz) opposing the airframe's own motion through air,
    from body-frame velocity `body_vel_xyz` (m/s). Independent of RPM/thrust entirely --
    this acts even with the propellers stopped.
    """
    vx, vy, vz = body_vel_xyz
    fx = -0.5 * AIR_DENSITY_KG_M3 * DRAG_COEFF_XY * DRAG_AREA_XY_M2 * vx * abs(vx)
    fy = -0.5 * AIR_DENSITY_KG_M3 * DRAG_COEFF_XY * DRAG_AREA_XY_M2 * vy * abs(vy)
    fz = -0.5 * AIR_DENSITY_KG_M3 * DRAG_COEFF_Z * DRAG_AREA_Z_M2 * vz * abs(vz)
    return (fx, fy, fz)
