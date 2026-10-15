"""
Direct, open-loop proof that motor lag and ground effect (swarm/droneClasses/droneRealism.py)
are actually changing the physics -- bypasses the position controller entirely so nothing is
compensating for (and hiding) the effect being measured.
"""

import os

import numpy as np
import genesis as gs

from swarm.scene_builders.outdoorScene import OutdoorScene
from swarm.droneClasses.droneRealism import KF, MotorLagFilter, ground_effect_extra_thrust

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
DRONE_URDF = os.path.join(REPO_ROOT, "genesis_drones/robots/assets/drone_urdf/custom_racer.urdf")
DT = 0.01
HOVER_RPM = 15146


def motor_lag_test():
    print("\n=== Motor lag: commanded RPM steps from 0 to hover_rpm instantly at t=0 ===")
    lag = MotorLagFilter(n_motors=1)
    cmd = [HOVER_RPM]
    print(f"{'t (ms)':>8s} {'commanded rpm':>15s} {'actual rpm (lagged)':>20s} {'% of step':>10s}")
    for step in range(11):
        actual = lag.step(cmd, DT) if step > 0 else [0.0]
        pct = 100.0 * actual[0] / HOVER_RPM
        print(f"{step*10:8d} {cmd[0]:15.0f} {actual[0]:20.1f} {pct:9.1f}%")
        if step == 0:
            lag.rpm = [0.0]


def ground_effect_test():
    print("\n=== Ground effect: identical hover_rpm on all 4 motors, two starting heights ===")
    gs.init(backend=gs.cpu)

    for label, start_z in (("IN ground effect (z=0.05m)", 0.05), ("OUT of ground effect (z=1.00m)", 1.00)):
        scene = OutdoorScene(dt=DT, show_viewer=False)
        drone = scene.add_drone(name="drone_1", urdf_path=DRONE_URDF, position=(0, 0, start_z))
        scene.build()

        prop_links = [drone.get_link(f"prop{i}_link") for i in range(4)]
        z0 = float(drone.get_pos()[2])
        for _ in range(100):  # 1 simulated second at constant, exact hover_rpm, no feedback
            drone.set_propellers_rpm([HOVER_RPM] * 4)
            height = float(drone.get_pos()[2])
            for link in prop_links:
                extra = ground_effect_extra_thrust(KF * HOVER_RPM**2, height)
                if extra > 0.0:
                    link.apply_external_force(force=[0.0, 0.0, extra], ref=gs.link_ref_frame.link_COM, local=True)
            scene.step()
        z1 = float(drone.get_pos()[2])
        vz = float(drone.get_vel()[2])
        print(f"{label:32s} height {z0:.3f}m -> {z1:.3f}m after 1s (vz={vz:+.3f} m/s)")
        scene.close()


if __name__ == "__main__":
    motor_lag_test()
    ground_effect_test()
