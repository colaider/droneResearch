"""
Statistically verify the MPU-6000 IMU noise model is actually active: spawn the drone,
let it settle motionless on the ground (props off), then collect many IMU samples and
compare the measured standard deviation against the theoretical values from
swarm/droneClasses/mpu6000Imu.py. If the noise model were NOT active, every sample would
be numerically identical (measured std == 0.0).
"""

import os

import numpy as np
import genesis as gs

from swarm.scene_builders.outdoorScene import OutdoorScene
from swarm.droneClasses.mpu6000Imu import compute_mpu6000_imu_noise

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
DRONE_URDF = os.path.join(REPO_ROOT, "genesis_drones/robots/assets/drone_urdf/custom_racer.urdf")

SETTLE_STEPS = 100
SAMPLE_STEPS = 2000


def main():
    gs.init(backend=gs.cpu)

    scene = OutdoorScene(dt=0.01, show_viewer=False)
    drone = scene.add_drone(name="drone_1", urdf_path=DRONE_URDF, position=(0, 0, 0.05))
    scene.build()

    for _ in range(SETTLE_STEPS):
        drone.set_propellers_rpm([0.0, 0.0, 0.0, 0.0])
        scene.step()

    acc_samples = []
    gyro_samples = []
    for _ in range(SAMPLE_STEPS):
        drone.set_propellers_rpm([0.0, 0.0, 0.0, 0.0])
        scene.step()
        reading = drone.imu.read()
        acc_samples.append(reading.lin_acc.numpy())
        gyro_samples.append(reading.ang_vel.numpy())

    acc_samples = np.array(acc_samples)
    gyro_samples = np.array(gyro_samples)
    expected = compute_mpu6000_imu_noise()

    print(f"\nCollected {SAMPLE_STEPS} IMU samples with the drone at rest, props off.\n")
    print(f"{'':16s} {'measured std':>14s} {'expected white-noise std':>26s}")
    for axis, label in enumerate(("x", "y", "z")):
        print(f"accel {label:>2s} (m/s^2)  {acc_samples[:, axis].std():14.5f} {expected['acc_noise'][axis]:26.5f}")
    for axis, label in enumerate(("x", "y", "z")):
        print(f"gyro  {label:>2s} (rad/s)  {gyro_samples[:, axis].std():14.6f} {expected['gyro_noise'][axis]:26.6f}")

    print(
        "\nMeasured std should land close to the expected white-noise std (it will run a "
        "bit higher since bias/random-walk drift also contribute to the sample spread over "
        f"{SAMPLE_STEPS} steps). If measured std were 0.0, the noise model would not be active."
    )


if __name__ == "__main__":
    main()
