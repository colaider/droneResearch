"""Run explicitly with python -m tests.smoke_forward; needs OpenGL rendering.

Unlike a hover-only check, this exercises the actual demo reference after
starting on the ground, including IMU estimation and camera velocity fusion.
"""
from pathlib import Path
import genesis as gs
import numpy as np
from swarm.main import flight_setpoint
from swarm.scene_builders.customScene import CustomScene
from swarm.droneClasses.droneVisulaControler import DroneVisulaCTRL


def main():
    gs.init(backend=gs.cpu, logging_level='error', seed=7)
    scene = CustomScene(show_viewer=False)
    path = Path(__file__).resolve().parents[1] / 'genesis_drones/robots/assets/drone_urdf/drone.urdf'
    drone = scene.add_drone('forward_test', str(path), position=(0, 0, 0))
    controller = DroneVisulaCTRL(drone)
    drone.camera_show = lambda: None
    scene.build()
    positions = []
    visual_samples = 0
    try:
        for step in range(800):  # Eight simulated seconds, including takeoff.
            controller.camera_update_step(step)
            visual_samples += int(controller.camera_fresh and drone.frame_processor.valid)
            if controller.start(step) == 1:
                controller.position_ctrl_fused(flight_setpoint(step * controller.dt - 2.0))
            scene.step()
            position = controller.get_position()
            assert np.isfinite(position).all()
            assert np.isfinite(controller.get_imu_pos()).all()
            if step >= 300:
                assert 0.4 < position[2] < 1.6, position
            positions.append(position)
        positions = np.asarray(positions)
        # Detect the regression where the whole flight was confined below 1 m.
        assert positions[:, 0].max() > 2.0, positions[:, 0].max()
        assert visual_samples > 10, visual_samples
        print(f'PASS: forward travel {positions[:, 0].max():.2f} m; '
              f'{visual_samples} valid camera samples; altitude maintained')
    finally:
        scene.close()


if __name__ == '__main__':
    main()
