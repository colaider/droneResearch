from pathlib import Path

import genesis as gs
import numpy as np

from swarm.scene_builders.customScene import CustomScene
from swarm.droneClasses.stateMachine import StateMachine
from swarm.utilities.rtPlotter import LivePlotter
from swarm.utilities.mapViewer import MapViewer


dt = 0.01


def main():
    gs.init(backend=gs.cpu)
    print("\n📍 Creating scene...")
    scene = CustomScene(
        dt=dt, gravity=(0, 0, -9.81), show_viewer=True,
        camera_pos=(10, 10, 5), camera_lookat=(0, 0, 1),
    )

    drone = scene.add_drone(
        name="drone_1",
        urdf_path=str(
            Path(__file__).resolve().parents[1]
            / "genesis_drones" / "robots" / "assets" / "drone_urdf" / "drone.urdf"
        ),
        position=(0, 0, 0),
        orientation=(0, 0, 0),
    )

    plotter = LivePlotter(
        plots={
            'Cam Pos':    ['x', 'y', 'z', 'yaw'],
            'Act Pos':    ['x', 'y', 'z'],
            'Camera Vel': ['x', 'y', 'z'],
        },
        buffer_size=1000,
        title='IMU Live',
    )
    depth_plotter = LivePlotter(
        plots={
            'Depth (m)':     ['stereo', 'true'],
            'Error (m)':     ['stereo - true'],
            'Valid matches': ['fraction'],
        },
        buffer_size=1000,
        title='Stereo Depth',
    )
    map_viewer = MapViewer(title='Terrain Map')

    sm = StateMachine(drone_entity=drone, dt=dt, camera_fps=30)
    scene.build()

    for step in range(100000):
        sim_time = step * dt

        sm.steping(step, sim_time)

        if sm.stage >= 1:
            ctrl = sm.controller

            real_pos = ctrl.get_position()
            real_pos[1] = -1 * real_pos[1]
            vid = ctrl.get_camera_lin_v()

            plotter.push('Cam Pos',    ctrl.cam_pos)
            plotter.push('Act Pos',    real_pos)
            plotter.push('Camera Vel', vid)
            plotter.update()

            fp = ctrl.drone.frame_processor
            true_z = float(ctrl.get_position()[2])
            depth_plotter.push('Depth (m)',     [fp.depth_median, true_z])
            depth_plotter.push('Error (m)',     fp.depth_median - true_z)
            depth_plotter.push('Valid matches', fp.depth_valid_frac)
            depth_plotter.update()

        scene.step()

    print("\n✅ Simulation complete!")
    scene.close()


if __name__ == "__main__":
    main()