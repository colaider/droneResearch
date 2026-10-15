from pathlib import Path
import atexit

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
            'Cam Pos':    ['raw x', 'raw y', 'raw z', 'raw yaw'],
            'Act Pos':    ['x', 'y', 'z'],
            'Camera Vel': ['x', 'y', 'z'],
        },
        buffer_size=1000,
        title='IMU Live',
    )
    depth_plotter = LivePlotter(
        plots={
            'Depth (m)':     ['stereo median'],
            'Valid matches': ['fraction'],
        },
        buffer_size=1000,
        title='Stereo Depth',
    )
    map_viewer = MapViewer(title='Stereo point map — simulation pose')

    sm = StateMachine(drone_entity=drone, dt=dt, camera_fps=30)
    atexit.register(lambda: drone.frame_processor.save_velocity_log('camera_velocity_log.csv'))
    scene.build()

    for step in range(100000):
        sim_time = step * dt

        sm.steping(step, sim_time)
        map_viewer.update(drone.point_map, drone.map_pose_position)

        if sm.stage >= 1:
            ctrl = sm.controller

            real_pos = ctrl.get_position()
            real_pos[1] = -1 * real_pos[1]
            vid = ctrl.get_camera_lin_v()

            plotter.push('Cam Pos',    ctrl.raw_cam_pos)
            plotter.push('Act Pos',    real_pos)
            plotter.push('Camera Vel', vid)
            plotter.update()

            fp = ctrl.drone.frame_processor
            # Optical-axis depth varies across roofs/ground; it is not drone altitude.
            depth_plotter.push('Depth (m)',     fp.depth_median)
            depth_plotter.push('Valid matches', fp.depth_valid_frac)
            depth_plotter.update()

        scene.step()

    print("\n✅ Simulation complete!")
    map_viewer.close()
    scene.close()


if __name__ == "__main__":
    main()