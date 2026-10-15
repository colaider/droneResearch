from pathlib import Path

import genesis as gs

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
        },
        buffer_size=1000,
        title='IMU Live',
    )
    camera_velocity_plotter = LivePlotter(
        plots={'Camera Velocity': ['vx', 'vy', 'vz', 'yaw_rate']},
        buffer_size=1000,
        title='Visual Estimator Camera Velocity',
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
    scene.build()
    last_camera_frame_idx = None

    for step in range(100000):
        sim_time = step * dt

        sm.steping(step, sim_time)
        map_viewer.update(drone.point_map, drone.map_pose_position)

        if sm.stage >= 1:
            ctrl = sm.controller

            real_pos = ctrl.get_position()
            real_pos[1] = -1 * real_pos[1]
            fp = ctrl.drone.frame_processor

            plotter.push('Cam Pos',    ctrl.raw_cam_pos)
            plotter.push('Act Pos',    real_pos)
            plotter.update()
            if fp.current_frame.idx != last_camera_frame_idx:
                camera_velocity_plotter.push(
                    'Camera Velocity', fp.camera_velocity
                )
                camera_velocity_plotter.update()
                last_camera_frame_idx = fp.current_frame.idx

            # Optical-axis depth varies across roofs/ground; it is not drone altitude.
            depth_plotter.push('Depth (m)',     fp.depth_median)
            depth_plotter.push('Valid matches', fp.depth_valid_frac)
            depth_plotter.update()

        scene.step()

    print("\n✅ Simulation complete!")
    camera_velocity_plotter.close()
    map_viewer.close()
    scene.close()


if __name__ == "__main__":
    main()