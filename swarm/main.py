from pathlib import Path

import genesis as gs
from swarm.scene_builders.customScene import CustomScene
from swarm.droneClasses.droneVisulaControler import DroneVisulaCTRL 
import numpy as np
from swarm.utilities.rtPlotter import LivePlotter


dt = 0.01  # Simulation timestep

def main():    
    gs.init(backend=gs.cpu)  
    print("\n📍 Creating scene...")
    scene = CustomScene( dt=dt, gravity=(0, 0, -9.81), show_viewer=True, camera_pos=(10, 10, 5), camera_lookat=(0, 0, 1))

    drone = scene.add_drone(
        name="drone_1",
        urdf_path=str(
            Path(__file__).resolve().parents[1]
            / "genesis_drones"
            / "robots"
            / "assets"
            / "drone_urdf"
            / "drone.urdf"
        ),
        position=(0, 0, 0),
        orientation=(0,0,0)
    )

    plotter = LivePlotter(
        plots={
            'Cam Pos': ['x', 'y', 'z', 'yaw'],
            'Act Pos':    ['x', 'y', 'z'],
            'Camera Vel': ['x', 'y', 'z'],
        },
        buffer_size=1000,
        title='IMU Live',
    )

    controller = DroneVisulaCTRL(drone_entity=drone, dt=dt, camera_fps=30)
    scene.build()

    for step in range(100000):
        sim_time = step * dt
        controller.camera_update_step(step, sim_time)
        if controller.start(step) == 1:
            controller.position_ctrl_fused(np.array([0, 0, 1, 0]))
            real_pos = controller.get_position()
            real_pos[1] = -1 * real_pos[1]
            vid = controller.get_camera_lin_v()
            plotter.push('Cam Pos', controller.cam_pos)
            plotter.push('Act Pos', real_pos)
            plotter.push('Camera Vel', vid)
            plotter.update()

        scene.step()

    print("\n✅ Simulation complete!")
    scene.close()


if __name__ == "__main__":
    main()
