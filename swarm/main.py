from pathlib import Path
import genesis as gs
from swarm.scene_builders.customScene import CustomScene
from swarm.droneClasses.droneVisulaControler import DroneVisulaCTRL 
import numpy as np
from swarm.utilities.rtPlotter import LivePlotter


dt = 0.01  # Simulation timestep

def flight_setpoint(elapsed_time):
    """Return the original demo's [x, y, z, yaw] position reference.

    elapsed_time is seconds since the two-second startup ended. Positions are
    world metres and yaw is radians. The 8 m cosine has angular frequency
    0.5 rad/s; its phase puts the first forward peak three seconds after startup
    (five seconds after simulation starts), matching the original demo.
    This commands forward/backward travel along x, not a circular flight path.
    """
    x = 8.0 * np.cos(0.5 * (elapsed_time - 3.0))
    return np.array([x, 0., 1., 0.])


def main():    
    gs.init(backend=gs.cpu)  
    print("\n📍 Creating scene...")
    scene = CustomScene( dt=dt, gravity=(0, 0, -9.81), show_viewer=True, camera_pos=(10, 10, 5), camera_lookat=(0, 0, 1))

    drone = scene.add_drone(
        name="drone_1",
        urdf_path=str(Path(__file__).resolve().parents[1] / "genesis_drones/robots/assets/drone_urdf/drone.urdf"),
        position=(0, 0, 0),
        orientation=(0,0,0)
    )

    plotter = LivePlotter(
        plots={
            'Cam Pos': ['x', 'y', 'z'],
            'Act Pos':    ['x', 'y', 'z'],
            # 'Used Pos': ['x', 'y', 'z'],
            'Camera Vel': ['x', 'y', 'z'],
        },
        buffer_size=1000,
        title='IMU Live',
    )

    controller = DroneVisulaCTRL(drone_entity=drone, dt=dt)
    scene.build()

    for step  in range(100000):
        controller.camera_update_step(step)
        if controller.start(step) == 1:
            t = step * controller.dt - 2.0   # time since startup finished
            controller.position_ctrl_fused(flight_setpoint(t))
            real_pos = controller.get_position()
            vid = controller.get_camera_lin_v()
            plotter.push('Cam Pos', controller.cam_pos[:3])
            plotter.push('Act Pos', real_pos)
            # plotter.push('Used Pos', pos[:3])
            plotter.push('Camera Vel', vid)
            plotter.update()

        scene.step()

    print("\n✅ Simulation complete!")
    plotter.close()
    scene.close()





   
if __name__ == "__main__":
    main()
