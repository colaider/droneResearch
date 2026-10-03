from pathlib import Path

import genesis as gs
from swarm.scene_builders.customScene import CustomScene
from swarm.droneClasses.droneVisulaControler import DroneVisulaCTRL 
import numpy as np
from swarm.utilities.rtPlotter import LivePlotter
from swarm.utilities.mapViewer import MapViewer
from swarm.compVision.terrainMap import TerrainMap


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
    depth_plotter = LivePlotter(
        plots={
            'Depth (m)': ['stereo', 'true'],
            'Error (m)': ['stereo - true'],
            'Valid matches': ['fraction'],
        },
        buffer_size=1000,
        title='Stereo Depth',
    )
    terrain = TerrainMap(drone.cam_cfg, size=30.0, cell=0.05)
    map_viewer = MapViewer(title='Terrain Map')

    controller = DroneVisulaCTRL(drone_entity=drone, dt=dt)
    scene.build()

    def stepping():
        controller.camera_update_step(step)

        # true pose for now; swap in the estimated pose once the map looks right
        fp = controller.drone.frame_processor
        pos = np.asarray(controller.get_position(), dtype=float)
        quat = np.asarray(controller.drone.get_quat(), dtype=float)
        terrain.add(fp.depth_points, pos, quat, fp.depth_frame_idx)
        map_viewer.update(terrain, pos)

        
    def plotting():
        reading = controller.drone.imu.read()
        acc  = reading.lin_acc.numpy()   # array shape (3,)
        gyro = reading.ang_vel.numpy()   # array shape (3,)
        
        pos= controller.get_position()
        plotter.push('Accelerometer (m/s²)', acc)
        plotter.push('Gyroscope (rad/s)',    gyro)
        plotter.push('Actual pos', pos)
        plotter.update()
    prev_p = np.zeros([3])

    for step  in range(100000):
        stepping()
        x = 0
        if controller.start(step) == 1:
            t = (step - 500) * controller.dt   # time since startup finished
            radius = 8.0
            omega = 0.5  
            x = radius * np.cos(omega * t)
            y = radius * np.sin(omega * t)
            z = 1.0
            yaw = 0.0

            pos = controller.get_position()
            # controller.position_control(np.array([x, y, z, yaw]))
            controller.position_ctrl_fused(np.array([0,0,1,0]))      
            pos = controller.get_imu_pos()
            real_pos = controller.get_position()
            real_pos[1] = -1*real_pos[1]
            acc = controller.get_lin_vel()
            vid = controller.get_camera_lin_v()
            plotter.push('Cam Pos', controller.cam_pos)
            plotter.push('Act Pos', real_pos)
            # plotter.push('Used Pos', acc[:3])
            plotter.push('Camera Vel', vid)
            plotter.update()

            fp = controller.drone.frame_processor
            true_z = float(controller.get_position()[2])
            depth_plotter.push('Depth (m)', [fp.depth_median, true_z])
            depth_plotter.push('Error (m)', fp.depth_median - true_z)
            depth_plotter.push('Valid matches', fp.depth_valid_frac)
            depth_plotter.update()

        scene.step()

    print("\n✅ Simulation complete!")
    scene.close()





   
if __name__ == "__main__":
    main()







   
    # for step in range(100000000000):
        
    #     if controller.start(step) == 1:
    #         

    #     scene.step()