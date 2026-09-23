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
        urdf_path=r"C:\Users\penturas\Desktop\codits\Python\instullingGenesis\GenesisDroneEnv\genesis_drones\robots\assets\drone_urdf\drone.urdf",
        position=(0, 0, 0),
        orientation=(0,0,0)
    )

    plotter = LivePlotter(
        plots={
            'Accelerometer (m/s²)': ['ax', 'ay', 'az'],
            'Gyroscope (rad/s)':    ['gx', 'gy', 'gz'],
            'Actual pos': ['x', 'y', 'z'],
        },
        buffer_size=1000,
        title='IMU Live',
    )

    controller = DroneVisulaCTRL(drone_entity=drone, dt=dt)
    scene.build()

    import numpy as np

    def stepping():
        controller.camera_update_step(step)

        
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
            z = 7.0
            yaw = 0.0

            pos = controller.get_position()
            # controller.position_control(np.array([x, y, z, yaw]))
            controller.position_ctrl_fused(np.array([x,0,1,0]))      
            pos = controller.get_position()


            acc = controller.get_lin_vel()
            vid = controller.get_camera_lin_v()
            plotter.push('Accelerometer (m/s²)', acc)
            plotter.push('Gyroscope (rad/s)', vid[:3])
            plotter.update()

        scene.step()

    print("\n✅ Simulation complete!")
    scene.close()





   
if __name__ == "__main__":
    main()







   
    # for step in range(100000000000):
        
    #     if controller.start(step) == 1:
    #         

    #     scene.step()