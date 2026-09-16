import genesis as gs
from swarm.scene_builders.customScene import CustomScene
from swarm.droneClasses.dronePositionCTRL import DronePositionCTRL
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
        },
        buffer_size=1000,
        title='IMU Live',
    )

    controller = DronePositionCTRL(drone_entity=drone, dt=dt)
    scene.build()

    import numpy as np


    for step  in range(100000):
        if controller.start(step) == 1:
            controller.step()
            controller.position_control(np.array([0,0,1,0]))
            reading = controller.drone.imu.read()
            acc  = reading.lin_acc.numpy()   # array shape (3,)
            gyro = reading.ang_vel.numpy()   # array shape (3,)
            mag  = reading.mag.numpy() 

            plotter.push('Accelerometer (m/s²)', acc)
            plotter.push('Gyroscope (rad/s)',    gyro)
            plotter.update()
            controller.drone.cameraShow()
        scene.step()


    # for step in range(100000000000):
    #     controller.step()
    #     if controller.start(step) == 1:
    #         t = (step - 500) * controller.dt   # time since startup finished
    #         radius = 2.0
    #         omega = 0.5   # rad/s → circle period = 2π/0.5 ≈ 12.5s
            
    #         x = radius * np.cos(omega * t)
    #         y = radius * np.sin(omega * t)
    #         z = 1.0
    #         yaw = 0.0
            
    #         controller.position_control(np.array([x, y, z, yaw]))
    #     scene.step()
        
    print("\n✅ Simulation complete!")
    scene.close()

if __name__ == "__main__":
    main()