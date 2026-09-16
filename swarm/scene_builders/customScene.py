import genesis as gs
from pathlib import Path
from typing import List, Tuple, Optional
from swarm.droneClasses.droneStruct import DroneStruct


class CustomScene:
    
    def __init__(self, dt: float = 0.01, gravity: Tuple[float, float, float] = (0, 0, -9.81), show_viewer: bool = True, camera_pos: Tuple[float, float, float] = (5, 5, 3), camera_lookat: Tuple[float, float, float] = (0, 0, 0)):
   
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions( dt=dt, gravity=gravity),
            viewer_options=gs.options.ViewerOptions(camera_pos=camera_pos, camera_lookat=camera_lookat, camera_fov=40),
            show_viewer=show_viewer,
        )
        
        self.entities = {}  # Store references to added entities
        self.static_objects = []
        self.robots = []
        self.drones = []
        
        # Add ground plane by default
        self.add_ground_plane()

    
    def add_ground_plane(self, size: float = 50.0):
        """Add a ground plane to the scene."""
        plane = self.scene.add_entity(gs.morphs.Plane())
        self.entities['ground_plane'] = plane
        self.static_objects.append(plane)
        return plane

    
    def add_static_object( self, name: str, mesh_path: str, position: Tuple[float, float, float] = (0, 0, 0), scale: float = 1.0):
     
        mesh = gs.morphs.Mesh( file=mesh_path, scale=(scale, scale, scale), pos=position)
        entity = self.scene.add_entity(mesh)       
        self.entities[name] = entity
        self.static_objects.append(entity)
           
        print(f"✓ Added static object: {name} at {position}")
        return entity

    
    def add_mjcf_robot( self, name: str, mjcf_path: str, position: Tuple[float, float, float] = (0, 0, 0)):
        robot = self.scene.add_entity( gs.morphs.MJCF(file=mjcf_path))
        robot.set_pos(gs.np.array(position))
        self.entities[name] = robot
        self.robots.append(robot)    
        print(f"✓ Added MJCF robot: {name} at {position}")
        return robot

    
    def add_urdf_robot(self, name: str, urdf_path: str, position: Tuple[float, float, float] = (0, 0, 0)):
    
        robot = self.scene.add_entity( gs.morphs.URDF(file=urdf_path))
        robot.set_pos(gs.np.array(position))
        
        self.entities[name] = robot
        self.robots.append(robot)
        
        print(f"✓ Added URDF robot: {name} at {position}")
        return robot

    
    def add_drone( self, name: str, urdf_path: str, position: Tuple[float, float, float] = (0, 0, 1), orientation = (0,0,0)) -> DroneStruct:
        drone = self.scene.add_entity(gs.morphs.Drone(file=urdf_path, pos=position, euler=orientation))
        imu = self.scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=drone.idx,
            link_idx_local=0,
            # Sensor imperfections — model a mid-tier consumer IMU
            acc_noise=(0.05, 0.05, 0.05),          # m/s² white noise std
            gyro_noise=(0.005, 0.005, 0.005),      # rad/s white noise std
            acc_bias=(0.02, 0.02, 0.02),           # m/s² constant offset
            gyro_bias=(0.005, 0.005, 0.005),       # rad/s constant offset
            acc_random_walk=(0.001, 0.001, 0.001), # m/s² slow drift
            gyro_random_walk=(0.0005, 0.0005, 0.0005),  # rad/s slow drift
            draw_debug=False,
            )
        )

        left_cam = self.scene.add_sensor(
            gs.sensors.RasterizerCameraOptions(
                entity_idx=drone.idx,
                res=(500, 600),
                pos=(0.0, -0.01, 0.0),        # relative to the link frame once attached
                up=(0, 0, 0),
                fov=70.0,
            )
        )

        right_cam = self.scene.add_sensor(
            gs.sensors.RasterizerCameraOptions(
                entity_idx=drone.idx,
                link_idx_local=0,
                res=(500, 600),
                pos=(0, 0.01, 0.0),        # relative to the link frame once attached
                up=(0, 0, 0),
                fov=70.0,
            )
        )

        
        self.entities[name] = drone
        self.drones.append(drone)      
        print(f"✓ Added drone: {name} at {position}")
        return DroneStruct(drone, imu, left_cam, right_cam)

    
    def add_swarm(self): pass

        
    def build(self):
        """Build the scene (finalize physics setup)."""
        print("Building scene...")
        self.scene.build()
        print("✓ Scene built successfully")

    
    def step(self):
        """Step the simulation by one timestep."""
        self.scene.step()

    
    def run(self, num_steps: int = 1000):
        """Run the simulation."""
        print(f"Running simulation for {num_steps} steps...")
        for i in range(num_steps):
            self.scene.step()
            if (i + 1) % 100 == 0:
                print(f"  Step {i + 1}/{num_steps}")

    
    def get_entity(self, name: str):
        """Get an entity by name."""
        return self.entities.get(name)

    
    def close(self):
        """Close the scene."""
        if hasattr(self.scene, 'close'):
            self.scene.close()