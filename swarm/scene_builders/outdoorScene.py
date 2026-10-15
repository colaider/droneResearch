import os
from typing import List, Tuple

import genesis as gs
import numpy as np

from swarm.droneClasses.droneStruct import DroneStruct
from swarm.droneClasses.mpu6000Imu import compute_mpu6000_imu_noise
from swarm.scene_builders.customScene import CustomScene

OBJECTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "objects")


class OutdoorScene(CustomScene):
    """
    CustomScene extended with an outdoor obstacle course (trees, buildings) for
    avoidance testing. Adds new capability only; does not modify CustomScene.
    """

    def add_ground_plane(self, size: float = 50.0):
        """
        Overrides CustomScene.add_ground_plane (called from CustomScene.__init__, so this
        version runs instead via normal Python method resolution). The base version's grass
        texture path uses Windows backslashes, which never resolve on Linux.
        """
        plane = self.scene.add_entity(
            morph=gs.morphs.Plane(
                plane_size=(size, size),
                tile_size=(5.0, 5.0),
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ImageTexture(
                    image_path=os.path.join(OBJECTS_DIR, "grass.jpg")
                )
            ),
        )
        self.entities['ground_plane'] = plane
        self.static_objects.append(plane)
        return plane

    def add_drone(self, name: str, urdf_path: str, position: Tuple[float, float, float] = (0, 0, 1), orientation=(0, 0, 0)) -> DroneStruct:
        """
        Overrides CustomScene.add_drone to use physically-derived MPU-6000 IMU noise
        (see swarm/droneClasses/mpu6000Imu.py) instead of the base version's generic
        "mid-tier consumer IMU" placeholder. Camera sensor setup is otherwise identical
        to CustomScene.add_drone, duplicated here rather than modifying that file.
        """
        drone = self.scene.add_entity(gs.morphs.Drone(file=urdf_path, pos=position, euler=orientation))

        imu = self.scene.add_sensor(
            gs.sensors.IMU(
                entity_idx=drone.idx,
                link_idx_local=0,
                draw_debug=False,
                **compute_mpu6000_imu_noise(),
            )
        )

        res = (500, 600)
        fov = 85

        left_cam = self.scene.add_sensor(
            gs.sensors.RasterizerCameraOptions(
                entity_idx=drone.idx,
                res=res,
                pos=(0.0, -0.01, 0.0),        # relative to the link frame once attached
                up=(0, 0, 0),
                fov=fov,
            )
        )
        right_cam = self.scene.add_sensor(
            gs.sensors.RasterizerCameraOptions(
                entity_idx=drone.idx,
                link_idx_local=0,
                res=res,
                pos=(0, 0.01, 0.0),        # relative to the link frame once attached
                up=(0, 0, 0),
                fov=fov,
            )
        )

        self.entities[name] = drone
        self.drones.append(drone)
        print(f"✓ Added drone: {name} at {position} (MPU-6000 IMU noise model)")
        return DroneStruct(drone, imu, left_cam, right_cam, res, fov)

    def add_box(self, name: str, position: Tuple[float, float, float], size: Tuple[float, float, float] = (2.0, 2.0, 4.0), color: Tuple[float, float, float] = (0.55, 0.55, 0.6)):
        """Add a fixed box obstacle (used as a building)."""
        box = self.scene.add_entity(
            morph=gs.morphs.Box(pos=position, size=size, fixed=True),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=color)),
        )
        self.entities[name] = box
        self.static_objects.append(box)
        return box

    def add_tree(self, name: str, position: Tuple[float, float, float], trunk_height: float = 3.0, trunk_radius: float = 0.15, canopy_radius: float = 1.0):
        """Add a fixed tree obstacle: a cylinder trunk topped by a spherical canopy."""
        x, y, z = position
        trunk = self.scene.add_entity(
            morph=gs.morphs.Cylinder(pos=(x, y, z + trunk_height / 2), height=trunk_height, radius=trunk_radius, fixed=True),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.40, 0.26, 0.13))),
        )
        canopy = self.scene.add_entity(
            morph=gs.morphs.Sphere(pos=(x, y, z + trunk_height + canopy_radius * 0.6), radius=canopy_radius, fixed=True),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.13, 0.45, 0.15))),
        )
        self.entities[f"{name}_trunk"] = trunk
        self.entities[f"{name}_canopy"] = canopy
        self.static_objects += [trunk, canopy]
        return trunk, canopy

    def populate_outdoor_obstacles(self, num_trees: int = 12, num_buildings: int = 4, area: Tuple[float, float] = (20.0, 20.0), clear_radius: float = 3.0, min_separation: float = 2.5, seed: int = 0):
        """
        Scatter trees and buildings across a rectangular area, leaving a circular
        clear zone (radius `clear_radius`) around the origin free for takeoff.
        """
        rng = np.random.default_rng(seed)
        placed: List[Tuple[float, float]] = []

        def sample_position() -> Tuple[float, float]:
            for _ in range(200):
                x = rng.uniform(-area[0] / 2, area[0] / 2)
                y = rng.uniform(-area[1] / 2, area[1] / 2)
                if np.hypot(x, y) < clear_radius:
                    continue
                if all(np.hypot(x - px, y - py) > min_separation for px, py in placed):
                    placed.append((x, y))
                    return x, y
            placed.append((x, y))
            return x, y

        for i in range(num_buildings):
            x, y = sample_position()
            w, d, h = rng.uniform(2.0, 4.0), rng.uniform(2.0, 4.0), rng.uniform(3.0, 8.0)
            self.add_box(f"building_{i}", position=(x, y, h / 2), size=(w, d, h))

        for i in range(num_trees):
            x, y = sample_position()
            trunk_height = rng.uniform(2.5, 5.0)
            canopy_radius = rng.uniform(0.7, 1.3)
            self.add_tree(f"tree_{i}", position=(x, y, 0.0), trunk_height=trunk_height, canopy_radius=canopy_radius)

        print(f"✓ Populated {num_buildings} buildings and {num_trees} trees over a {area[0]}x{area[1]}m area")
