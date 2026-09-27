"""A small street assembled from the bundled CC0 Kenney city assets.

Meshes are authored Y-up. Genesis is Z-up: roll 90 degrees converts them.
The 8x unit scale makes each road tile 8 m square. Buildings have detailed
visual meshes and conservative invisible box colliders; this is a depth
perception sandbox, not an autonomous obstacle-avoidance implementation.
"""
from pathlib import Path
import numpy as np
import genesis as gs
from swarm.scene_builders.customScene import CustomScene
from swarm.utilities.geometry import rotation

ASSETS = Path(__file__).resolve().parents[1] / 'objects/city'


class CityScene(CustomScene):
    def __init__(self, dt=0.01, show_viewer=True):
        super().__init__(dt=dt, show_viewer=show_viewer,
                         camera_pos=(8, -65, 75), camera_lookat=(12, 0, 8))
        self.add_city()

    def add_ground_plane(self, size=120.):
        """Add a level grey ground at world z=0; size is metres."""
        plane = self.scene.add_entity(gs.morphs.Plane(plane_size=(size,size)),
                                      surface=gs.surfaces.Rough(color=(0.33,0.36,0.32)))
        self.entities['ground_plane'] = plane
        self.static_objects.append(plane)
        return plane

    def add_asset(self, pack, name, position, scale=8., yaw=0., collider=False):
        """Place a bundled model and optionally a conservative box collider.

        pack: 'commercial' or 'roads'; name: OBJ filename stem. position: world
        [x,y,z] metres. scale: metres per asset unit; yaw: degrees about world z.
        collider boxes encompass the visual mesh, including protrusions.
        """
        path = ASSETS/pack/(name+'.obj')
        if not path.is_file():
            raise FileNotFoundError(f'Missing bundled city asset: {path}')
        entity = self.scene.add_entity(
            gs.morphs.Mesh(file=str(path), pos=position, euler=(90.,0.,yaw),
                           scale=scale, fixed=True, collision=False),
            surface=gs.surfaces.Rough())
        self.static_objects.append(entity)
        if collider:
            # Compute the actual rotated bounds rather than assuming each model
            # is centred or has identical dimensions. Only local OBJ data is read.
            vertices = np.array([[float(x) for x in line.split()[1:4]]
                                 for line in path.read_text().splitlines() if line.startswith('v ')])
            points = vertices*scale @ rotation(np.deg2rad([90.,0.,yaw])).T
            lo,hi = points.min(axis=0),points.max(axis=0)
            self.scene.add_entity(gs.morphs.Box(size=tuple(hi-lo),
                pos=tuple(np.asarray(position)+(hi+lo)/2), fixed=True, visualization=False))
        return entity

    def add_city(self):
        """Create a clear x-directed street, building rows and roadside details."""
        # Keep the flight corridor |y|<3 clear; the drone climbs at x=8, then surveys toward x=24.
        for x in range(-16, 41, 8):
            kind = {8: 'road-crossing', 16: 'road-crossroad'}.get(x, 'road-straight')
            self.add_asset('roads',kind,(x,0,0.01),yaw=90.)
        for y in (-16,-8,8,16):
            self.add_asset('roads','road-straight',(16,y,0.01))
        models = ['building-a','building-b','building-c','building-d','building-e','building-f']
        for side in (-1,1):
            for i,x in enumerate((-8,4,28,40)):
                name = models[(i+(0 if side<0 else 2)) % len(models)]
                self.add_asset('commercial',name,(x,side*9.,0.16),yaw=0 if side>0 else 180,
                               collider=True)
            for x in (-4,20,36):
                self.add_asset('roads','light-square',(x,side*3.7,0.16),yaw=0 if side>0 else 180)
        for x,y,name in [(4,22,'building-skyscraper-a'),(28,-22,'building-skyscraper-c'),
                         (40,22,'building-skyscraper-c')]:
            self.add_asset('commercial',name,(x,y,0.),collider=True)
        self.add_asset('roads','road-sign-stop',(12,-3.7,0.16),yaw=90)
