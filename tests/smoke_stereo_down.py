"""Render ground and a raised roof to verify downward stereo, not flat-ground ranging.

Run: python -m tests.smoke_stereo_down (requires OpenGL).
"""
from pathlib import Path
import tempfile
import cv2
import genesis as gs
import numpy as np
from swarm.compVision.stereo_depth import StereoCalibration, StereoDepth
from swarm.sensors.stereo_rig import StereoRig


def main():
    gs.init(backend=gs.cpu, logging_level='error', seed=7)
    with tempfile.TemporaryDirectory(prefix='stereo-down-') as directory:
        folder = Path(directory)
        # Coarse enough for ORB corners to survive projection on both surfaces.
        texture = np.random.default_rng(17).integers(20, 235, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(folder/'texture.png'), cv2.GaussianBlur(texture, (3, 3), .7))
        scene = gs.Scene(show_viewer=False)
        for name, xmin, xmax, ymin, ymax, z in [('ground', -12, 12, -12, 12, 0),
                                               ('roof', -4, 4, -6, 0, 4)]:
            obj = folder/f'{name}.obj'
            obj.write_text(f'v {xmin} {ymin} {z}\nv {xmax} {ymin} {z}\n'
                           f'v {xmax} {ymax} {z}\nv {xmin} {ymax} {z}\n'
                           'vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n'
                           'f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n')
            scene.add_entity(gs.morphs.Mesh(file=str(obj), fixed=True, collision=False),
                             surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ImageTexture(
                                 image_path=str(folder/'texture.png'))))
        urdf = Path(__file__).resolve().parents[1]/'genesis_drones/robots/assets/drone_urdf/drone.urdf'
        drone = scene.add_entity(gs.morphs.Drone(file=str(urdf), pos=(0, 0, 12.12)))
        c = StereoCalibration(baseline_m=.5)
        rig = StereoRig(scene, drone, c)
        scene.build()
        try:
            left, right = rig.read()
            result = StereoDepth(c).compute(left, right)
            T = rig.world_from_left_camera()
            measured_depths = []
            for world, expected in [([0, 3, 0, 1], 12.), ([0, -3, 4, 1], 8.)]:
                optical = np.linalg.inv(T) @ world
                uvw = c.K @ optical[:3]
                u, v = np.rint(uvw[:2]/uvw[2]).astype(int)
                patch = result['depth_m'][v-15:v+16, u-15:u+16]
                assert np.isfinite(patch).mean() > .5
                measured = np.nanmedian(patch)
                assert abs(measured-expected)/expected < .08, (measured, expected)
                measured_depths.append(float(measured))
            assert measured_depths[0] - measured_depths[1] > 3.
            points = result['points_m']
            world_points = np.column_stack((points, np.ones(len(points)))) @ T.T
            assert np.count_nonzero(np.abs(world_points[:, 2]) < .7) > 20
            assert np.count_nonzero(np.abs(world_points[:, 2]-4.) < .5) > 20
            print(f'PASS downward: ground depth {measured_depths[0]:.2f} m (expected 12), '
                  f'roof depth {measured_depths[1]:.2f} m (expected 8); '
                  f'{len(points)} triangulated points at two different surface heights')
        finally:
            if hasattr(scene, 'close'):
                scene.close()


if __name__ == '__main__':
    main()
