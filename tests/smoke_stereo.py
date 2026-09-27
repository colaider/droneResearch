"""Run with python -m tests.smoke_stereo; requires Genesis and OpenGL.

Render a textured wall whose optical depth is known, checking the actual
camera mounts, image convention, intrinsics and triangulation end to end.
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
    with tempfile.TemporaryDirectory(prefix='stereo-wall-') as directory:
        folder = Path(directory)
        texture = np.random.default_rng(17).integers(20, 235, (512, 512, 3), dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 3), .7)
        cv2.imwrite(str(folder/'texture.png'), texture)
        (folder/'wall.obj').write_text(
            'v 8 -3 0\nv 8 -3 6\nv 8 3 6\nv 8 3 0\n'
            'vt 0 0\nvt 0 1\nvt 1 1\nvt 1 0\n'
            'f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n')
        scene = gs.Scene(show_viewer=False)
        scene.add_entity(gs.morphs.Mesh(file=str(folder/'wall.obj'), fixed=True, collision=False),
                         surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ImageTexture(
                             image_path=str(folder/'texture.png'))))
        urdf = Path(__file__).resolve().parents[1]/'genesis_drones/robots/assets/drone_urdf/drone.urdf'
        drone = scene.add_entity(gs.morphs.Drone(file=str(urdf), pos=(0, 0, 3)))
        c = StereoCalibration()
        rig = StereoRig(scene, drone, c, direction='forward')
        scene.build()
        try:
            left, right = rig.read()
            result = StereoDepth(c).compute(left, right)
            # Wall at world x=8; optical centre has forward offset +0.22 m.
            expected = 8.-.22
            region = result['depth_m'][140:340, 220:420]
            measured = np.nanmedian(region)
            assert np.isfinite(region).mean() > .6, np.isfinite(region).mean()
            assert abs(measured-expected)/expected < .06, (measured, expected)
            points = result['points_m']
            assert len(points) > 30, len(points)
            sparse_depth = np.median(points[:, 2])
            assert abs(sparse_depth-expected)/expected < .08, (sparse_depth, expected)
            T = rig.world_from_left_camera()
            world = np.column_stack((points, np.ones(len(points)))) @ T.T
            assert abs(np.median(world[:, 0])-8.) < .6
            print(f'PASS: rendered wall depth {measured:.3f} m, expected {expected:.3f} m; '
                  f'{len(points)} triangulated features; world-frame export checked')
        finally:
            if hasattr(scene, 'close'):
                scene.close()


if __name__ == '__main__':
    main()
