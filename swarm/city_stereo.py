"""Climb above the city and reconstruct rooftops with downward binocular cameras.

Run: python -m swarm.city_stereo
For an aerial pair: python -m swarm.city_stereo --start-airborne --headless --steps 1
The two downward cameras triangulate arbitrary scene geometry. The existing
flat-ground optical-flow estimator is not used for city depth.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import cv2
import genesis as gs
import numpy as np
from swarm.compVision.stereo_depth import StereoCalibration, StereoDepth
from swarm.scene_builders.city_scene import CityScene
from swarm.sensors.stereo_rig import StereoRig
from swarm.droneClasses.dronePositionCTRL import DronePositionCTRL

ROOT = Path(__file__).resolve().parents[1]


class AerialSurveyFlight:
    """Climb vertically, then survey along +X only after reaching altitude.

    altitude: target world Z in metres (default 40, above the 33 m skyline).
    speed: survey speed in m/s. start_airborne skips the climb for experiments.
    Reference order is [world x, world y, world z, yaw] in metres/radians.
    """
    def __init__(self, altitude=40., speed=0.5, start_airborne=False):
        self.altitude, self.speed = altitude, speed
        self.start_altitude = altitude if start_airborne else 0.6
        # 1.875 is the peak derivative of quintic smoothstep. This duration
        # bounds reference climb speed to 1.5 m/s and starts/ends at rest.
        self.climb_time = 0. if start_airborne else 1.875*(altitude-self.start_altitude)/1.5
        self.cruise_start = None
        self.phase = 'settle' if start_airborne else 'climb'

    @property
    def initial_position(self):
        """Spawn above the crosswalk, clear of buildings, at [x,y,z] metres."""
        return (8., 0., self.start_altitude)

    def setpoint(self, elapsed_s, position):
        """Return a smooth reference using time (s) and measured world XYZ (m).

        Position is used only for flight-phase gating, never stereo depth.
        Horizontal travel waits for the climb to finish AND measured height
        to reach within 1 m of the survey altitude, plus a 3 s settling period.
        """
        fraction = 1. if self.climb_time == 0 else np.clip(elapsed_s/self.climb_time, 0., 1.)
        blend = fraction**3*(10.-15.*fraction+6.*fraction**2)
        z = self.start_altitude+(self.altitude-self.start_altitude)*blend
        if self.cruise_start is None and elapsed_s >= self.climb_time:
            self.phase = 'settle'
            if abs(position[2]-self.altitude) <= 1.:
                self.cruise_start = elapsed_s+3.
        travel_time = max(0., elapsed_s-self.cruise_start) if self.cruise_start is not None else 0.
        if travel_time > 0:
            self.phase = 'survey'
        # Smoothly approach requested horizontal speed with a 2 s time constant.
        distance = self.speed*(travel_time-2.*(1.-np.exp(-travel_time/2.)))
        # Ease to a stop at x=24 rather than abruptly clipping the target.
        x = 8.+16.*np.tanh(distance/16.)
        return np.array([x, 0., z, 0.])


def save_capture(folder, left, right, result, estimator, rig, simulation_time):
    """Save the latest pair, metric reconstruction and calibration to folder.

    Depth/points use metres in the left optical frame. The JSON stores the
    optical-to-world transform and capture time in simulated seconds. Invalid
    dense pixels are NaN; black preview areas mean unknown, not free space.
    """
    folder.mkdir(parents=True,exist_ok=True)
    for name,img in [('left.png',left),('right.png',right),
                     ('preview.png',estimator.preview(left,right,result))]:
        if not cv2.imwrite(str(folder/name),img):
            raise IOError(f'Could not save {folder/name}')
    np.savez_compressed(folder/'depth.npz',**result)
    P1,P2 = estimator.calibration.projection_matrices
    metadata = dict(**asdict(estimator.calibration),K=estimator.calibration.K.tolist(),
                    P_left=P1.tolist(),P_right=P2.tolist(),distortion=[0.,0.,0.,0.,0.],
                    camera_direction=rig.direction,rectified=True,frame='left_optical_X_right_Y_down_Z_forward',
                    depth_units='metres_optical_Z',simulation_time_s=simulation_time,
                    world_from_left_camera=rig.world_from_left_camera().tolist())
    (folder/'calibration.json').write_text(json.dumps(metadata,indent=2))
    points,rows,cols = estimator.point_cloud(result['depth_m'])
    colors = left[rows,cols,::-1]  # BGR -> RGB for PLY viewers.
    with (folder/'points.ply').open('w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(points)}\n'
                'property float x\nproperty float y\nproperty float z\n'
                'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for xyz,rgb in zip(points,colors):
            f.write(' '.join(f'{v:.6f}' for v in xyz)+' '+' '.join(str(int(v)) for v in rgb)+'\n')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--headless',action='store_true',help='No windows; rendering still requires OpenGL')
    p.add_argument('--steps',type=int,default=8000,help='Number of 0.01 s physics steps')
    p.add_argument('--speed',type=float,default=0.5,help='Survey speed after climbing, m/s; 0 to hover above the city')
    p.add_argument('--baseline',type=float,default=0.5,help='Stereo camera spacing in metres')
    p.add_argument('--altitude',type=float,default=40.,help='Survey height above world z=0, metres (35 to 60)')
    p.add_argument('--start-airborne',action='store_true',help='Start at survey altitude, skipping takeoff')
    p.add_argument('--width',type=int,default=640)
    p.add_argument('--height',type=int,default=480)
    p.add_argument('--fov',type=float,default=65.,help='Vertical field of view, degrees')
    p.add_argument('--disparities',type=int,default=128,help='Disparity search in pixels, multiple of 16')
    p.add_argument('--camera-fps',type=float,default=10.,help='Depth updates per simulated second')
    p.add_argument('--output',type=Path,default=None,help='Capture directory; default outputs/city_stereo/<timestamp>')
    args = p.parse_args()
    if args.steps < 1 or not 0 <= args.speed <= 2 or not 0 < args.camera_fps <= 100:
        p.error('Require steps>=1, speed in [0,2] m/s and camera-fps in (0,100]')
    if not 35. <= args.altitude <= 60.:
        p.error('Survey altitude must be 35 to 60 m, above the skyline and within the depth range')
    return args


def main():
    args = parse_args()
    calibration = StereoCalibration(args.width,args.height,args.fov,args.baseline)
    estimator = StereoDepth(calibration,num_disparities=args.disparities)
    output = args.output or ROOT/'outputs/city_stereo'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    flight = AerialSurveyFlight(args.altitude,args.speed,args.start_airborne)
    gs.init(backend=gs.cpu,logging_level='warning',seed=7)
    scene = CityScene(show_viewer=not args.headless)
    drone = scene.add_drone('city_drone',str(ROOT/'genesis_drones/robots/assets/drone_urdf/drone.urdf'),
                            position=flight.initial_position)
    rig = StereoRig(scene.scene,drone,calibration,direction='down')
    controller = DronePositionCTRL(drone)
    scene.build()
    next_capture = 0.
    last_capture = None
    previous_phase = None
    try:
        for step in range(args.steps):
            now = step*controller.dt
            if now+1e-9 >= next_capture:
                left,right = rig.read()
                result = estimator.compute(left,right)
                last_capture = (left,right,result,now)
                # Save immediately at capture time so the exported pose belongs
                # to these images, rather than the later end-of-loop drone pose.
                save_capture(output,left,right,result,estimator,rig,now)
                if not args.headless:
                    cv2.imshow('City from above: left / right / depth / features',estimator.preview(left,right,result))
                    if cv2.waitKey(1) & 0xff in (27,ord('q')):
                        break
                while next_capture <= now+1e-9:
                    next_capture += 1./args.camera_fps
            controller.step(step)
            # Long climbs accumulate IMU position drift. This perception demo
            # uses simulator position as a GPS-like flight reference; only the
            # flight controller receives it, never the stereo reconstruction.
            controller.set_pose(controller.get_position())
            target = flight.setpoint(now,controller.get_position())
            if flight.phase != previous_phase:
                print(f'{flight.phase.capitalize()}: height {controller.get_position()[2]:.1f} m '
                      f'-> survey height {args.altitude:g} m',flush=True)
                previous_phase = flight.phase
            controller.lowLevelControl(controller.position_control(target))
            scene.step()
        if last_capture is not None:
            result = last_capture[2]
            print(f'Final drone position (m): {controller.get_position().round(3)}')
            print(f'Capture saved: {output.resolve()}')
            print(f'Valid dense pixels: {result["valid"].mean():.1%}; '
                  f'triangulated feature points: {len(result["points_m"])}')
    finally:
        cv2.destroyAllWindows()
        scene.close()


if __name__ == '__main__':
    main()
