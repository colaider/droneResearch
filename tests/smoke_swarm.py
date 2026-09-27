"""Explicit camera/IMU smoke check; requires an OpenGL-capable session."""
from pathlib import Path
import genesis as gs
import numpy as np
from swarm.scene_builders.customScene import CustomScene
from swarm.droneClasses.droneVisulaControler import DroneVisulaCTRL


def main():
    gs.init(backend=gs.cpu,logging_level='error')
    scene=CustomScene(show_viewer=False)
    path=Path(__file__).resolve().parents[1]/'genesis_drones/robots/assets/drone_urdf/drone.urdf'
    drone=scene.add_drone('test',str(path),position=(0,0,1))
    controller=DroneVisulaCTRL(drone)
    drone.camera_show=lambda:None
    scene.build()
    for step in range(12):
        controller.camera_update_step(step)
        controller.position_ctrl_fused(np.array([0.,0.,1.,0.]))
        scene.step()
        assert np.isfinite(controller.get_lin_vel()).all()
        assert np.isfinite(controller.get_imu_pos()).all()
    assert controller.stp_cam_count==4
    assert controller.drone.frame_processor.valid
    print('PASS: 12 sensor/control steps, four stereo-image reads, valid planar velocity')
    scene.close()

if __name__=='__main__': main()
