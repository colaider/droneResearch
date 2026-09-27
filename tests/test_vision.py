from types import SimpleNamespace
import numpy as np
import cv2
from swarm.compVision.visualAccEst import VisulaAcEst
from swarm.utilities.geometry import rotation
from swarm.droneClasses.droneVisulaControler import DroneVisulaCTRL


def project(est, points, pose, camera=0):
    p,att=pose
    lever=np.array([0.,(-1 if camera==0 else 1)*est.camera_saperation/2,0.])
    body=(points-p) @ rotation(att)-lever
    return est.center+np.column_stack((body[:,0],-body[:,1]))/(-body[:,2,None])*est.foc_l


def test_planar_translation_rotation_and_camera_lever_arm():
    est=VisulaAcEst((500,600),85); est.camera_saperation=.05; est.set_dt(.03)
    old=(np.array([0.,0.,1.]),np.array([.1,-.1,.4]))
    velocity=np.array([.4,-.2,.1])
    new=(old[0]+velocity*est.dt,old[1]+np.array([.01,.02,.03]))
    ground=np.array([[x,y,0.] for x in [-.2,0,.2] for y in [-.2,0,.2]])
    for cam in [0,1]:
        measured=est.estimate_velocity(project(est,ground,old,cam),project(est,ground,new,cam),old,new,cam)
        assert measured is not None
        assert np.allclose(measured[0],velocity,atol=1e-9)


def test_blank_images_and_ground_level_are_safe():
    e=VisulaAcEst((100,80),85)
    blank=np.zeros((80,100,3),np.uint8)
    for i in range(3):
        e.processing([blank,blank],i)
        assert not e.valid
        assert np.isfinite(e.camera_velocity).all()
    points=np.array([[10.,10.],[20.,10.],[30.,10.],[40.,10.]])
    pose=(np.zeros(3),np.zeros(3))
    assert e.estimate_velocity(points,points,pose,pose,0) is None


def test_lost_optical_flow(monkeypatch):
    e=VisulaAcEst((100,80),85)
    image=np.random.default_rng(2).integers(0,255,(80,100,3),dtype=np.uint8)
    monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',lambda *a,**k:(None,None,None))
    _,old,new=e.lucas_kanade_flow(image,image)
    assert old.shape==new.shape==(0,2)


def test_image_tracking_and_raw_frame_history():
    e=VisulaAcEst((200,160),85); e.drone_pos[2]=1; e.set_dt(.04)
    image=np.random.default_rng(4).integers(0,255,(160,200,3),dtype=np.uint8)
    shifted=cv2.warpAffine(image,np.float32([[1,0,-2],[0,1,0]]),(200,160))
    e.processing([image,image],0); e.processing([shifted,shifted],1)
    assert e.valid and e.camera_velocity[0]>0
    assert np.array_equal(e.buffer[-1][0],shifted)
    assert e.filters[0] is not e.filters[1]


def test_camera_schedule_and_fusion():
    processor=SimpleNamespace(valid=True,camera_velocity=np.array([1.,0.,0.]))
    intervals=[]
    drone=SimpleNamespace(frame_processor=processor,set_camera_dt=intervals.append,
                          camera_step=lambda:None,camera_show=lambda:None)
    c=DroneVisulaCTRL(drone,camera_weight=.5)
    c.step=lambda step:None
    c.get_attitude=lambda:np.zeros(3)
    c.get_ang_vel=lambda:np.zeros(3)
    c.get_position=lambda:np.array([0.,0.,1.])
    c.v_x_sum=c.v_y_sum=c.v_z_sum=0.
    c.camera_update_step(0)
    assert c.get_lin_vel()[0]==.5
    c.camera_update_step(1)
    assert c.get_lin_vel()[0]==.5  # Do not fuse a stale image twice.
    for i in range(2,11): c.camera_update_step(i)
    assert np.allclose(intervals[1:],[.04,.03,.03])
    processor.valid=False
    before=c.get_lin_vel().copy()
    for i in range(11,15): c.camera_update_step(i)
    assert np.array_equal(c.get_lin_vel(),before)
