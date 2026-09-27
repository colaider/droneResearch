from pathlib import Path
from types import SimpleNamespace
import torch
import yaml
from genesis_drones.envs.genesis_env import Genesis_env
from genesis_drones.sensors.odom import Odom
from genesis_drones.controllers.pid_controller import PIDcontroller
from genesis_drones.tasks.track_task import Track_task

ROOT=Path(__file__).resolve().parents[1]

class Drone:
    def __init__(self,n):
        self.pos=torch.zeros(n,3); self.pos[:,2]=1
        self.quat=torch.tensor([[1.,0.,0.,0.]]).repeat(n,1)
    def get_pos(self): return self.pos.clone()
    def get_quat(self): return self.quat.clone()
    def get_vel(self): return torch.zeros_like(self.pos)
    def get_ang(self): return torch.zeros_like(self.pos)
    def set_pos(self,p,envs_idx,zero_velocity): self.pos[envs_idx]=p
    def set_quat(self,q,envs_idx,zero_velocity): self.quat[envs_idx]=q
    def set_propellers_rpm(self,rpm): self.rpm=rpm.clone()


def env():
    e=Genesis_env.__new__(Genesis_env)
    e.device=torch.device('cpu'); e.num_envs=2; e.render_cam=False; e.target=None
    e.env_config=dict(fixed_init_pos=True,drone_init_pos=[0.,0.,1.],dt=.01,num_envs=2)
    e.drone=Drone(2)
    e.drone.odom=Odom(2,device=e.device)
    e.drone.odom.set_drone(e.drone)
    cfg=yaml.safe_load((ROOT/'config/track_rl/flight.yaml').read_text())
    e.drone.controller=PIDcontroller(2,torch.zeros(7),e.drone.odom,cfg,device=e.device)
    e.drone.controller.set_drone(e.drone)
    e.scene=SimpleNamespace(step=lambda:None)
    e.reset()
    return e


def test_reset_none_and_partial_do_not_advance_physics():
    e=env()
    def forbidden(): raise AssertionError('reset advanced physics')
    e.scene.step=forbidden
    e.drone.pos[1]=torch.tensor([2.,3.,4.])
    e.drone.odom.world_pos[1]=e.drone.pos[1]
    e.drone.odom.last_world_pos[1]=torch.tensor([1.,2.,3.])
    before=e.drone.odom.last_world_pos[1].clone()
    e.drone.controller.throttle_command[:]=.2
    e.reset(torch.tensor([0]))
    assert torch.equal(e.drone.pos[1],torch.tensor([2.,3.,4.]))
    assert torch.equal(e.drone.odom.last_world_pos[1],before)
    assert e.drone.controller.throttle_command[1]==.2
    assert e.drone.controller.throttle_command[0]==0
    assert torch.equal(e.drone.odom.world_pos[0],e.drone.pos[0])
    e.reset([])
    e.reset()


def test_step_applies_action_before_physics_and_updates_state_once():
    e=env(); calls=[]
    e.drone.controller.step=lambda action:calls.append('action')
    e.scene.step=lambda:calls.append('physics')
    e.drone.odom.odom_update=lambda:calls.append('odom')
    e.step(torch.zeros(2,4))
    assert calls==['action','physics','odom']


def test_invalid_quaternion_marks_reset_without_crashing():
    e=env(); e.drone.quat[0,0]=float('nan')
    e.drone.odom.odom_update()
    assert e.drone.odom.has_nan.tolist()==[True,False]
    assert torch.isfinite(e.drone.odom.body_quat).all()


def task():
    e=env(); cfg=yaml.safe_load((ROOT/'config/track_rl/rl_env.yaml').read_text())
    t=Track_task(e,e.env_config,cfg['train'],cfg['task'])
    return t,e


def test_reset_and_target_resample_observations_are_consistent():
    t,e=task(); obs=t.reset()['state']
    assert torch.allclose(obs[:,:3],e.drone.pos)
    assert torch.allclose(obs[:,6:9],obs[:,3:6]-obs[:,:3])
    t.command_buf[:]=e.drone.pos
    action=torch.zeros(2,4); action[:,3]=-.4
    obs,reward,done,extra=t.step(action); obs=obs['state']
    assert not done.any()
    assert torch.allclose(obs[:,6:9],obs[:,3:6]-obs[:,:3])
    assert torch.equal(obs[:,-4:],action)


def test_autoreset_does_not_leak_actions_or_timeout_into_survivors():
    t,e=task(); t.reset()
    t.episode_length_buf[0]=t.max_episode_length-1
    t.command_buf[:]=torch.tensor([.5,.5,1.])
    action=torch.full((2,4),.1)
    obs,reward,done,extra=t.step(action); obs=obs['state']
    assert done.tolist()==[True,False]
    assert extra['time_outs'].tolist()==[True,False]
    assert (obs[0,-4:]==0).all()
    assert torch.equal(obs[1,-4:],action[1])
    assert torch.allclose(obs[:,6:9],obs[:,3:6]-obs[:,:3])
    assert t.episode_length_buf.tolist()==[0,1]


def test_degree_termination_threshold():
    t,e=task(); t.reset(); t.task_config['termination_if_roll_greater_than']=45
    e.drone.quat[0]=torch.tensor([.8660254,.5,0.,0.])  # 60 degrees
    _,_,done,_=t.step(torch.zeros(2,4))
    assert done.tolist()==[True,False]
