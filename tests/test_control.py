from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from genesis_drones.controllers.pid_controller import PIDcontroller
from genesis_drones.controllers.se3_controller import SE3Controller
from genesis_drones.sensors.odom import Odom, ve2vb
from scripts.eval.se3_controller_eval import controller_action_convert
from swarm.droneClasses.dronePositionCTRL import DronePositionCTRL
from swarm.utilities.geometry import rotation, integrate_attitude

ROOT = Path(__file__).resolve().parents[1]

def controller(mode='position'):
    odom = Odom(1, device=torch.device('cpu'))
    cfg = yaml.safe_load((ROOT/'config/pos_ctrl_eval/flight.yaml').read_text())
    pid = PIDcontroller(1, torch.zeros(7), odom, cfg, controller=mode, device='cpu')
    return pid, odom

def test_position_collective_tracks_altitude_both_directions():
    pid, odom = controller()
    for height, relation in [(0., 0), (0.5, 1), (-0.5, -1)]:
        pid.reset()
        pid.position_controller(torch.tensor([[0., 0., height, 0.]]))
        rpm = pid.mixer(torch.zeros(1, 4)).mean().item()
        if relation == 0:
            assert abs(rpm-pid.base_rpm) < 0.01
        elif relation > 0:
            assert rpm > pid.base_rpm
        else:
            assert rpm < pid.base_rpm

def test_rc_collective_uses_normalized_units():
    pid, _ = controller('rate')
    pid.rc_command[3] = 0.01
    expected = pid.base_rpm * np.sqrt(0.01*pid.TWR)
    assert torch.allclose(pid.mixer(), torch.full((1,4), expected), atol=0.01)

def test_world_to_body_yaw():
    assert torch.allclose(ve2vb(torch.tensor([[1.,0.,0.]]), torch.tensor([torch.pi/2])),
                          torch.tensor([[0.,-1.,0.]]), atol=1e-6)

def test_position_command_clip_and_damping():
    p = DronePositionCTRL(SimpleNamespace())
    p.get_attitude = lambda: np.zeros(3)
    p.get_ang_vel = lambda: np.zeros(3)
    p.get_imu_pos = lambda: np.zeros(3)
    p.get_lin_vel = lambda: np.zeros(3)
    assert np.max(np.abs(p.position_control(np.array([100.,0.,0.,0.])))) <= 2
    p.k_f2[:] = 0
    p.get_lin_vel = lambda: np.array([1.,0.,0.])
    p.pr_setpoint = np.zeros(4)
    assert p.position_control(np.zeros(4))[0] < 0

def test_body_rate_integration_matches_rotation():
    rpy = np.array([0.4,-0.3,0.8])
    rates = np.array([0.2,0.4,-0.1])
    dt = 1e-5
    R = rotation(rpy)
    derivative = (rotation(integrate_attitude(rpy,rates,dt))-R)/dt
    wx = np.array([[0.,-rates[2],rates[1]],[rates[2],0.,-rates[0]],[-rates[1],rates[0],0.]])
    assert np.allclose(derivative, R@wx, atol=1e-5)

def test_se3_hover_without_randomization_and_no_input_mutation():
    cfg = yaml.safe_load((ROOT/'config/se3_controller_eval/flight.yaml').read_text())
    c = SE3Controller(cfg['se3_control'], 'cpu')
    state = dict(x=torch.zeros(1,3),v=torch.zeros(1,3),w=torch.zeros(1,3),q=torch.tensor([[1.,0.,0.,0.]]))
    q = state['q'].clone()
    flat = {k:torch.zeros(1,3) for k in ['x','x_dot','x_ddot','x_dddot']}
    flat.update(yaw=torch.zeros(1,1),yaw_dot=torch.zeros(1,1))
    out = c.update(0,state,flat)
    assert torch.equal(state['q'],q)
    assert torch.allclose(out['cmd_motor_speeds'],torch.full((1,4),cfg['base_rpm']),atol=0.02)
    flat['x'][:] = torch.tensor([[100.,0.,-100.]])
    out = c.update(0,state,flat)
    assert torch.isfinite(out['cmd_motor_speeds']).all()
    assert (out['cmd_motor_speeds']>=0).all()
    assert (out['cmd_motor_speeds']<=c.rotor_speed_max).all()

def test_hopf_quaternion_matches_matrix_in_both_charts():
    cfg = yaml.safe_load((ROOT/'config/se3_controller_eval/flight.yaml').read_text())
    c = SE3Controller(cfg['se3_control'], 'cpu')
    force = torch.tensor([[1.,2.,3.],[1.,2.,-3.],[0.,0.,-1.]])
    R,w,q = c._batch_safe_hopf(force,torch.zeros_like(force),torch.zeros(3),torch.zeros(3))
    assert torch.allclose(c._quat_to_rot_matrix(q),R,atol=1e-6)
    assert torch.allclose(R[:,:,2],torch.nn.functional.normalize(force,dim=1),atol=1e-6)

def test_quaternion_action_conversion_uses_standard_rpy():
    from swarm.utilities.geometry import rpy_to_quat
    rpy=np.array([.3,-.2,.4]); q=rpy_to_quat(rpy)
    control={'cmd_thrust':torch.tensor([1.]), 'cmd_w':torch.zeros(1,3),
             'cmd_q':torch.tensor(np.r_[q[1:],q[:1]][None],dtype=torch.float32)}
    action=controller_action_convert(control,{'min_t':0.,'max_t':10.},{'controller':'angle'},'cpu')
    assert np.allclose(action[0,:3].numpy(),rpy,atol=1e-6)
