"""Run explicitly: python -m tests.smoke_simulation (not part of quick unit tests)."""
from pathlib import Path
import genesis as gs
import torch
import yaml
from genesis_drones.envs.genesis_env import Genesis_env
from genesis_drones.tasks.track_task import Track_task


def main():
    root=Path(__file__).resolve().parents[1]
    gs.init(backend=gs.cpu,logging_level='error')
    env_cfg=yaml.safe_load((root/'config/track_rl/genesis_env.yaml').read_text())
    env_cfg.update(num_envs=2,show_viewer=False,vis_waypoints=False,render_cam=False,
                   fixed_init_pos=True,drone_init_pos=[0.,0.,1.])
    flight=yaml.safe_load((root/'config/track_rl/flight.yaml').read_text())
    rl=yaml.safe_load((root/'config/track_rl/rl_env.yaml').read_text())
    env=Genesis_env(env_cfg,flight)
    task=Track_task(env,env_cfg,rl['train'],rl['task'])
    task.reset()
    action=torch.zeros(2,4,device=env.device)
    action[:,3]=2/flight['TWR']-1
    for _ in range(20):
        obs,reward,done,_=task.step(action)
        assert torch.isfinite(obs['state']).all()
        assert not done.any()
    before=env.drone.get_pos().clone()
    task.reset(torch.tensor([0],device=env.device))
    assert torch.allclose(env.drone.get_pos()[1],before[1])
    assert torch.allclose(task.obs_buf[:,6:9],task.command_buf-env.drone.odom.world_pos)
    assert torch.max(torch.abs(env.drone.get_pos()[:,2]-1))<.1
    print('PASS: 20 hover steps, finite observations, partial reset preserves survivor')
    pid = env.drone.controller
    pid.controller_mode = 'position'
    pid.controller = pid.position_controller
    env.reset()
    target = torch.tensor([[0.,0.,1.2,0.],[0.,0.,.8,0.]], device=env.device)
    for _ in range(120):
        env.step(target)
    heights = env.drone.get_pos()[:,2]
    assert heights[0] > 1.05 and .5 < heights[1] < .95, heights
    print('PASS: position control raises and lowers altitude', heights.tolist())

    from rsl_rl.runners import OnPolicyRunner
    pid.controller_mode = 'angle'
    pid.controller = pid.angle_controller
    task.reset()
    train = rl['train']
    train['num_steps_per_env'] = 4
    train['algorithm']['num_learning_epochs'] = 1
    train['algorithm']['num_mini_batches'] = 1
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix="drone-ppo-smoke-") as log_dir:
        runner = OnPolicyRunner(task, train, log_dir, device=str(env.device))
        runner.learn(num_learning_iterations=1)
    print('PASS: one PPO rollout and optimizer update')

if __name__=='__main__': main()
