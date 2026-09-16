
import genesis as gs
import torch
from rsl_rl.env.vec_env import VecEnv
from tensordict import TensorDict
import statistics
from collections import deque

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Track_task(VecEnv):
    def __init__(self, genesis_env, env_config, train_config, task_config, num_envs=None):
        # configs
        self.genesis_env = genesis_env
        self.env_config = env_config
        self.task_config = task_config
        self.train_config = train_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # shapes
        if num_envs is None:
            self.num_envs = self.env_config.get("num_envs", 1)
        else:
            self.num_envs = num_envs
        self.num_actions = task_config["num_actions"]
        self.num_commands = task_config["num_commands"]
        self.num_obs = task_config["num_obs"]

        # parameters
        self.max_horizon_vel = self.task_config.get("max_horizon_vel", 2.0)
        self.max_vertical_vel = self.task_config.get("max_vertical_vel", 1.0)
        self.max_episode_length = self.task_config.get("max_episode_length", 1500)
        self.reward_scales = task_config.get("reward_scales", {})
        self.obs_scales = task_config.get("obs_scales", {})
        self.command_cfg = self.task_config.get("command_cfg", {})
        self.step_dt = self.env_config.get("dt", 0.01)
        self.gamma = self.train_config["algorithm"]["gamma"]

        # buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.reward_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.command_buf = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.crash_condition_buf = torch.ones((self.num_envs,), device=self.device, dtype=bool)
        self.cur_pos_error = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_pos_error = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        # infos
        self.reward_functions = dict()
        self.episode_reward_sums = dict()
        self._register_reward_fun()
        self.reward_buffer_for_log = {name: deque(maxlen=10) for name in self.reward_functions.keys()}
        self.extras = dict()  # extra information for logging

        # counters
        self.cur_iter = 1
        self.step_cnt = 0
        self.num_steps_per_env = self.task_config.get("num_steps_per_env", 100)

        self.pi = torch.tensor(torch.pi, dtype=gs.tc_float, device=self.device)


    def compute_reward(self):
        for name, reward_func in self.reward_functions.items():
            reward = reward_func() * self._scale(name)
            self.reward_buf += torch.nan_to_num(reward, nan=-100, posinf=1.0, neginf=-1.0)
            self.episode_reward_sums[name] += reward

    def _scale(self, name):
        if self.cur_iter == 150 and name == "agular": 
            self.reward_scales[name] *= 5.0
        base = self.reward_scales[name]     
        return base * self.step_dt


    def _reward_target(self):

        target_reward = -torch.sum(torch.square(self.cur_pos_error), dim=1) * 0.1
        target_reward += torch.sum(torch.abs(self.last_pos_error) - torch.abs(self.cur_pos_error), dim=1)

        # target_reward = -torch.norm(self.cur_pos_error, dim=1) * 0.05
        # target_reward += torch.norm(self.last_pos_error, dim=1) - torch.norm(self.cur_pos_error, dim=1)

        target_reward[self._at_target()] += 20
        return target_reward

    def _reward_smooth(self):
        smooth_reward_rpy = torch.norm(self.actions[:, :3] - self.last_actions[:, :3], dim=1)
        smooth_reward_thrust = torch.abs(self.actions[:, 3] - self.last_actions[:, 3]) * 5.0
        smooth_reward = smooth_reward_rpy + smooth_reward_thrust
        return smooth_reward
    
    def _reward_vel(self):
        horizon_vel = torch.norm(self.genesis_env.drone.odom.world_linear_vel[:, :2], dim=1)
        vertical_vel = torch.abs(self.genesis_env.drone.odom.world_linear_vel[:, 2])
        hor_reward = -torch.relu(horizon_vel - self.max_horizon_vel)
        ver_reward = -torch.relu(vertical_vel - self.max_vertical_vel)
        vel_reward = hor_reward + ver_reward
        return vel_reward

    def _reward_angular(self):
        angular_reward = torch.sum(torch.abs(self.genesis_env.drone.odom.body_ang_acc), dim=1)
        return angular_reward

    def _reward_yaw(self):
        yaw = self.genesis_env.drone.odom.body_euler[:, 2]
        yaw_reward = torch.exp(self.task_config["yaw_lambda"] * torch.abs(yaw)) - 1
        return yaw_reward

    def _reward_crash(self):
        crash_reward = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        crash_reward[self.crash_condition_buf] = 1
        return crash_reward
    
    def _reward_lazy(self):
        lazy_reward = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        condition = self.genesis_env.drone.odom.world_pos[:, 2] < 0.1
        lazy_reward[condition] = self.episode_length_buf[condition] / self.max_episode_length
        
        return -lazy_reward
        
    def _resample_commands(self, envs_idx):
        self.command_buf[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),), self.device)
        self.command_buf[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(envs_idx),), self.device)
        self.command_buf[envs_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(envs_idx),), self.device)

    def _register_reward_fun(self):
        for name in self.reward_scales.keys():
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_reward_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

    def _at_target(self):
        at_target = ((torch.norm(self.cur_pos_error, dim=1) < self.task_config["target_thr"]).nonzero(as_tuple=False).flatten())
        return at_target

    def step(self, action):
        self.reward_buf[:] = 0.0
        self.step_cnt += 1
        if self.step_cnt == self.num_steps_per_env + 1:
            self.step_cnt = 0
            self.cur_iter = self.cur_iter + 1

        self.actions = torch.clip(action, -self.task_config["clip_actions"], self.task_config["clip_actions"])
        exec_actions = self.actions
        
        if self.genesis_env.target is not None:
            self.genesis_env.target.set_pos(self.command_buf, zero_velocity=True, envs_idx=list(range(self.num_envs)))
        self.genesis_env.step(exec_actions)
        self.episode_length_buf += 1

        # cal pos error
        self.last_pos_error[:] = self.command_buf - self.genesis_env.drone.odom.last_world_pos
        self.cur_pos_error[:] = self.command_buf - self.genesis_env.drone.odom.world_pos
        
        self.crash_condition_buf = (
            (torch.abs(self.genesis_env.drone.odom.body_euler[:, 1]) > self.task_config["termination_if_pitch_greater_than"])
            | (torch.abs(self.genesis_env.drone.odom.body_euler[:, 0]) > self.task_config["termination_if_roll_greater_than"])
            | (torch.abs(self.cur_pos_error[:, 0]) > self.task_config["termination_if_x_greater_than"])
            | (torch.abs(self.cur_pos_error[:, 1]) > self.task_config["termination_if_y_greater_than"])
            | (torch.abs(self.cur_pos_error[:, 2]) > self.task_config["termination_if_z_greater_than"])
            | (self.genesis_env.drone.odom.world_pos[:, 2] < self.task_config["termination_if_close_to_ground"])
        )
        self.reset_buf = (
            (self.episode_length_buf > self.max_episode_length) 
            | self.crash_condition_buf 
            | self.genesis_env.drone.odom.has_nan
        )
        self.compute_reward()
        self.reset(self.reset_buf.nonzero(as_tuple=False).flatten())
        self._resample_commands(self._at_target())
        self._update_obs()
        self.last_actions[:] = self.actions[:]  

        return self.get_observations(), self.reward_buf, self.reset_buf, self.extras


    def reset(self, env_idx=None):
        if env_idx is None:
            reset_range = torch.arange(self.num_envs, device=self.device)
        else:
            reset_range = env_idx

        self.genesis_env.reset(reset_range)
        self.last_actions[reset_range] = 0.0
        self.episode_length_buf[reset_range] = 0
        self.reset_buf[reset_range] = True
        self._update_extras(reset_range)
        self._resample_commands(reset_range)
        return self.get_observations()

    def get_observations(self):
        group_obs =  TensorDict({
            "state": self.obs_buf}, batch_size=self.num_envs
        )
        return group_obs


    def _update_obs(self):
        self.obs_buf = torch.cat(
            [
                self.genesis_env.drone.odom.world_pos,
                self.command_buf,
                self.cur_pos_error,
                self.genesis_env.drone.odom.body_quat,
                self.genesis_env.drone.odom.world_linear_vel,
                self.genesis_env.drone.odom.body_ang_vel,
                self.last_actions,
            ],
            axis=-1,
        )

    def get_privileged_observations(self):
        return None
    
    def _update_extras(self, env_idx=None):
        if env_idx is None:
            reset_range = torch.arange(self.num_envs, device=self.device)
        else:
            reset_range = env_idx
        self.extras["episode"] = {}

        for key in self.episode_reward_sums.keys():
            rewards = self.episode_reward_sums[key][reset_range].cpu().numpy().tolist()
            self.reward_buffer_for_log[key].extend(rewards)

            if len(self.reward_buffer_for_log[key]) > 0:
                mean_reward = statistics.mean(self.reward_buffer_for_log[key])
            else:
                mean_reward = 0.0

            self.extras["episode"]["reward_" + key] = mean_reward
            self.episode_reward_sums[key][reset_range] = 0.0


