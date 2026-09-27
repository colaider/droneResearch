from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


import shutil
import os
import yaml
import torch
import time
from datetime import datetime
import genesis as gs
import warp as wp
from genesis_drones.envs.genesis_env import Genesis_env
from genesis_drones.utils.mavlink_rc import start_mavlink_receive_thread

def gs_rand_float(lower, upper, device=None):
    shape = lower.shape  # scalar
    return (upper - lower) * torch.rand(size=shape, device=lower.device if device is None else device) + lower



def main():
    # logging_level="warning"
    gs.init(backend=gs.cuda if torch.cuda.is_available() else gs.cpu, logging_level="warning")
    num_envs = 1
    command_buf = torch.zeros((num_envs, 4), device=gs.device, dtype=gs.tc_float)
    def update_commands(cur_pos, envs_idx=None):
        if envs_idx is None:
            idx = torch.arange(num_envs, device=gs.device)
        else:
            idx = envs_idx
        command_buf[idx, 0] = gs_rand_float(cur_pos[idx, 0]-0.5, cur_pos[idx, 0]+0.5)
        command_buf[idx, 1] = gs_rand_float(cur_pos[idx, 1]-0.5, cur_pos[idx, 1]+0.5)
        command_buf[idx, 2] = gs_rand_float(torch.clamp(cur_pos[idx, 2]-0.2, min=0.3, max=2.0), torch.clamp(cur_pos[idx, 2]+0.2, min=0.5, max=2.2))

    
    def at_target(cur_pos):
        cur_pos_error = cur_pos - command_buf[:, :3]
        at_target = ((torch.norm(cur_pos_error, dim=1) < 0.1).nonzero(as_tuple=False).flatten())
        return at_target

    with open(ROOT / "config/pos_ctrl_eval/genesis_env.yaml", "r") as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(ROOT / "config/pos_ctrl_eval/flight.yaml", "r") as file:
        flight_config = yaml.load(file, Loader=yaml.FullLoader)


    genesis_env = Genesis_env(
        env_config = env_config, 
        flight_config = flight_config,
    )

    device = flight_config.get("USB_path", "/dev/ttyUSB0")
    if not os.path.exists(device):
        print(f"[MAVLINK] Device {device} not found, skipping mavlink thread.")
    else :
        start_mavlink_receive_thread(device, rc_config=flight_config)

    while True:
        cur_pos = genesis_env.drone.odom.world_pos
        update_commands(cur_pos, at_target(cur_pos))
        genesis_env.target.set_pos(command_buf[:, :3], zero_velocity=True, envs_idx=list(range(num_envs)))
        genesis_env.step(command_buf)

if __name__ == "__main__" :
    wp.config.enable_backward_log = True
    main()


    