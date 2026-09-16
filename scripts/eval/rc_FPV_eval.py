

import shutil
import os
import yaml
import torch
from genesis_drones.controllers import PIDcontroller
from genesis_drones.sensors.odom import Odom
from genesis_drones.utils.mavlink_rc import start_mavlink_receive_thread
from genesis_drones.envs.genesis_env import Genesis_env
from rsl_rl.runners import OnPolicyRunner
import time
from datetime import datetime
import genesis as gs
import warp as wp

def main():

    # logging_level="warning"
    gs.init()
    
    with open("config/rc_FPV_eval/genesis_env.yaml", "r") as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)
    with open("config/rc_FPV_eval/flight.yaml", "r") as file:
        flight_config = yaml.load(file, Loader=yaml.FullLoader)


    genesis_env = Genesis_env(
        env_config = env_config, 
        flight_config = flight_config,
    )

    device = "/dev/ttyUSB0"
    if not os.path.exists(device):
        print(f"[MAVLINK] Device {device} not found, skipping mavlink thread.")
    else :
        start_mavlink_receive_thread(device)

    while True:
        genesis_env.step()
if __name__ == "__main__" :
    wp.config.enable_backward_log = True
    main()


    