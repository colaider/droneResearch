import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


import shutil
import os
import yaml
from genesis_drones.envs.genesis_env import Genesis_env
from genesis_drones.tasks.track_task import Track_task
from rsl_rl.runners import OnPolicyRunner
import time
from datetime import datetime
import genesis as gs
import warp as wp

def main():
    # logging_level="warning"
    gs.init(backend=gs.cuda if torch.cuda.is_available() else gs.cpu, logging_level="warning")
    # gs.init(backend=gs.gpu, debug=True, logging_level="debug")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = str(ROOT / f"logs/track_rl/track_{timestamp}")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    bash_content = f"""#!/bin/bash
    tensorboard --logdir="{log_dir}"
    """

    bash_path = ROOT / "scripts/shell/launch_tb.bash"
    with open(bash_path, "w") as f:
        f.write(bash_content)

    with open(ROOT / "config/track_rl/genesis_env.yaml", "r") as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)

    with open(ROOT / "config/track_rl/rl_env.yaml", "r") as file:
        rl_config = yaml.load(file, Loader=yaml.FullLoader)

    with open(ROOT / "config/track_rl/flight.yaml", "r") as file:
        flight_config = yaml.load(file, Loader=yaml.FullLoader)

    task_config = rl_config["task"]
    train_config = rl_config["train"]


    genesis_env = Genesis_env(
        env_config = env_config, 
        flight_config = flight_config,
    )

    track_task = Track_task(
        genesis_env = genesis_env, 
        env_config = env_config, 
        task_config = task_config,
        train_config = train_config,
    )

    track_task.reset()
    runner = OnPolicyRunner(track_task, train_config, log_dir, device=genesis_env.device)
    # runner.load("logs/track_rl/track_2025-10-15_00:49:51/model_5900.pt")
    runner.learn(num_learning_iterations=train_config["max_iterations"], init_at_random_ep_len=True)

if __name__ == "__main__" :
    wp.config.enable_backward_log = True
    main()


    