import torch
import yaml
import genesis as gs
from genesis_drones.envs.genesis_env import Genesis_env
from genesis_drones.tasks.track_task import Track_task
from rsl_rl.runners import OnPolicyRunner

def main():
    gs.init(logging_level="warning")
    max_sim_step = 10000


    with open("config/track_rl/genesis_env.yaml", "r") as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)

    with open("config/track_rl/rl_env.yaml", "r") as file:
        rl_config = yaml.load(file, Loader=yaml.FullLoader)

    with open("config//track_rl/flight.yaml", "r") as file:
        flight_config = yaml.load(file, Loader=yaml.FullLoader)

    task_config = rl_config["task"]
    train_config = rl_config["train"]


    genesis_env = Genesis_env(
        env_config = env_config, 
        flight_config = flight_config,
        num_envs=1,
    )

    track_task = Track_task(
        genesis_env = genesis_env, 
        env_config = env_config, 
        task_config = task_config,
        train_config = train_config,
        num_envs=1,
    )

    runner = OnPolicyRunner(track_task, train_config, "", device="cuda:0")
    runner.load("logs/track_rl/policy_demo/model_500.pt")
    policy = runner.get_inference_policy(device="cuda:0")
    obs = track_task.reset()    # tensordict

    with torch.no_grad():
        for step in range(max_sim_step):
            actions = policy(obs)
            obs, rews, dones, infos = track_task.step(actions)

if __name__ == "__main__" :
    main()


    