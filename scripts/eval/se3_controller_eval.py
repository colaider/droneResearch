import argparse
import torch
import yaml
import genesis as gs
import numpy as np
from genesis_drones.envs.genesis_env import Genesis_env
from genesis_drones.tasks.track_task import Track_task
from genesis_drones.controllers import SE3Controller
from genesis_drones.utils.flatness import compute_altitude


def circular_trajectory(t, R, omega, center, device=None, dtype=torch.float32):
    cx, cy, cz = center
    device = device or torch.device("cpu")
    t = torch.as_tensor(t, dtype=dtype, device=device)

    R = torch.as_tensor(R, dtype=dtype, device=device)
    omega = torch.as_tensor(omega, dtype=dtype, device=device)
    cx, cy, cz = [torch.as_tensor(v, dtype=dtype, device=device) for v in (cx, cy, cz)]

    x_ref = torch.stack(
        [
            cx + R * torch.cos(omega * t),
            cy,
            cz + R * torch.sin(omega * t),
        ]
    )
    x_dot_ref = torch.stack(
        [
            -R * omega * torch.sin(omega * t),
            torch.zeros_like(t),
            R * omega * torch.cos(omega * t),
        ]
    )
    x_ddot_ref = torch.stack(
        [
            -R * omega**2 * torch.cos(omega * t),
            torch.zeros_like(t),
            -R * omega**2 * torch.sin(omega * t),
        ]
    )
    x_dddot_ref = torch.stack(
        [
            R * omega**3 * torch.sin(omega * t),
            torch.zeros_like(t),
            -R * omega**3 * torch.cos(omega * t),
        ]
    )

    psi_info_cmd, omega_cmd = yaw_cmd(x_dot_ref, x_ddot_ref, x_dddot_ref, device=device)
    yaw = psi_info_cmd["psi"]
    yaw_dot = psi_info_cmd["dpsi"]

    return x_ref, x_dot_ref, x_ddot_ref, x_dddot_ref, yaw, yaw_dot


def yaw_cmd(vel_cmd, acc_cmd, jerk_cmd, device=None):
    with open("config/track_rl/flight.yaml", "r") as file:
        flight_config = yaml.load(file, Loader=yaml.FullLoader)

    G = torch.tensor([0.0, 0.0, -flight_config["g"]], device=device)

    # --- Compute Commands ---

    f_cmd = flight_config["weight"] * (acc_cmd - G)

    thrust_cmd = torch.linalg.norm(f_cmd)

    zb_cmd = f_cmd / (thrust_cmd + 1e-6)

    proj_cmd = torch.eye(3).to(device) - torch.outer(zb_cmd, zb_cmd)

    dz_cmd = proj_cmd @ jerk_cmd / thrust_cmd

    dz_mag_cmd = torch.linalg.norm(dz_cmd)

    omega_xy_scale_cmd = torch.clamp(
        flight_config["max_roll_rate"] / (dz_mag_cmd + 1e-2), None, 1
    )
    real_dz_cmd = dz_cmd * omega_xy_scale_cmd

    # Compute the desired body frame axes (xb, yb), angular velocity (omega),
    xb_cmd, yb_cmd, omega_cmd, dx_cmd, psi_info_cmd = compute_altitude(
        vel=vel_cmd, acc=acc_cmd, zb=zb_cmd, dz=real_dz_cmd
    )
    return psi_info_cmd, omega_cmd


def controller_action_convert(control, flight_config, env_config, device):
    """
    normlize the ctrl output to action space
    """
    # ensure tensors on device
    cmd_thrust = control["cmd_thrust"]  # (B,)
    cmd_w = control["cmd_w"]  # (B, 3)
    cmd_q = control["cmd_q"]  # (B, 4) [x,y,z,w]

    min_t = flight_config["min_t"]
    max_t = flight_config["max_t"]

    # -------- thrust normalize --------
    thrust_norm = (cmd_thrust - min_t) / (max_t - min_t)
    thrust_norm = thrust_norm * 2.0 - 1.0

    if env_config["controller"] == "rate":
        # -------- rate normalize --------
        # cmd_w is (B, 3) -> roll, pitch, yaw rates
        wx = cmd_w[:, 0]
        wy = cmd_w[:, 1]
        wz = cmd_w[:, 2]

        roll_norm = torch.clamp(wx / flight_config["max_roll_rate"], -1.0, 1.0)
        pitch_norm = torch.clamp(wy / flight_config["max_pitch_rate"], -1.0, 1.0)
        yaw_norm = torch.clamp(wz / flight_config["max_yaw_rate"], -1.0, 1.0)

        # Stack into (B, 4) -> [roll, pitch, yaw, thrust]
        action = torch.stack([roll_norm, pitch_norm, yaw_norm, thrust_norm], dim=1)

    elif env_config["controller"] == "angle":
        # -------- angle normalize --------
        # cmd_q: (B, 4) in [x, y, z, w] Scipy format
        
        # 1. (x, y, z, w)
        x, y, z, w = cmd_q.unbind(-1)
        
        # 2.  Euler Angles (Roll-Pitch-Yaw)
        # Roll (x-axis rotation)
        # atan2( 2(wx - yz), 1 - 2(x^2 + y^2) )
        sin_roll = 2.0 * (w * x - y * z)
        cos_roll = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sin_roll, cos_roll)
        
        # Pitch (y-axis rotation)
        # asin( 2(wy + xz) )
        sin_pitch = 2.0 * (w * y + x * z)
        sin_pitch = torch.clamp(sin_pitch, -1.0, 1.0)
        pitch = torch.asin(sin_pitch)
        
        # Yaw (z-axis rotation)
        # atan2( 2(wz - xy), 1 - 2(y^2 + z^2) )
        sin_yaw = 2.0 * (w * z - x * y)
        cos_yaw = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(sin_yaw, cos_yaw)
        
        eulers = torch.stack([roll, pitch, yaw], dim=-1) # (B, 3)

        action = torch.cat([eulers, thrust_norm.view(-1, 1)], dim=1)  # (B, 4)

    return action


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--use-trajectory",
        action="store_true",
        help="Use reference trajectory. If not provided, command buffer is used.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    use_trajectory = args.use_trajectory
    gs.init(logging_level="warning")
    max_sim_step = 10000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("config/se3_controller_eval/genesis_env.yaml", "r") as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)

    with open("config/se3_controller_eval/rl_env.yaml", "r") as file:
        rl_config = yaml.load(file, Loader=yaml.FullLoader)

    with open("config/se3_controller_eval/flight.yaml", "r") as file:
        flight_config = yaml.load(file, Loader=yaml.FullLoader)

    task_config = rl_config["task"]
    train_config = rl_config["train"]

    genesis_env = Genesis_env(
        env_config=env_config,
        flight_config=flight_config,
        num_envs=1,
    )

    track_task = Track_task(
        genesis_env=genesis_env,
        env_config=env_config,
        task_config=task_config,
        train_config=train_config,
        num_envs=1,
    )
    controller = SE3Controller(flight_config["se3_control"], device)
    controller.randomize_params(num_envs=1, mass_std=0.0, pid_scale_range=(1.0, 1.0))
    obs = track_task.reset()  # tensordict
    with torch.no_grad():
        for step in range(max_sim_step):
            state = {
                "x": genesis_env.drone.odom.world_pos.view(1, 3),
                "v": genesis_env.drone.odom.world_linear_vel.view(1, 3),
                "q": genesis_env.drone.odom.body_quat.view(1, 4),
                "w": genesis_env.drone.odom.body_ang_vel.view(1, 3),
            }
            if use_trajectory:
                dt = env_config["dt"]

                t = step * dt

                # circular trajectory
                R = 0.5
                omega = 3
                cx, cy, cz = 0.0, 0.0, 0.8  # center of the circle
                x_ref, x_dot_ref, x_ddot_ref, x_dddot_ref, yaw, yaw_dot = (
                    circular_trajectory(
                        t, R, omega, (cx, cy, cz), device, torch.float32
                    )
                )
            else:
                x_ref = track_task.command_buf.view(1, 3)
                x_dot_ref = torch.zeros(3, device=device)
                x_ddot_ref = torch.zeros(3, device=device)
                x_dddot_ref = torch.zeros(3, device=device)
                yaw = torch.tensor(0.0, device=device)
                yaw_dot = torch.tensor(0.0, device=device)

            flat = {
                "x": x_ref.view(1, 3),
                "x_dot": x_dot_ref.view(1, 3),
                "x_ddot": x_ddot_ref.view(1, 3),
                "x_dddot": x_dddot_ref.view(1, 3),
                "yaw": yaw.view(1, 1),
                "yaw_dot": yaw_dot.view(1, 1),
            }
            ctrl = controller.update(0, state, flat, None, "wxyz")
            action = controller_action_convert(
                control=ctrl,
                flight_config=flight_config,
                env_config=env_config,
                device=device,
            )

            obs, rews, dones, infos = track_task.step(action)


if __name__ == "__main__":
    main()
