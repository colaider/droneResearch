import numpy as np
import torch
import yaml
from typing import Union


class SE3Controller(object):
    """
    PyTorch-based Vectorized SE3 Controller.
    Fully differentiable and GPU-accelerated.

    Quadrotor trajectory tracking controller based on https://ieeexplore.ieee.org/document/5717652
    with Hopf Fibration based attitude control from
    'Control of Quadrotors Using the Hopf Fibration on SO(3)'

    Fix: handle c -> -1 singularity by flipping the S^2 chart (force c >= 0),
         compute Hopf quantities in the stable chart, then flip back R_des and ω_des.
    """

    def __init__(self, config: Union[str, dict], device):
        # Load config
        if isinstance(config, dict):
            cfg = config
        elif isinstance(config, str):
            with open(config, "r") as file:
                cfg = yaml.load(file, Loader=yaml.FullLoader)

        self.device = device

        # =====================
        # Inertia (3, 3)
        # =====================
        self.nominal_mass = float(cfg["inertia"]["mass"])

        # Construct Inertia Matrix directly as Tensor
        Ixx = cfg["inertia"]["Ixx"]
        Iyy = cfg["inertia"]["Iyy"]
        Izz = cfg["inertia"]["Izz"]
        Ixy = cfg["inertia"]["Ixy"]
        Ixz = cfg["inertia"]["Ixz"]
        Iyz = cfg["inertia"]["Iyz"]

        self.nominal_inertia = torch.tensor(
            [
                [Ixx, Ixy, Ixz],
                [Ixy, Iyy, Iyz],
                [Ixz, Iyz, Izz],
            ],
            device=self.device,
            dtype=torch.float32,
        )

        self.g = float(cfg["flight"]["g"])
        self.omega_z_limit = float(cfg["flight"]["omega_z_limit"])
        self.omega_xy_limit = float(cfg["flight"]["omega_xy_limit"])

        # =====================
        # Gains
        # =====================
        self.nominal_kp_pos = torch.tensor(cfg["gains"]["kp_pos"], device=self.device)
        self.nominal_kd_pos = torch.tensor(cfg["gains"]["kd_pos"], device=self.device)
        self.nominal_kp_att = torch.tensor(cfg["gains"]["kp_att"], device=self.device)
        self.nominal_kd_att = torch.tensor(cfg["gains"]["kd_att"], device=self.device)
        # self.kp_vel = 0.1 * self.kp_pos
        self.mass = torch.tensor([self.nominal_mass], device=self.device)
        self.inertia = self.nominal_inertia.unsqueeze(0)  # (1, 3, 3)
        self.kp_pos = self.nominal_kp_pos.unsqueeze(0)  # (1, 3)
        self.kd_pos = self.nominal_kd_pos.unsqueeze(0)
        self.kp_att = self.nominal_kp_att.unsqueeze(0)
        self.kd_att = self.nominal_kd_att.unsqueeze(0)
        # =====================
        # Allocation
        # =====================
        d = cfg["arm_length"]
        self.num_rotors = cfg["geometry"]["num_rotors"]

        rotor_pos_np = np.array(
            [
                cfg["geometry"]["rotor_pos"][k]
                for k in sorted(cfg["geometry"]["rotor_pos"].keys())
            ]
        )
        rotor_pos_np = d * rotor_pos_np  # Scale by arm length

        rotor_dir = np.array(cfg["geometry"]["rotor_directions"])

        k_eta = cfg["rotor"]["k_eta"]
        k_m = cfg["rotor"]["k_m"]
        k = k_m / k_eta
        self.k_eta = k_eta

        # Build Allocation Matrix (NumPy first, then convert to Torch)
        f_to_TM = np.vstack(
            (
                np.ones((1, self.num_rotors)),
                np.hstack(
                    [
                        np.cross(r, np.array([0, 0, 1]))[:2].reshape(-1, 1)
                        for r in rotor_pos_np
                    ]
                ),
                (k * rotor_dir).reshape(1, -1),
            )
        )
        TM_to_f_np = np.linalg.inv(f_to_TM)

        # Convert to Tensor (4, 4)
        self.TM_to_f = torch.from_numpy(TM_to_f_np).to(
            device=self.device, dtype=torch.float32
        )

    # ------------------------
    # TORCH HELPERS
    # ------------------------
    @staticmethod
    def quat_mul(q1, q2):
        """
        q1 * q2 (Hamilton product).
        q: (B, 4) [w, x, y, z]
        """
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)

        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )

    @staticmethod
    def vee(S):
        """
        S: (B, 3, 3) skew-symmetric
        Returns: (B, 3) vector
        """
        return torch.stack([-S[:, 1, 2], S[:, 0, 2], -S[:, 0, 1]], dim=1)

    @staticmethod
    def _quat_to_rot_matrix(q):
        """
        q: (B, 4) [x, y, z, w] (Scipy convention input)
        """
        x, y, z, w = q.unbind(-1)

        x2, y2, z2 = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        # Row 0
        r00 = 1 - 2 * (y2 + z2)
        r01 = 2 * (xy - wz)
        r02 = 2 * (xz + wy)

        # Row 1
        r10 = 2 * (xy + wz)
        r11 = 1 - 2 * (x2 + z2)
        r12 = 2 * (yz - wx)

        # Row 2
        r20 = 2 * (xz - wy)
        r21 = 2 * (yz + wx)
        r22 = 1 - 2 * (x2 + y2)

        # Stack to (B, 3, 3)
        row0 = torch.stack([r00, r01, r02], dim=1)
        row1 = torch.stack([r10, r11, r12], dim=1)
        row2 = torch.stack([r20, r21, r22], dim=1)
        return torch.stack([row0, row1, row2], dim=1)

    # ------------------------
    # HOPF LOGIC (TORCH)
    # ------------------------
    def _batch_safe_hopf(self, zeta, zeta_dot, yaw, yaw_dot, eps=1e-6):
        """
        All inputs are Tensors on GPU.
        zeta: (B, 3)
        yaw: (B, 1) or (B,)
        """
        # Ensure shapes
        if yaw.dim() == 1:
            yaw = yaw.unsqueeze(1)  # (B, 1)
        if yaw_dot.dim() == 1:
            yaw_dot = yaw_dot.unsqueeze(1)

        zeta_norm = torch.norm(zeta, dim=1, keepdim=True)  # (B, 1)
        zeta_norm_safe = torch.clamp(zeta_norm, min=eps)

        s = zeta / zeta_norm_safe  # (B, 3)
        a, b, c = s[:, 0], s[:, 1], s[:, 2]  # (B,) views

        # s_dot calculation
        s_dot_zeta = torch.sum(s * zeta_dot, dim=1, keepdim=True)
        s_dot = (zeta_dot - s * s_dot_zeta) / zeta_norm_safe

        a_dot, b_dot, c_dot = s_dot[:, 0], s_dot[:, 1], s_dot[:, 2]

        # ---- Flip Chart Logic (c < 0) ----
        # No branching, use masking
        mask_neg = c < 0.0  # Bool tensor
        flip_sign = torch.where(mask_neg, -1.0, 1.0)  # (B,)

        # Apply flip
        a = a * flip_sign
        b = b * flip_sign
        c = c * flip_sign
        a_dot = a_dot * flip_sign
        b_dot = b_dot * flip_sign
        c_dot = c_dot * flip_sign

        # ---- Hopf Quaternion q_abc [w, x, y, z] ----
        one_plus_c = torch.clamp(1.0 + c, min=eps)
        denom = torch.sqrt(2.0 * one_plus_c)

        qw = (1.0 + c) / denom
        qx = -b / denom
        qy = a / denom
        qz = torch.zeros_like(qw)
        q_abc = torch.stack([qw, qx, qy, qz], dim=1)

        # ---- Yaw Quaternion q_psi [w, x, y, z] ----
        # yaw is (B, 1), flatten to (B,)
        yaw = yaw.view(-1)
        yaw = (yaw + torch.pi) % (2.0 * torch.pi) - torch.pi
        half_yaw = 0.5 * yaw

        zeros = torch.zeros_like(yaw)
        q_psi = torch.stack(
            [torch.cos(half_yaw), zeros, zeros, torch.sin(half_yaw)], dim=1
        )

        # ---- Total Quaternion ----
        q_tot = self.quat_mul(q_abc, q_psi)  # [w, x, y, z]

        # Output q in Scipy order [x, y, z, w] for compatibility if needed
        q_scipy = torch.stack(
            [q_tot[:, 1], q_tot[:, 2], q_tot[:, 3], q_tot[:, 0]], dim=1
        )

        # R_des
        R_des = self._quat_to_rot_matrix(q_scipy)

        # ---- Omega Calculation ----
        sinp = torch.sin(yaw)
        cosp = torch.cos(yaw)

        omg_term = c_dot / one_plus_c

        omega1 = sinp * a_dot - cosp * b_dot - (a * sinp - b * cosp) * omg_term
        omega2 = cosp * a_dot + sinp * b_dot - (a * cosp + b * sinp) * omg_term
        omega3 = (b * a_dot - a * b_dot) / one_plus_c + yaw_dot.view(-1)

        omega3 = torch.clamp(omega3, -self.omega_z_limit, self.omega_z_limit)
        w_des = torch.stack([omega1, omega2, omega3], dim=1)
        w_des = torch.clamp(w_des, -self.omega_xy_limit, self.omega_xy_limit)

        # ---- Flip Back Logic ----
        # R_des columns 0 and 2, w_des x and z need flipping if mask was true
        f_sign = flip_sign.view(-1, 1, 1)  # (B, 1, 1) for broadcasting

        # Create a flip matrix or just operate on slices (torch slices are views, careful)
        # R_des is (B, 3, 3). We want to multiply col 0 and 2 by f_sign
        R_des = R_des.clone()  # Avoid in-place modification errors in gradients
        R_des[:, :, 0] = R_des[:, :, 0] * f_sign.view(-1, 1)
        R_des[:, :, 2] = R_des[:, :, 2] * f_sign.view(-1, 1)

        w_des = w_des.clone()
        w_des[:, 0] = w_des[:, 0] * flip_sign
        w_des[:, 2] = w_des[:, 2] * flip_sign

        return R_des, w_des, q_scipy

    def randomize_params(self, num_envs, mass_std=0.05, pid_scale_range=(0.8, 1.2)):
        """
        num_envs: Number of environments (Batch Size)
        mass_std: Standard deviation for mass Gaussian distribution (kg)
        pid_scale_range: Uniform distribution range for PID parameter scaling (min, max)
        """

        # 1. Mass Randomization (Gaussian)
        # shape: (B, 1) to support broadcasting
        mass_noise = torch.normal(
            mean=0.0, std=mass_std, size=(num_envs, 1), device=self.device
        )
        self.mass = self.nominal_mass + mass_noise

        # Safety clamp, avoid mass <= 0
        self.mass = torch.clamp(self.mass, min=0.01)

        # 2. Inertia Matrix Randomization
        # Physically, inertia matrix usually scales linearly with mass: J_new = J_nom * (m_new / m_nom)
        mass_ratio = self.mass / self.nominal_mass  # (B, 1)

        # nominal_inertia: (3, 3) -> (1, 3, 3)
        # mass_ratio: (B, 1) -> (B, 1, 1)
        # Result: (B, 3, 3)
        self.inertia = self.nominal_inertia.unsqueeze(0) * mass_ratio.unsqueeze(-1)

        # 3. PID Parameter Randomization (Uniform distribution scaling)
        # Generate random scale matrix (B, 3)
        low, high = pid_scale_range

        def get_rand_gains(nominal_gain):
            # random scale shape: (B, 3) -> independent random for x, y, z axes
            scale = torch.rand((num_envs, 3), device=self.device) * (high - low) + low
            return nominal_gain.unsqueeze(0) * scale  # (1, 3) * (B, 3) -> (B, 3)

        self.kp_pos = get_rand_gains(self.nominal_kp_pos)
        self.kd_pos = get_rand_gains(self.nominal_kd_pos)
        self.kp_att = get_rand_gains(self.nominal_kp_att)
        self.kd_att = get_rand_gains(self.nominal_kd_att)
        self.kp_vel = 0.1 * self.kp_pos

    # ============================================================
    #                       MAIN UPDATE
    # ============================================================
    def update(self, t, state, flat, omega_cmd=None, quat_format="wxyz"):
        """
        Inputs should be Tensors on GPU.
        state: {'x': (B,3), 'v': (B,3), 'q': (B,4), 'w': (B,3)}
        flat:  {'x': (B,3), ... 'yaw': (B,1)}
        """
        q_input = state["q"] # 期望形状 (B, 4)
        
        if quat_format.lower() == "wxyz":
            state["q"] = torch.cat([q_input[:, 1:], q_input[:, 0:1]], dim=1)
        elif quat_format.lower() == "xyzw":
            state["q"] = q_input
        else:
            raise ValueError(f"Unknown quat_format: {quat_format}")
        # 1. Desired Force
        pos_err = state["x"] - flat["x"]
        vel_err = state["v"] - flat["x_dot"]

        target_acc = -self.kp_pos * pos_err - self.kd_pos * vel_err + flat["x_ddot"]
        target_acc[:, 2] += self.g

        F_des = self.mass * target_acc  # (B, 3)

        # 2. Current Attitude
        R = self._quat_to_rot_matrix(state["q"])  # (B, 3, 3)
        b3 = R[:, :, 2]  # (B, 3)

        # 3. Thrust (u1)
        u1 = torch.sum(F_des * b3, dim=1)  # (B,)

        # 4. Desired Att
        R_des, w_des, q_des = self._batch_safe_hopf(
            target_acc, flat["x_dddot"], flat["yaw"], flat["yaw_dot"]
        )
        if omega_cmd is not None:
            w_des = omega_cmd
        # 5. Att Control
        # R_des^T * R - R^T * R_des
        R_des_T = R_des.transpose(1, 2)
        R_T = R.transpose(1, 2)

        R_err_mat = torch.matmul(R_des_T, R) - torch.matmul(R_T, R_des)
        att_err = self.vee(0.5 * R_err_mat)  # (B, 3)

        w_err = state["w"] - w_des

        # Torque u2
        # (J @ att_err)
        term1 = -self.kp_att * att_err - self.kd_att * w_err
        J_term1 = torch.matmul(self.inertia, term1.unsqueeze(-1)).squeeze(-1)

        # w x Jw
        Jw = torch.matmul(self.inertia, state["w"].unsqueeze(-1)).squeeze(-1)
        w_cross_Jw = torch.linalg.cross(state["w"], Jw, dim=1)

        u2 = J_term1 + w_cross_Jw  # (B, 3)
        cmd_w = w_des - self.kp_att * att_err - self.kd_att * w_err
        # 6. Motor Allocation
        # TM: (4, B)
        TM = torch.stack([u1, u2[:, 0], u2[:, 1], u2[:, 2]], dim=0)

        # thrusts = InvAlloc @ TM
        rotor_thrusts = torch.matmul(self.TM_to_f, TM)  # (4, 4) @ (4, B) -> (4, B)

        # Speeds
        motor_speeds_sq = rotor_thrusts / self.k_eta
        motor_speeds = torch.sign(motor_speeds_sq) * torch.sqrt(
            torch.abs(motor_speeds_sq)
        )

        return {
            "cmd_motor_speeds": motor_speeds.T,  # (B, 4)
            "cmd_motor_thrusts": rotor_thrusts.T,  # (B, 4)
            "cmd_thrust": u1,  # (B,)
            "cmd_moment": u2,  # (B, 3)
            "cmd_q": q_des,  # (B, 4)
            "cmd_w": cmd_w,  # (B, 3)
            "cmd_v": -self.kp_vel * pos_err + flat["x_dot"],  # (B, 3)
            "cmd_acc": F_des / self.mass,  # (B, 3)
        }
