import torch
import yaml


def compute_flatness(vel_cmd, acc_cmd, jerk_cmd, mass, g, max_roll_rate, device=None):
    G = torch.tensor([0.0, 0.0, -g], device=device)

    # --- Compute Commands ---

    f_cmd = mass * (acc_cmd - G)

    thrust_cmd = torch.norm(f_cmd)

    zb_cmd = f_cmd / (thrust_cmd + 1e-6)

    proj_cmd = torch.eye(3, device=device) - torch.outer(zb_cmd, zb_cmd)

    dz_cmd = proj_cmd @ jerk_cmd / thrust_cmd

    dz_mag_cmd = torch.norm(dz_cmd)

    omega_xy_scale_cmd = torch.clamp(max_roll_rate / (dz_mag_cmd + 1e-2), max=1.0)
    real_dz_cmd = dz_cmd * omega_xy_scale_cmd

    xb_cmd, yb_cmd, omega_cmd, dx_cmd, psi_info_cmd = compute_altitude(
        vel=vel_cmd, acc=acc_cmd, zb=zb_cmd, dz=real_dz_cmd
    )
    return thrust_cmd, omega_cmd


def compute_altitude(vel, acc, zb, dz):
    z0, z1, z2 = zb.unbind()

    # Flip for numerial stability
    if z2 < 0:
        zb = -zb
        z0, z1, z2 = zb.unbind()
        dz = -dz
        vel = -vel
        acc = -acc
        flip = -1
    else:
        flip = 1
    # flip = 1

    tilt_den_sqr = torch.clamp(2.0 * (1.0 + z2), min=1e-6)
    tilt_den = torch.sqrt(tilt_den_sqr)

    qw = 0.5 * tilt_den
    qx = -z1 / tilt_den
    qy = z0 / tilt_den

    x0 = 1.0 - 2.0 * qy * qy
    x1 = 2.0 * qx * qy
    x2 = -2.0 * qw * qy
    xb = torch.stack([x0, x1, x2])

    dz0, dz1, dz2 = dz.unbind()
    tilt_den_inv = tilt_den.reciprocal()
    dtilt_den = dz2 * tilt_den_inv
    dtilt0 = 0.5 * dtilt_den
    dtilt1 = -(dz1 - z1 * dtilt_den * tilt_den_inv) * tilt_den_inv
    dtilt2 = (dz0 - z0 * dtilt_den * tilt_den_inv) * tilt_den_inv

    dqw, dqx, dqy = dtilt0, dtilt1, dtilt2
    dx0 = -4.0 * dqy * qy
    dx1 = 2.0 * (dqx * qy + dqy * qx)
    dx2 = -2.0 * (dqw * qy + dqy * qw)
    dx = torch.stack([dx0, dx1, dx2])

    vel_norm = torch.linalg.norm(vel)
    vel_mask = vel_norm < 1e-3

    vel_dot_z_b = torch.sum(vel * zb)
    vel_x_b = vel - vel_dot_z_b.unsqueeze(0) * zb
    vel_x_b_norm = torch.linalg.norm(vel_x_b).clamp_min(1e-6)

    acc_dot_z_b = torch.sum(zb * acc) + torch.sum(vel * dz)
    acc_x_b = acc - zb * acc_dot_z_b.unsqueeze(0) - dz * vel_dot_z_b.unsqueeze(0)
    acc_x_b_norm = torch.linalg.norm(acc_x_b).clamp_min(1e-6)

    xb_cross_vel_xb = torch.linalg.cross(xb, vel_x_b)
    xb_cross_vel_xb_norm = torch.linalg.norm(xb_cross_vel_xb)
    sign = torch.sign((zb * xb_cross_vel_xb).sum()).clamp(min=-1.0, max=1.0)
    xb_dot_vel_xb = (xb * vel_x_b).sum()

    # Update compute
    psi = sign * torch.atan2(xb_cross_vel_xb_norm, xb_dot_vel_xb)
    dxb_cross_vel_xb = torch.linalg.cross(dx, vel_x_b) + torch.linalg.cross(xb, acc_x_b)
    dxb_cross_vel_xb_norm = (xb_cross_vel_xb * dxb_cross_vel_xb).sum() / (
        xb_cross_vel_xb_norm + 1e-6
    )
    dxb_dot_vel_xb = (vel_x_b * dx + xb * acc_x_b).sum()

    denom_dpsi = (xb_cross_vel_xb_norm**2 + xb_dot_vel_xb**2).clamp_min(1e-6)
    dpsi = (
        sign
        * (
            xb_dot_vel_xb * dxb_cross_vel_xb_norm
            - xb_cross_vel_xb_norm * dxb_dot_vel_xb
        )
        / denom_dpsi
    )

    if vel_mask.any():
        dpsi = torch.where(vel_mask, torch.zeros_like(dpsi), dpsi)

    # compute omega
    s_psi = torch.sin(psi)
    c_psi = torch.cos(psi)
    omg_term = dz2 / (z2 + 1.0)
    omega_x = dz0 * s_psi - dz1 * c_psi - (z0 * s_psi - z1 * c_psi) * omg_term
    omega_y = dz0 * c_psi + dz1 * s_psi - (z0 * c_psi + z1 * s_psi) * omg_term
    omega_z = (z1 * dz0 - z0 * dz1) / (z2 + 1.0) + dpsi

    psi_info = {
        "psi": psi,
        "dpsi": dpsi,
        "vel_x_b_norm": vel_x_b_norm,
        "sign": sign,
        "dxb_dot_vel_xb": dxb_dot_vel_xb,
        "xb_dot_vel_xb": xb_dot_vel_xb,
    }

    c_half_psi = torch.cos(0.5 * psi)
    s_half_psi = torch.sin(0.5 * psi)
    tilt0 = qw
    tilt1 = qx
    tilt2 = qy

    qw = tilt0 * c_half_psi
    qx = tilt1 * c_half_psi + tilt2 * s_half_psi
    qy = tilt2 * c_half_psi - tilt1 * s_half_psi
    qz = tilt0 * s_half_psi

    R00 = 1 - 2 * (qy * qy + qz * qz)
    R01 = 2 * (qx * qy - qw * qz)

    R10 = 2 * (qx * qy + qw * qz)
    R11 = 1 - 2 * (qx * qx + qz * qz)

    R20 = 2 * (qx * qz - qw * qy)
    R21 = 2 * (qy * qz + qw * qx)

    xb = torch.stack([R00, R10, R20], dim=-1)
    yb = torch.stack([R01, R11, R21], dim=-1)

    # xb X yb = zb, no need to flip yb, omega_y
    xb = flip * xb
    omega_x = flip * omega_x
    omega_z = flip * omega_z
    dx = flip * dx

    omega = torch.stack([omega_x, omega_y, omega_z])

    return xb, yb, omega, dx, psi_info
