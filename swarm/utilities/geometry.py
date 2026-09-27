"""NumPy frame helpers. World/body are right-handed; quaternions are wxyz."""
import numpy as np


def as_numpy(value):
    """Return a floating NumPy copy of value, preserving its shape.

    value can be array-like or a Torch tensor. Tensors are detached and moved
    to CPU; this conversion does not preserve gradients or modify the input.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float).copy()


def rotation(rpy):
    """Return a (3, 3) body-to-world rotation from rpy=[roll, pitch, yaw] radians.

    Uses Rz(yaw) @ Ry(pitch) @ Rx(roll). For a (3,) column-style vector,
    world = R @ body; body = R.T @ world. For arrays of row vectors,
    world_rows = body_rows @ R.T.
    """
    roll, pitch, yaw = rpy
    cr, cp, cy = np.cos(rpy)
    sr, sp, sy = np.sin(rpy)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def quat_to_rpy(q):
    """Convert a nonzero (4,) quaternion q=[w, x, y, z] to (3,) Euler radians.

    Normalizes q internally. Output order is roll, pitch, yaw using the same
    convention as rotation(). Euler angles remain singular at pitch +/-pi/2.
    """
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                     np.arcsin(np.clip(2*(w*y-z*x), -1, 1)),
                     np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))])


def rpy_to_quat(rpy):
    """Convert (3,) [roll, pitch, yaw] in radians to a unit (4,) [w, x, y, z] quaternion."""
    cr, cp, cy = np.cos(np.asarray(rpy)/2)
    sr, sp, sy = np.sin(np.asarray(rpy)/2)
    return np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                     cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])


def integrate_attitude(rpy, body_rates, dt):
    """Advance orientation using body-frame gyro rates held constant over dt.

    Args:
        rpy: Current (3,) [roll, pitch, yaw] relative to world, in radians.
        body_rates: (3,) [p, q, r] gyro readings about body axes, in rad/s.
            These are not generally equal to Euler-angle derivatives.
        dt: Integration interval in seconds.

    Returns:
        New (3,) roll, pitch, yaw in radians. A quaternion increment is
        multiplied on the right because rates are expressed in body axes.
        The sinc expression has a finite zero-rate limit without division by zero.
    """
    q = rpy_to_quat(rpy)
    angle = np.linalg.norm(body_rates) * dt
    dq = np.r_[np.cos(angle/2), body_rates * dt * 0.5 * np.sinc(angle/(2*np.pi))]
    w, v = q[0], q[1:]
    d, u = dq[0], dq[1:]
    return quat_to_rpy(np.r_[w*d-v@u, w*u+d*v+np.cross(v, u)])
