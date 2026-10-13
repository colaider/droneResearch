"""sensor_msgs/CameraInfo from the stereo calibration, a ROS camera_calibration YAML, or a nominal
pinhole model."""
import math

import yaml
from sensor_msgs.msg import CameraInfo


def nominal(width, height, vfov_deg, baseline_m=0.0, right=False):
    """Pinhole model from a vertical field of view (same convention as the Genesis simulation)."""
    f = (height / 2) / math.tan(math.radians(vfov_deg) / 2)
    cx, cy = (width - 1) / 2, (height - 1) / 2
    info = CameraInfo(width=width, height=height, distortion_model="plumb_bob")
    info.d = [0.0] * 5
    info.k = [f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [f, 0.0, cx, -f * baseline_m if right else 0.0, 0.0, f, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return info


def rectified(rectifier, side):
    """Images rb3_stereo already rectified (calibration.Rectifier): no distortion, identity R."""
    P = rectifier.P1 if side == "left" else rectifier.P2
    info = CameraInfo(width=rectifier.size[0], height=rectifier.size[1], distortion_model="plumb_bob")
    info.d = [0.0] * 5
    info.k = [float(x) for x in P[:, :3].ravel()]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [float(x) for x in P.ravel()]
    return info


def load(path, width, height):
    """Load a camera_calibration YAML (ost.yaml format); scale it if it was made at the other
    resolution (640x400 is the 1280x800 view binned 2x2, same field of view)."""
    with open(path) as f:
        c = yaml.safe_load(f)
    s = width / c["image_width"]
    if abs(height / c["image_height"] - s) > 1e-6:
        raise ValueError(f"{path}: aspect ratio {c['image_width']}x{c['image_height']} does not match "
                         f"{width}x{height}")
    k = list(c["camera_matrix"]["data"])
    p = list(c["projection_matrix"]["data"])
    for i in (0, 2, 4, 5):
        k[i] *= s
    for i in (0, 2, 3, 5, 6):
        p[i] *= s
    info = CameraInfo(width=width, height=height, distortion_model=c.get("distortion_model", "plumb_bob"))
    info.d = [float(x) for x in c["distortion_coefficients"]["data"]]
    info.k = [float(x) for x in k]
    info.r = [float(x) for x in c["rectification_matrix"]["data"]]
    info.p = [float(x) for x in p]
    return info


def vfov_deg(info):
    return math.degrees(2 * math.atan(info.height / (2 * info.k[4])))


def baseline_m(info):
    """Baseline from the right camera's P (Tx, or Ty for cameras stacked vertically)."""
    return math.hypot(info.p[3], info.p[7]) / info.p[0] if info.p[0] else 0.0


def disparity_direction(info):
    """Image shift of the right view relative to the left (unit vector) from the right camera's P."""
    n = math.hypot(info.p[3], info.p[7])
    return (info.p[3] / n, info.p[7] / n) if n else (-1.0, 0.0)
