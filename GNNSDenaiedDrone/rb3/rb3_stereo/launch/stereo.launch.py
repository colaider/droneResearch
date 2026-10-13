"""Stereo camera node of the RB3 Gen 2.

    ros2 launch rb3_stereo stereo.launch.py                       # config/stereo.yaml defaults
    ros2 launch rb3_stereo stereo.launch.py fps:=60
    ros2 launch rb3_stereo stereo.launch.py resolution:=1280x800 fps:=30
    ros2 launch rb3_stereo stereo.launch.py auto_exposure:=false exposure_us:=3000 gain:=2.0
Any other node parameter can be passed the same way (name:=value) or put in a params_file.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

SHARE = get_package_share_directory("rb3_stereo")
OVERRIDES = ("resolution", "fps", "left_camera", "flip", "genlock", "auto_exposure", "exposure_us", "gain",
             "ae_target", "ae_max_exposure_us", "ae_max_gain", "stamp", "left_calibration",
             "right_calibration", "nominal_vfov_deg", "baseline_m", "preview_fps", "preview_scale",
             "qos_reliable")


def node(context):
    params = [LaunchConfiguration("params_file").perform(context)]
    given = {k: LaunchConfiguration(k).perform(context) for k in OVERRIDES}
    params.append({k: yaml.safe_load(v) for k, v in given.items() if v != ""})
    return [Node(package="rb3_stereo", executable="stereo_camera", name="stereo_camera",
                 namespace=LaunchConfiguration("namespace"), parameters=params, output="screen",
                 respawn=True, respawn_delay=3.0)]


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument("namespace", default_value="stereo"),
         DeclareLaunchArgument("params_file", default_value=os.path.join(SHARE, "config", "stereo.yaml"))]
        + [DeclareLaunchArgument(k, default_value="", description=f"override parameter {k}") for k in OVERRIDES]
        + [OpaqueFunction(function=node)])
