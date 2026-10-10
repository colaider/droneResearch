"""Launch the flight controller bridge in the /drone namespace.

    ros2 launch rb3_control control.launch.py
    ros2 launch rb3_control control.launch.py port:=/dev/ttyACM1 require_override:=false
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGS = {
    "namespace": "/drone",
    "port": "/dev/ttyACM0",
    "require_override": "true",
    "require_armed": "true",
}


def generate_launch_description():
    params = os.path.join(get_package_share_directory("rb3_control"), "config", "control.yaml")
    declarations = [DeclareLaunchArgument(name, default_value=value) for name, value in ARGS.items()]

    node = Node(
        package="rb3_control",
        executable="flight_controller",
        name="flight_controller",
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        parameters=[
            params,
            {
                "port": LaunchConfiguration("port"),
                "require_override": LaunchConfiguration("require_override"),
                "require_armed": LaunchConfiguration("require_armed"),
            },
        ],
    )
    return LaunchDescription(declarations + [node])
