"""Start ALL the RB3 topics on one node: IMU, barometer, stereo cameras, command bridge.

    python3 -m rb3.start_rb3          # from inside GNNSDenaiedDrone/, with ROS 2 sourced

Each topic is an independent small class; a missing/failed one (no cameras, no flight controller)
is a warning, not fatal, so the rest still come up.
"""
import os
import sys

# rb3_control and rb3_stereo now live NESTED under this folder:
#   rb3/rb3_control/rb3_control/...   and   rb3/rb3_stereo/rb3_stereo/...
# Put each package root on sys.path so `import rb3_control` / `import rb3_stereo` (used by the
# topic classes) resolve to these moved copies. Must run BEFORE importing the topic classes.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _pkg in ("rb3_control", "rb3_stereo"):
    _root = os.path.join(_HERE, _pkg)
    if _root not in sys.path:
        sys.path.insert(0, _root)

import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

try:                                   # works both as a package and when run directly
    from .imu_topic import ImuTopic
    from .baro_topic import BaroTopic
    from .camera_topic import CameraTopic
    from .command_topic import CommandTopic
except ImportError:
    from imu_topic import ImuTopic
    from baro_topic import BaroTopic
    from camera_topic import CameraTopic
    from command_topic import CommandTopic


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("rb3")
    log = node.get_logger()
    topics = []

    def add(name, make):
        try:
            topics.append(make())
            log.info("started %s" % name)
        except Exception as exc:                       # noqa: BLE001 - report and continue
            log.warn("%s disabled: %s" % (name, exc))

    def sensors_info():
        """RB3 sensor-core streams for <ns>/status, merged from the IMU and baro topics."""
        info = {}
        for t in topics:
            if hasattr(t, "sensors"):
                info.update(t.sensors.info())
        return info

    add("ImuTopic", lambda: ImuTopic(node))
    add("BaroTopic", lambda: BaroTopic(node))
    add("CameraTopic", lambda: CameraTopic(node))
    add("CommandTopic", lambda: CommandTopic(node, sensors_info=sensors_info))

    log.info("RB3 up with: " + ", ".join(t.__class__.__name__ for t in topics))
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    # rclpy turns SIGINT (Ctrl-C, or `systemctl stop` of drone-rb3.service) into a context
    # shutdown; catching it lets the stop end cleanly instead of leaving the unit "failed".
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        for t in topics:
            try:
                t.stop()
            except Exception:
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
