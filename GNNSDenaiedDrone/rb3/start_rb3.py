"""Start ALL the RB3 topics on one node: IMU, barometer, stereo cameras, command bridge.

    python3 -m rb3.start_rb3          # from inside GNNSDenaiedDrone/, with ROS 2 sourced

Each topic is an independent small class; a missing/failed one (no cameras, no flight controller)
is a warning, not fatal, so the rest still come up.
"""
import rclpy
from rclpy.executors import MultiThreadedExecutor

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

    add("ImuTopic", lambda: ImuTopic(node))
    add("BaroTopic", lambda: BaroTopic(node))
    add("CameraTopic", lambda: CameraTopic(node))
    add("CommandTopic", lambda: CommandTopic(node))

    log.info("RB3 up with: " + ", ".join(t.__class__.__name__ for t in topics))
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
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
