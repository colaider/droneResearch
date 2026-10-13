"""IMU topic: the RB3's OWN accelerometer + gyro, from the sensor core (see.py).

Only the RB3 IMU is used here (not Betaflight's), published as sensor_msgs/Imu on <ns>/imu.
"""
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from rb3_control import see


class ImuTopic:
    def __init__(self, node, ns="/drone", gyro_hz=200.0, accel_hz=200.0,
                 publish_hz=100.0, frame="imu_link", sudo=True):
        self.node = node
        self.frame = frame
        self.pub = node.create_publisher(Imu, f"{ns}/imu", qos_profile_sensor_data)
        self.sensors = see.SensorSet({"gyro": gyro_hz, "accel": accel_hz},
                                     sudo=sudo, logger=node.get_logger())
        self.sensors.start()
        self.timer = node.create_timer(1.0 / publish_hz, self._publish)

    def _publish(self):
        gyro = self.sensors.get("gyro")
        accel = self.sensors.get("accel")
        g = gyro.peek() if gyro is not None else None
        a = accel.peek() if accel is not None else None
        if g is None or len(g[2]) < 3: return


        msg = Imu()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.frame

        gv = g[2]
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = gv[0], gv[1], gv[2]
        if a is not None and len(a[2]) >= 3:
            av = a[2]
            print(av)
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = av[0], av[1], av[2]
        else:
            msg.linear_acceleration_covariance[0] = -1.0   # no accel this cycle
        msg.orientation.w = 1.0
        msg.orientation_covariance[0] = -1.0               # no orientation estimate here
        self.pub.publish(msg)

    def stop(self):
        self.sensors.stop()
