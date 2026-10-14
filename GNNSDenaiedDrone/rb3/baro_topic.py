"""Barometer topic: the RB3's own barometer (sensor core), published as pressure + altitude.

<ns>/pressure       sensor_msgs/FluidPressure [Pa]
<ns>/baro_altitude  std_msgs/Float32 [m] above the start-up level (zeroed from the first samples)
<ns>/baro_altitude_stamped geometry_msgs/PointStamped, altitude in point.z with ROS stamp
"""
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import FluidPressure
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32

from rb3_control import see


class BaroTopic:
    def __init__(self, node, ns="/drone", rate_hz=25.0, publish_hz=25.0, sudo=True):
        self.node = node
        self.pub_pressure = node.create_publisher(FluidPressure, f"{ns}/pressure", qos_profile_sensor_data)
        self.pub_altitude = node.create_publisher(Float32, f"{ns}/baro_altitude", 10)
        self.pub_altitude_stamped = node.create_publisher(
            PointStamped, f"{ns}/baro_altitude_stamped", qos_profile_sensor_data
        )
        self.sensors = see.SensorSet({"pressure": rate_hz}, sudo=sudo, logger=node.get_logger())
        self.sensors.start()
        self.reference = None          # ground pressure, averaged from the first samples
        self._samples = []
        self.timer = node.create_timer(1.0 / publish_hz, self._publish)

    def _publish(self):
        stream = self.sensors.get("pressure")
        if stream is None:
            return
        for _host, _sensor_time, values in stream.drain():
            hpa = values[0]
            stamp = self.node.get_clock().now().to_msg()
            p = FluidPressure()
            p.header.stamp = stamp
            p.fluid_pressure = hpa * 100.0      # hPa -> Pa
            self.pub_pressure.publish(p)

            if self.reference is None:
                self._samples.append(hpa)
                if len(self._samples) >= 20:
                    self.reference = sum(self._samples) / len(self._samples)
                    self.node.get_logger().info("barometer zeroed at %.2f hPa" % self.reference)
                continue
            altitude = float(see.altitude_from_pressure(hpa, self.reference))
            self.pub_altitude.publish(Float32(data=altitude))
            stamped = PointStamped()
            stamped.header.stamp = stamp
            stamped.header.frame_id = "barometer"
            stamped.point.z = altitude
            self.pub_altitude_stamped.publish(stamped)

    def stop(self):
        self.sensors.stop()
