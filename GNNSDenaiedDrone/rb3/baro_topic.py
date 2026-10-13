"""Barometer topic: the RB3's own barometer (sensor core), published as pressure + altitude.

<ns>/pressure       sensor_msgs/FluidPressure [Pa]
<ns>/baro_altitude  std_msgs/Float32 [m] above the start-up level (zeroed from the first samples)
"""
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import FluidPressure
from std_msgs.msg import Float32

from rb3_control import see


class BaroTopic:
    def __init__(self, node, ns="/drone", rate_hz=25.0, publish_hz=25.0, sudo=True):
        self.node = node
        self.pub_pressure = node.create_publisher(FluidPressure, f"{ns}/pressure", qos_profile_sensor_data)
        self.pub_altitude = node.create_publisher(Float32, f"{ns}/baro_altitude", 10)
        self.sensors = see.SensorSet({"pressure": rate_hz}, sudo=sudo, logger=node.get_logger())
        self.sensors.start()
        self.reference = None          # ground pressure, averaged from the first samples
        self._samples = []
        self.timer = node.create_timer(1.0 / publish_hz, self._publish)

    def _publish(self):
        stream = self.sensors.get("pressure")
        if stream is None:
            return
        for _host, _ssc, values in stream.drain():
            hpa = values[0]
            p = FluidPressure()
            p.header.stamp = self.node.get_clock().now().to_msg()
            p.fluid_pressure = hpa * 100.0      # hPa -> Pa
            self.pub_pressure.publish(p)

            if self.reference is None:
                self._samples.append(hpa)
                if len(self._samples) >= 20:
                    self.reference = sum(self._samples) / len(self._samples)
                    self.node.get_logger().info("barometer zeroed at %.2f hPa" % self.reference)
                continue
            self.pub_altitude.publish(
                Float32(data=float(see.altitude_from_pressure(hpa, self.reference))))

    def stop(self):
        self.sensors.stop()
