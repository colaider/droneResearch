"""Telemetry straight from DDS, for when the panel can actually reach the drone's topics.

This is the alternative to ssh_link: use it with `--transport dds` when the panel runs
somewhere the board can send to - on the board itself, or on a native Linux machine on the
same LAN. It does NOT work from WSL: the board cannot route to WSL's address, so discovery
half-completes (topic names appear, no publisher ever matches) and no sample arrives.

It fills the same Store and exposes the same methods as SshLink, so the server does not care.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time

from .store import Store


class RosLink:
    def __init__(self, config, on_event=None):
        self.cfg = config
        self.on_event = on_event or (lambda *a, **k: None)
        self.store = Store(config.history_seconds, on_event=self.on_event)
        self.node = None
        self.executor = None
        self.thread = None
        self.connected = False

    def start(self):
        import rclpy
        from geometry_msgs.msg import Vector3Stamped
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import (BatteryState, CompressedImage, FluidPressure, Imu,
                                     MagneticField)
        from std_msgs.msg import Bool, Float32, String

        os.environ.setdefault("ROS_DOMAIN_ID", str(self.cfg.domain_id))
        rclpy.init(args=None)
        self.node = Node("drone_gui")
        drone = self.cfg.drone_ns.rstrip("/")
        stereo = self.cfg.stereo_ns.rstrip("/")

        def apply(name, value):
            self.connected = True
            self.store.apply({name: value})

        def json_or_none(text):
            try:
                return json.loads(text)
            except ValueError:
                return None

        sub = self.node.create_subscription
        sub(String, drone + "/status", lambda m: apply("status", json_or_none(m.data)), 10)
        sub(Bool, drone + "/armed", lambda m: apply("armed", m.data), 10)
        sub(Vector3Stamped, drone + "/attitude",
            lambda m: apply("attitude", [m.vector.x, m.vector.y, m.vector.z]), 10)
        sub(Float32, drone + "/altitude", lambda m: apply("altitude", m.data), 10)
        sub(Float32, drone + "/baro_altitude", lambda m: apply("baro_altitude", m.data), 10)
        sub(BatteryState, drone + "/battery",
            lambda m: apply("battery", [m.voltage, -m.current, bool(m.present)]), 10)
        sub(Imu, drone + "/imu", lambda m: apply("imu", [
            [m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z],
            [m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z]]),
            qos_profile_sensor_data)
        sub(MagneticField, drone + "/mag", lambda m: apply("mag", [
            m.magnetic_field.x * 1e6, m.magnetic_field.y * 1e6, m.magnetic_field.z * 1e6]),
            qos_profile_sensor_data)
        sub(FluidPressure, drone + "/pressure",
            lambda m: apply("pressure", m.fluid_pressure / 100.0), qos_profile_sensor_data)
        sub(String, stereo + "/status",
            lambda m: apply("stereo_status", json_or_none(m.data)), 10)
        sub(CompressedImage, stereo + "/preview/compressed",
            lambda m: self.store.apply_frame(bytes(m.data)), qos_profile_sensor_data)

        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self._spin, daemon=True, name="ros-link")
        self.thread.start()
        self.on_event("telemetry link: DDS, domain %d" % self.cfg.domain_id)

    def _spin(self):
        try:
            self.executor.spin()
        except Exception:                   # noqa: BLE001 - the panel must outlive it
            pass

    def stop(self):
        import rclpy
        if self.executor is not None:
            self.executor.shutdown()
        if self.node is not None:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    # --- the Store API the server uses ------------------------------------
    def get(self, name, max_age=None):
        return self.store.get(name) if max_age is None else self.store.get(name, max_age)

    def age(self, name):
        return self.store.age(name)

    def rates(self):
        return self.store.rates()

    def latest_frame(self, max_age=5.0):
        return self.store.latest_frame(max_age)

    def frame_age(self):
        return self.store.frame_age()

    @property
    def frame_count(self):
        return self.store.frame_count

    def plot(self, *names):
        return self.store.plot(*names)

    def link_info(self):
        return {"transport": "dds", "connected": self.connected,
                "attempts": 1, "error": "", "silent_for": None}
