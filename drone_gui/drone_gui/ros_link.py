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

from .agent import (ATTITUDE_HZ, DISCOVER_SECONDS, FRAME_HZ, AttitudeFilter, camera_topics,
                    imu_attitude_input, raw_to_jpeg)
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
        self._cameras = {}                  # topic -> "jpeg" | "raw", from the graph
        self._camera_subs = {}              # topic -> subscription, for watched topics only
        self._raw_sent = {}                 # topic -> when a raw frame was last encoded
        self._camera_lock = threading.Lock()
        self._msg_types = None
        self.attitude = AttitudeFilter()        # the same filter the SSH agent runs
        self._attitude_sent = 0.0

    def start(self):
        import rclpy
        from geometry_msgs.msg import Vector3Stamped
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import (BatteryState, CompressedImage, FluidPressure, Image, Imu,
                                     MagneticField)
        from std_msgs.msg import Bool, Float32, String
        self._msg_types = {"jpeg": CompressedImage, "raw": Image}
        self._qos = qos_profile_sensor_data

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
        def on_imu(m):
            apply("imu", [[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z],
                          [m.linear_acceleration.x, m.linear_acceleration.y,
                           m.linear_acceleration.z]])
            self.attitude.update(*imu_attitude_input(m))
            now = time.monotonic()
            if now - self._attitude_sent >= 1.0 / ATTITUDE_HZ and self.attitude.updates:
                self._attitude_sent = now
                self.store.apply_attitude(self.attitude.state())

        # Depth 50: the attitude filter integrates every message (see agent.py).
        sub(Imu, drone + "/imu", on_imu,
            QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT))
        sub(MagneticField, drone + "/mag", lambda m: apply("mag", [
            m.magnetic_field.x * 1e6, m.magnetic_field.y * 1e6, m.magnetic_field.z * 1e6]),
            qos_profile_sensor_data)
        sub(FluidPressure, drone + "/pressure",
            lambda m: apply("pressure", m.fluid_pressure / 100.0), qos_profile_sensor_data)
        sub(String, stereo + "/status",
            lambda m: apply("stereo_status", json_or_none(m.data)), 10)
        # The control loop's estimates, including the camera's own x/y/z and velocity
        # (EstimatePublisher), and whatever publishes odometry - the same as the SSH agent.
        sub(String, self.cfg.vo_ns.rstrip("/") + "/status",
            lambda m: apply("vo_status", json_or_none(m.data)), 10)
        try:
            from nav_msgs.msg import Odometry

            def on_odom(m):
                p, t = m.pose.pose.position, m.twist.twist
                apply("odom", {"position": [p.x, p.y, p.z],
                               "velocity": [t.linear.x, t.linear.y, t.linear.z],
                               "yaw_rate": t.angular.z, "frame": m.header.frame_id})

            sub(Odometry, drone + "/odom", on_odom, 10)
        except ImportError:
            pass
        self.node.create_timer(DISCOVER_SECONDS, self._discover_cameras)

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

    # --- cameras: subscribed only while a page shows them -------------------
    def _discover_cameras(self):
        cameras = camera_topics(self.node)
        stats = self.store.frame_stats()
        self.store.apply({"camera_topics": [
            {"topic": topic, "kind": kind, "publishers": self.node.count_publishers(topic),
             "hz": round(stats[topic]["fps"], 1) if topic in stats else None}
            for topic, kind in sorted(cameras.items())]})
        with self._camera_lock:
            self._cameras = cameras
        self._update_camera_subscriptions()

    def _on_image(self, topic, kind, msg):
        if kind == "jpeg":
            self.store.apply_frame(topic, bytes(msg.data))
            return
        # Encoding costs CPU: only as often as the agent would send it.
        now = time.monotonic()
        if now - self._raw_sent.get(topic, 0.0) < 1.0 / FRAME_HZ:
            return
        self._raw_sent[topic] = now
        try:
            jpeg = raw_to_jpeg(msg)
        except ImportError:
            jpeg = None
        if jpeg is not None:
            self.store.apply_frame(topic, jpeg)

    def _update_camera_subscriptions(self):
        if self.node is None:
            return
        wanted = set(self.store.watched())
        with self._camera_lock:
            for topic in list(self._camera_subs):
                if topic not in wanted or topic not in self._cameras:
                    self.node.destroy_subscription(self._camera_subs.pop(topic))
            for topic in wanted - set(self._camera_subs):
                kind = self._cameras.get(topic)
                if kind is None:
                    continue                # not in the graph yet: retried on discovery
                self._camera_subs[topic] = self.node.create_subscription(
                    self._msg_types[kind], topic,
                    lambda m, t=topic, k=kind: self._on_image(t, k, m), self._qos)

    def watch(self, topic):
        if self.store.watch(topic):
            self._update_camera_subscriptions()

    def unwatch(self, topic):
        if self.store.unwatch(topic):
            self._update_camera_subscriptions()

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

    def latest_frame(self, topic, max_age=5.0):
        return self.store.latest_frame(topic, max_age)

    def frame_stats(self):
        return self.store.frame_stats()

    def plot(self, *names):
        return self.store.plot(*names)

    def attitude_plot(self):
        return self.store.attitude_plot()

    def zero_yaw(self):
        self.attitude.zero_yaw()
        return True

    def link_info(self):
        return {"transport": "dds", "connected": self.connected,
                "attempts": 1, "error": "", "silent_for": None}
