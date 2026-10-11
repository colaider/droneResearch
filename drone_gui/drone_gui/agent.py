"""Telemetry collector that runs ON THE BOARD, fed to it over SSH.

The panel pipes this file into `python3 -u -` on the board, so there is nothing to install or
keep in sync there: the copy that runs is always the one in this repo.

Why not subscribe to DDS from the panel's machine? Because the panel usually runs in WSL,
whose address is not reachable from the LAN (the board cannot even ping it), so the board can
never deliver samples to it. Discovery half-works, which makes this look like it should
function - the topic names show up but no publisher ever matches. Running the subscriber
where the publisher is sidesteps that completely, and the result travels back over the SSH
connection that is already working.

Output is newline-delimited JSON on stdout, one object per line:
    {"type": "hello",     "domain": 42, "topics": [...]}
    {"type": "telemetry", "values": {...}}      ~10 Hz, stale entries sent as null
    {"type": "frame",     "jpeg": "<base64>"}   the stereo preview, as published
    {"type": "error",     "message": "..."}

Freshness is decided here, on the clock that took the samples, so the panel never has to
reason about clock skew between the two machines.
"""
from __future__ import annotations

import base64
import json
import math
import sys
import time

TELEMETRY_HZ = 10.0
FRAME_HZ = 10.0
STALE = 2.0                 # a sample older than this is reported as null


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    try:
        import rclpy
        from geometry_msgs.msg import Vector3Stamped
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import (BatteryState, CompressedImage, FluidPressure, Imu,
                                     MagneticField)
        from std_msgs.msg import Bool, Float32, String
    except ImportError as exc:
        emit({"type": "error", "message": "rclpy not importable on the board: %s" % exc})
        return 1

    drone = (sys.argv[1] if len(sys.argv) > 1 else "/drone").rstrip("/")
    stereo = (sys.argv[2] if len(sys.argv) > 2 else "/stereo").rstrip("/")

    rclpy.init(args=None)
    node = Node("drone_gui_agent")
    latest = {}             # name -> (value, monotonic stamp)
    frame = {"jpeg": None, "stamp": 0.0, "sent": 0.0}

    def put(name, value):
        latest[name] = (value, time.monotonic())

    def fresh(name):
        entry = latest.get(name)
        if entry is None or (time.monotonic() - entry[1]) > STALE:
            return None
        return entry[0]

    def json_or_none(text):
        try:
            return json.loads(text)
        except ValueError:
            return None

    sub = node.create_subscription
    sub(String, drone + "/status", lambda m: put("status", json_or_none(m.data)), 10)
    sub(Bool, drone + "/armed", lambda m: put("armed", m.data), 10)
    sub(Vector3Stamped, drone + "/attitude",
        lambda m: put("attitude", [m.vector.x, m.vector.y, m.vector.z]), 10)
    sub(Float32, drone + "/altitude", lambda m: put("altitude", m.data), 10)
    sub(Float32, drone + "/baro_altitude", lambda m: put("baro_altitude", m.data), 10)
    sub(BatteryState, drone + "/battery",
        lambda m: put("battery", [m.voltage, -m.current, bool(m.present)]), 10)
    sub(Imu, drone + "/imu", lambda m: put("imu", [
        [m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z],
        [m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z],
    ]), qos_profile_sensor_data)
    sub(MagneticField, drone + "/mag", lambda m: put("mag", [
        m.magnetic_field.x * 1e6, m.magnetic_field.y * 1e6, m.magnetic_field.z * 1e6,
    ]), qos_profile_sensor_data)
    sub(FluidPressure, drone + "/pressure",
        lambda m: put("pressure", m.fluid_pressure / 100.0), qos_profile_sensor_data)
    sub(String, stereo + "/status", lambda m: put("stereo_status", json_or_none(m.data)), 10)

    def on_frame(msg):
        frame["jpeg"] = bytes(msg.data)
        frame["stamp"] = time.monotonic()

    sub(CompressedImage, stereo + "/preview/compressed", on_frame, qos_profile_sensor_data)

    emit({"type": "hello", "domain": __import__("os").environ.get("ROS_DOMAIN_ID"),
          "topics": [drone, stereo]})

    names = ("status", "armed", "attitude", "altitude", "baro_altitude", "battery",
             "imu", "mag", "pressure", "stereo_status")
    telemetry_period, frame_period = 1.0 / TELEMETRY_HZ, 1.0 / FRAME_HZ
    next_telemetry = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            now = time.monotonic()

            if now >= next_telemetry:
                next_telemetry = now + telemetry_period
                emit({"type": "telemetry", "values": {name: fresh(name) for name in names}})

            # Only send a frame once, and no faster than FRAME_HZ.
            if frame["jpeg"] is not None and frame["stamp"] > frame["sent"] \
                    and (now - frame["sent"]) >= frame_period:
                frame["sent"] = frame["stamp"]
                emit({"type": "frame",
                      "jpeg": base64.b64encode(frame["jpeg"]).decode("ascii")})
    except KeyboardInterrupt:
        pass
    except Exception as exc:                        # noqa: BLE001 - tell the panel why we died
        emit({"type": "error", "message": "%s: %s" % (type(exc).__name__, exc)})
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
