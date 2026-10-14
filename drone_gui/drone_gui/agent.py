"""Telemetry collector that runs ON THE BOARD, fed to it over SSH.

The panel pipes this file to `python3 -u -c BOOTSTRAP` on the board (its length on one line,
then the source), so there is nothing to install or keep in sync there: the copy that runs is
always the one in this repo. The rest of stdin carries the panel's commands.

Why not subscribe to DDS from the panel's machine? Because the panel usually runs in WSL,
whose address is not reachable from the LAN (the board cannot even ping it), so the board can
never deliver samples to it. Discovery half-works, which makes this look like it should
function - the topic names show up but no publisher ever matches. Running the subscriber
where the publisher is sidesteps that completely, and the result travels back over the SSH
connection that is already working.

Output is newline-delimited JSON on stdout, one object per line:
    {"type": "hello",     "domain": 42, "topics": [...]}
    {"type": "telemetry", "values": {...}}      ~10 Hz, stale entries sent as null
    {"type": "frame",     "topic": "/stereo/preview/compressed", "jpeg": "<base64>"}
    {"type": "error",     "message": "..."}

Input, one JSON object per line on stdin:
    {"type": "watch", "topics": ["/stereo/preview/compressed", ...]}

Camera topics: every sensor_msgs CompressedImage (sent as published) and Image (encoded to
JPEG here) topic on the board is listed in telemetry as `camera_topics`, but only the ones the
panel watches are subscribed to and sent - images are by far the biggest thing on the link, so
nothing goes over it that no open page is showing.

Freshness is decided here, on the clock that took the samples, so the panel never has to
reason about clock skew between the two machines.
"""
from __future__ import annotations

import base64
import collections
import json
import math
import sys
import threading
import time

TELEMETRY_HZ = 10.0
FRAME_HZ = 10.0             # per camera topic
STALE = 2.0                 # a sample older than this is reported as null
DISCOVER_SECONDS = 2.0      # how often the topic graph is searched for camera topics
JPEG_QUALITY = 80           # for raw Image topics
CAMERA_TYPES = {"sensor_msgs/msg/CompressedImage": "jpeg", "sensor_msgs/msg/Image": "raw"}
RAW_CHANNELS = {"mono8": 1, "8uc1": 1, "bgr8": 3, "8uc3": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4}


def raw_to_jpeg(msg, quality=JPEG_QUALITY):
    """sensor_msgs/Image -> JPEG bytes, or None for an encoding this does not handle (16-bit
    depth, Bayer, YUV). Needs numpy and OpenCV, which the board has for the camera node."""
    import cv2
    import numpy as np

    encoding = msg.encoding.lower()
    channels = RAW_CHANNELS.get(encoding)
    if channels is None or msg.height == 0 or msg.width == 0:
        return None
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    image = rows[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    if encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif encoding == "rgba8":
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif encoding == "bgra8":
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpeg.tobytes() if ok else None


def camera_topics(node):
    """{topic: "jpeg" | "raw"} for every image topic in the ROS graph."""
    found = {}
    for name, types in node.get_topic_names_and_types():
        for type_name in types:
            if type_name in CAMERA_TYPES:
                found[name] = CAMERA_TYPES[type_name]
                break
    return found


def read_commands(stream, on_watch):
    """Panel -> agent commands, until stdin closes."""
    for line in stream:
        try:
            command = json.loads(line)
        except ValueError:
            continue
        if isinstance(command, dict) and command.get("type") == "watch":
            on_watch({t for t in command.get("topics") or () if isinstance(t, str)})


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    try:
        import rclpy
        from geometry_msgs.msg import Vector3Stamped
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import (BatteryState, CompressedImage, FluidPressure, Image, Imu,
                                     MagneticField)
        from std_msgs.msg import Bool, Float32, String
    except ImportError as exc:
        emit({"type": "error", "message": "rclpy not importable on the board: %s" % exc})
        return 1

    drone = (sys.argv[1] if len(sys.argv) > 1 else "/drone").rstrip("/")
    stereo = (sys.argv[2] if len(sys.argv) > 2 else "/stereo").rstrip("/")
    vo = (sys.argv[3] if len(sys.argv) > 3 else "/vo").rstrip("/")

    rclpy.init(args=None)
    node = Node("drone_gui_agent")
    latest = {}             # name -> (value, monotonic stamp)

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
    sub(String, vo + "/status", lambda m: put("vo_status", json_or_none(m.data)), 10)

    # Whatever eventually estimates position publishes it here; rb3_control already consumes
    # the same topic for its cmd_vel feedback. Nothing publishes it yet - see the README.
    try:
        from nav_msgs.msg import Odometry

        def on_odom(msg):
            p, t = msg.pose.pose.position, msg.twist.twist
            put("odom", {"position": [p.x, p.y, p.z],
                         "velocity": [t.linear.x, t.linear.y, t.linear.z],
                         "yaw_rate": t.angular.z,
                         "frame": msg.header.frame_id})

        sub(Odometry, drone + "/odom", on_odom, 10)
    except ImportError:
        pass

    # --- cameras: subscribed only while the panel watches them ---------------
    cameras = {}            # topic -> "jpeg" | "raw", from the graph
    publishers = {}         # topic -> publisher count
    camera_subs = {}        # topic -> subscription
    images = {}             # topic -> {"msg", "stamp", "sent", "arrivals"}
    unsupported = set()     # raw topics whose encoding raw_to_jpeg cannot handle (reported once)
    watch = {"topics": set()}
    watch_lock = threading.Lock()

    def on_watch(topics):
        with watch_lock:
            watch["topics"] = topics

    threading.Thread(target=read_commands, args=(sys.stdin.buffer, on_watch), daemon=True,
                     name="commands").start()

    def on_image(topic, msg):
        # Keep the message, not a copy of its bytes: at 30 fps most of them are never sent.
        entry = images.setdefault(topic, {"msg": None, "stamp": 0.0, "sent": 0.0,
                                          "arrivals": collections.deque(maxlen=20)})
        entry["msg"], entry["stamp"] = msg, time.monotonic()
        entry["arrivals"].append(entry["stamp"])

    def update_camera_subscriptions():
        with watch_lock:
            wanted = set(watch["topics"])
        for topic in list(camera_subs):
            if topic not in wanted or topic not in cameras:
                node.destroy_subscription(camera_subs.pop(topic))
                images.pop(topic, None)
        for topic in wanted - set(camera_subs):
            kind = cameras.get(topic)
            if kind is None:
                continue                    # not in the graph (yet): retried after discovery
            camera_subs[topic] = sub(CompressedImage if kind == "jpeg" else Image, topic,
                                     lambda m, t=topic: on_image(t, m), qos_profile_sensor_data)

    def source_hz(topic):
        entry = images.get(topic)
        if entry is None or len(entry["arrivals"]) < 2:
            return None
        times = entry["arrivals"]
        if time.monotonic() - times[-1] > STALE or times[-1] <= times[0]:
            return 0.0
        return round((len(times) - 1) / (times[-1] - times[0]), 1)

    def send_frames(now):
        for topic, entry in images.items():
            # Only send a frame once, and no faster than FRAME_HZ per topic.
            if entry["msg"] is None or entry["stamp"] <= entry["sent"] \
                    or (now - entry["sent"]) < frame_period:
                continue
            entry["sent"] = entry["stamp"]
            msg = entry["msg"]
            if cameras.get(topic) == "raw":
                try:
                    jpeg, reason = raw_to_jpeg(msg), "%s images" % msg.encoding
                except ImportError as exc:
                    jpeg, reason = None, "raw images without OpenCV (%s)" % exc
                if jpeg is None:
                    if topic not in unsupported:
                        unsupported.add(topic)
                        emit({"type": "error",
                              "message": "%s: cannot show %s" % (topic, reason)})
                    continue
            else:
                jpeg = bytes(msg.data)
            emit({"type": "frame", "topic": topic,
                  "jpeg": base64.b64encode(jpeg).decode("ascii")})

    emit({"type": "hello", "domain": __import__("os").environ.get("ROS_DOMAIN_ID"),
          "topics": [drone, stereo, vo]})

    names = ("status", "armed", "attitude", "altitude", "baro_altitude", "battery",
             "imu", "mag", "pressure", "stereo_status", "vo_status", "odom")
    telemetry_period, frame_period = 1.0 / TELEMETRY_HZ, 1.0 / FRAME_HZ
    next_telemetry = next_discover = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            now = time.monotonic()

            if now >= next_discover:
                next_discover = now + DISCOVER_SECONDS
                cameras = camera_topics(node)
                publishers = {topic: node.count_publishers(topic) for topic in cameras}
            update_camera_subscriptions()

            if now >= next_telemetry:
                next_telemetry = now + telemetry_period
                values = {name: fresh(name) for name in names}
                values["camera_topics"] = [
                    {"topic": topic, "kind": kind, "publishers": publishers.get(topic, 0),
                     "hz": source_hz(topic)}
                    for topic, kind in sorted(cameras.items())]
                emit({"type": "telemetry", "values": values})

            send_frames(now)
    except KeyboardInterrupt:
        pass
    except Exception as exc:                        # noqa: BLE001 - tell the panel why we died
        if type(exc).__name__ == "ExternalShutdownException":
            return 0                                # SIGTERM/SIGINT: a normal disconnect
        emit({"type": "error", "message": "%s: %s" % (type(exc).__name__, exc)})
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # Skip interpreter shutdown: the command thread is blocked reading stdin, and finalizing
    # under it aborts Python ("could not acquire lock for <stdin>") with a core dump.
    __import__("os")._exit(code or 0)
