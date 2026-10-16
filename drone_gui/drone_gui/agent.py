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
    {"type": "attitude",  "q": [w, x, y, z], "rpy": [...], ...}   ~30 Hz, see AttitudeFilter
    {"type": "frame",     "topic": "/stereo/preview/compressed", "jpeg": "<base64>"}
    {"type": "points",    "stamp": 12.3, "points": [[x, y, z], ...]}   each /vo/points message once
    {"type": "error",     "message": "..."}

Input, one JSON object per line on stdin:
    {"type": "watch", "topics": ["/stereo/preview/compressed", ...]}
    {"type": "zero_yaw"}                        the IMU attitude's current heading becomes yaw 0

Camera topics: every sensor_msgs CompressedImage (sent as published) and Image (encoded to
JPEG here) topic on the board is listed in telemetry as `camera_topics`, but only the ones the
panel watches are subscribed to and sent - images are by far the biggest thing on the link, so
nothing goes over it that no open page is showing.

Freshness is decided here, on the clock that took the samples, so the panel never has to
reason about clock skew between the two machines.

The IMU attitude is computed here too: the filter needs every /drone/imu message (~100 Hz),
and the link carries only a 10 Hz snapshot of it.
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
ATTITUDE_HZ = 30.0          # IMU attitude messages; the filter itself runs on every IMU message
CAMERA_TYPES = {"sensor_msgs/msg/CompressedImage": "jpeg", "sensor_msgs/msg/Image": "raw"}
RAW_CHANNELS = {"mono8": 1, "8uc1": 1, "bgr8": 3, "8uc3": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4}

G = 9.80665
# /drone/imu -> body frame: y and z negated, exactly as the control loop reads the topic
# (GNNSDenaiedDrone/controll/communication/Rb3ReadSend.py). That gives x forward, y right,
# z down, with the accelerometer reading +g on z when level - the frame the control loop's
# own attitude (sensors/IMU/IMU.py) is computed in, so the two agree on signs.
IMU_AXES = (1.0, -1.0, -1.0)


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def _qnormalize(q):
    n = math.sqrt(sum(v * v for v in q))
    return tuple(v / n for v in q) if n > 1e-12 else (1.0, 0.0, 0.0, 0.0)


def quaternion_from_euler(roll, pitch, yaw):
    """Body-to-world quaternion (w, x, y, z) for Z-Y-X Euler angles."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy)


def euler_from_quaternion(q):
    """(roll, pitch, yaw) in radians, Z-Y-X, from a body-to-world quaternion."""
    w, x, y, z = q
    return (math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
            math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class AttitudeFilter:
    """Orientation from the RB3's own gyro and accelerometer: a Mahony complementary filter on a
    quaternion, fed every /drone/imu message.

    The gyro is integrated (exactly, as a rotation) and the accelerometer pulls roll and pitch
    towards its gravity direction with gain KP, weighted down as |accel| leaves g so a hard
    manoeuvre does not drag the horizon. Frames: body as IMU_AXES; world x = the heading at
    start-up (or the last zero_yaw), y right of it, z down. So roll > 0 is right side down,
    pitch > 0 nose up, yaw > 0 nose turning right.

    Gyro bias: whenever the drone sits still for REST_WINDOW samples the mean gyro is the bias,
    and it is learned with time constant BIAS_TAU. Without that the yaw drifts by the raw bias
    (~1 deg/s on this board). Yaw still has no absolute reference - the magnetometer is not
    used, it is next to the motors - so it is relative and creeps slowly in flight.
    """

    KP = 1.0                # accel correction [rad/s per rad]: horizon time constant ~1 s
    KP_REST = 3.0           # sitting still: the accel is trustworthy, settle faster
    KP_START = 10.0         # first START_SECONDS: converge from the first reading's tilt
    START_SECONDS = 1.5
    ACCEL_BAND = 0.15 * G   # accel trusted fully at |a| = g, not at all beyond g +- this
    MAX_DT = 0.25           # a longer gap in the stream is not integrated across
    REST_WINDOW = 40        # samples (~0.4 s)
    REST_GYRO_STD = 0.01    # rad/s, per axis (noise at rest is ~0.003)
    REST_ACCEL_STD = 0.15   # m/s^2, of |a|
    # A slow steady turn has a small spread too: once a bias has been learned, a mean further
    # than this from it is movement, not bias.
    REST_BIAS_STEP = 0.03   # rad/s
    BIAS_TAU = 2.0          # s

    def __init__(self):
        self.q = None
        self.bias = [0.0, 0.0, 0.0]
        self.rate = [0.0, 0.0, 0.0]
        self.rest = False
        self.trust = 0.0
        self.updates = 0
        self._rest_seconds = 0.0            # time spent learning the bias so far
        self._start = None
        self._last = None
        self._stamps = collections.deque(maxlen=50)
        self._gyro_window = collections.deque(maxlen=self.REST_WINDOW)
        self._norm_window = collections.deque(maxlen=self.REST_WINDOW)
        self._zero_yaw = False

    def zero_yaw(self):
        """Make the current heading yaw 0, from any thread: applied on the next update."""
        self._zero_yaw = True

    def update(self, stamp, gyro, accel):
        """One /drone/imu message: stamp [s], gyro [rad/s] and accel [m/s^2] as published
        (accel None when the message has none)."""
        g = [s * v for s, v in zip(IMU_AXES, gyro)]
        a = [s * v for s, v in zip(IMU_AXES, accel)] if accel is not None else None
        norm = math.sqrt(sum(v * v for v in a)) if a is not None else 0.0
        if not all(math.isfinite(v) for v in g) or not math.isfinite(norm):
            return
        if self.q is None:
            if norm < 0.5 * G:
                return                          # need a gravity reading to start from
            self.q = quaternion_from_euler(math.atan2(a[1], a[2]),
                                           math.atan2(-a[0], math.hypot(a[1], a[2])), 0.0)
            self._start = self._last = stamp
            return

        dt, self._last = stamp - self._last, stamp
        self._stamps.append(stamp)

        # Still? Then the mean gyro over the window is the bias.
        self._gyro_window.append(g)
        self._norm_window.append(norm)
        self.rest = False
        if len(self._gyro_window) == self.REST_WINDOW and abs(norm - G) < 0.5:
            n = float(self.REST_WINDOW)
            mean = [sum(s[i] for s in self._gyro_window) / n for i in range(3)]
            gyro_std = max(math.sqrt(sum((s[i] - mean[i]) ** 2 for s in self._gyro_window) / n)
                           for i in range(3))
            norm_mean = sum(self._norm_window) / n
            norm_std = math.sqrt(sum((v - norm_mean) ** 2 for v in self._norm_window) / n)
            self.rest = (gyro_std < self.REST_GYRO_STD and norm_std < self.REST_ACCEL_STD
                         and (self._rest_seconds < 2 * self.BIAS_TAU
                              or max(abs(m - b) for m, b in zip(mean, self.bias))
                              < self.REST_BIAS_STEP))
            if self.rest and 0 < dt <= self.MAX_DT:
                # A running mean until BIAS_TAU of rest is in, so the first estimate (and the
                # yaw drift it stops) settles in a fraction of a second; a slow average after.
                k = min(1.0, dt / min(self.BIAS_TAU, self._rest_seconds + dt))
                self.bias = [b + (m - b) * k for b, m in zip(self.bias, mean)]
                self._rest_seconds += dt

        self.rate = [v - b for v, b in zip(g, self.bias)]
        w = list(self.rate)
        self.trust = 0.0
        if norm > 1e-6:
            self.trust = max(0.0, 1.0 - abs(norm - G) / self.ACCEL_BAND)
        if self.trust > 0:
            # Gravity direction the estimate expects in the body frame (world +z, down), and
            # the measured one: their cross product is the rotation that closes the gap.
            qw, qx, qy, qz = self.q
            v = (2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz)
            m = [c / norm for c in a]
            e = (m[1] * v[2] - m[2] * v[1], m[2] * v[0] - m[0] * v[2], m[0] * v[1] - m[1] * v[0])
            kp = (self.KP_START if stamp - self._start < self.START_SECONDS
                  else self.KP_REST if self.rest else self.KP) * self.trust
            w = [wi + kp * ei for wi, ei in zip(w, e)]

        if 0 < dt <= self.MAX_DT:
            rx, ry, rz = (wi * dt for wi in w)
            angle = math.sqrt(rx * rx + ry * ry + rz * rz)
            if angle > 1e-12:
                s = math.sin(angle / 2) / angle
                dq = (math.cos(angle / 2), rx * s, ry * s, rz * s)
            else:
                dq = (1.0, rx / 2, ry / 2, rz / 2)
            self.q = _qnormalize(_qmul(self.q, dq))
        if self._zero_yaw:
            self._zero_yaw = False
            yaw = euler_from_quaternion(self.q)[2]
            self.q = _qnormalize(_qmul((math.cos(-yaw / 2), 0.0, 0.0, math.sin(-yaw / 2)), self.q))
        self.updates += 1

    def hz(self):
        times = self._stamps
        if len(times) < 2 or times[-1] <= times[0]:
            return None
        return round((len(times) - 1) / (times[-1] - times[0]), 1)

    def state(self):
        """JSON-ready: quaternion, roll/pitch/yaw and bias-corrected rates [rad, rad/s]."""
        if self.q is None:
            return None
        return {"q": [round(v, 6) for v in self.q],
                "rpy": [round(v, 5) for v in euler_from_quaternion(self.q)],
                "rate": [round(v, 4) for v in self.rate],
                "bias": [round(v, 5) for v in self.bias],
                "rest": self.rest, "trust": round(self.trust, 2), "hz": self.hz()}


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


def read_commands(stream, on_watch, on_zero_yaw=None):
    """Panel -> agent commands, until stdin closes."""
    for line in stream:
        try:
            command = json.loads(line)
        except ValueError:
            continue
        if not isinstance(command, dict):
            continue
        if command.get("type") == "watch":
            on_watch({t for t in command.get("topics") or () if isinstance(t, str)})
        elif command.get("type") == "zero_yaw" and on_zero_yaw is not None:
            on_zero_yaw()


def imu_attitude_input(msg):
    """(stamp [s], gyro, accel or None) from a sensor_msgs/Imu, for AttitudeFilter.update.
    rb3's ImuTopic marks a message without an accel reading with covariance[0] = -1."""
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    gyro = (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z)
    accel = None if msg.linear_acceleration_covariance[0] < 0 else (
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z)
    return stamp, gyro, accel


def cloud_points(msg, digits=3):
    """sensor_msgs/PointCloud2 with float32 x, y, z fields -> [[x, y, z], ...] (finite only).
    Shared with ros_link, so both transports read /vo/points the same way."""
    import struct

    offsets = {f.name: f.offset for f in msg.fields if f.datatype == 7}     # 7 = FLOAT32
    if not all(axis in offsets for axis in "xyz"):
        return []
    order = "<" if not msg.is_bigendian else ">"
    data, step = bytes(msg.data), msg.point_step
    points = []
    for base in range(0, msg.width * msg.height * step, step):
        p = [struct.unpack_from(order + "f", data, base + offsets[a])[0] for a in "xyz"]
        if all(math.isfinite(v) for v in p):
            points.append([round(v, digits) for v in p])
    return points


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    try:
        import rclpy
        from geometry_msgs.msg import Vector3Stamped
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import (BatteryState, CompressedImage, FluidPressure, Image, Imu,
                                     MagneticField, PointCloud2)
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
    attitude = AttitudeFilter()

    def on_imu(m):
        put("imu", [[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z],
                    [m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z]])
        attitude.update(*imu_attitude_input(m))

    # Depth 50, not the sensor-data profile's 5: the filter integrates every message, and a
    # busy spin (camera frames) must not drop them.
    sub(Imu, drone + "/imu", on_imu,
        QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT))
    sub(MagneticField, drone + "/mag", lambda m: put("mag", [
        m.magnetic_field.x * 1e6, m.magnetic_field.y * 1e6, m.magnetic_field.z * 1e6,
    ]), qos_profile_sensor_data)
    sub(FluidPressure, drone + "/pressure",
        lambda m: put("pressure", m.fluid_pressure / 100.0), qos_profile_sensor_data)
    sub(String, stereo + "/status", lambda m: put("stereo_status", json_or_none(m.data)), 10)
    sub(String, vo + "/status", lambda m: put("vo_status", json_or_none(m.data)), 10)
    # The 3D reconstruction's depth points: sent once per message (not with the 10 Hz
    # telemetry snapshot, which would repeat the same few hundred points every tick).
    sub(PointCloud2, vo + "/points", lambda m: emit({
        "type": "points", "stamp": m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
        "points": cloud_points(m)}), 10)

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

    threading.Thread(target=read_commands, args=(sys.stdin.buffer, on_watch, attitude.zero_yaw),
                     daemon=True, name="commands").start()

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
    attitude_period = 1.0 / ATTITUDE_HZ
    next_telemetry = next_discover = next_attitude = time.monotonic()
    attitude_sent = 0
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

            if now >= next_attitude and attitude.updates != attitude_sent:
                next_attitude = now + attitude_period
                attitude_sent = attitude.updates
                emit(dict(attitude.state(), type="attitude"))

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
