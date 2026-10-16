"""The web server: stdlib only, no pip install anywhere.

Three kinds of traffic, each matched to a plain-HTTP mechanism that needs no framework:

  GET  /api/stream          Server-Sent Events, one JSON state object per tick. One-way push
                            is all a dashboard needs, and unlike WebSocket it is a few lines
                            of stdlib and reconnects on its own.
  GET  /camera/stream?topic=/stereo/preview/compressed
                            multipart/x-mixed-replace MJPEG of one camera topic, built from
                            JPEG frames - the browser does the decoding, so no image library
                            here either. While a stream is open its topic is "watched": only
                            watched topics are subscribed on the board and sent over the link.
  POST /api/unit/...        process control, forwarded to systemd on the board over SSH.
  GET  /api/tracks[/<id>?from=N], POST /api/tracks/split
                            position tracks, one per flight run (tracks.py).

Board polling runs in its own thread, so an SSH call that hangs slows the health panel down
instead of blocking page requests.
"""
from __future__ import annotations

import json
import math
import os
import re
import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from .store import CAMERA_SERIES
from .tracks import ODOM_MAX_AGE, VO_MAX_AGE, displayed_position

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
STREAM_HZ = 5.0
BOARD_POLL_SECONDS = 3.0
MJPEG_HZ = 12.0
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "application/javascript; charset=utf-8", ".svg": "image/svg+xml",
                 ".ico": "image/x-icon"}
TOPIC = re.compile(r"^(/[A-Za-z_][A-Za-z0-9_]*)+$")       # a fully qualified ROS topic name


def _degrees(value):
    return None if value is None else math.degrees(value)


def _cameras(config, ros):
    """Every camera topic the board has, merged with what this panel has received of it."""
    listed = {entry["topic"]: entry for entry in (ros.get("camera_topics", 10.0) or [])}
    stats = ros.frame_stats()
    topics = []
    for topic in sorted(set(listed) | set(stats)):
        entry, stat = listed.get(topic, {}), stats.get(topic)
        topics.append({
            "topic": topic,
            "kind": entry.get("kind"),
            "publishers": entry.get("publishers"),
            "source_hz": entry.get("hz"),           # as published, measured on the board
            "fps": round(stat["fps"], 1) if stat else None,    # as received here
            "age": round(stat["age"], 1) if stat else None,
            "frames": stat["frames"] if stat else 0,
        })
    return {"topics": topics, "default": config.stereo_ns.rstrip("/") + "/preview/compressed"}


class State:
    """Assembles what the page shows, from the ROS link and the board poller."""

    def __init__(self, config, ros, board, events, tracks=None):
        self.cfg = config
        self.ros = ros
        self.board = board
        self.events = events
        self.tracks = tracks
        self._last_online = None
        self._last_uptime = None
        self._last_units = {}
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._poll_board, daemon=True, name="board-poll")

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop.set()

    def _poll_board(self):
        while not self._stop.is_set():
            health = self.board.poll()

            if self._last_online is not None and health.online != self._last_online:
                self.events.add("board %s" % ("online" if health.online else "unreachable"),
                                "info" if health.online else "error")
            self._last_online = health.online

            # "up 0 minutes" after it was up longer means the board rebooted while we watched.
            if health.online and health.uptime:
                if self._last_uptime and health.uptime != self._last_uptime \
                        and "0 minutes" in health.uptime:
                    self.events.add("board rebooted", "warning")
                self._last_uptime = health.uptime

            for unit, (active, sub, _enabled) in health.units.items():
                was = self._last_units.get(unit)
                if was is not None and was != active:
                    level = "error" if active == "failed" else "info"
                    self.events.add("%s: %s -> %s%s" % (unit, was, active,
                                                        " (%s)" % sub if sub and active == "failed"
                                                        else ""), level)
                self._last_units[unit] = active

            self._stop.wait(BOARD_POLL_SECONDS)

    # --- the payload ------------------------------------------------------
    def snapshot(self):
        ros = self.ros
        status = ros.get("status") or {}
        status_age = ros.age("status")
        attitude = ros.get("attitude", 2.0)
        battery = ros.get("battery", 10.0)
        imu = ros.get("imu", 2.0)
        mag = ros.get("mag", 2.0)
        health = self.board.health

        plots = ros.plot("altitude", "baro_altitude", "battery", "roll", "pitch", *CAMERA_SERIES)

        # "live" means the flight controller node is publishing right now; the panel has to
        # distinguish that from "the board is up but nothing is running".
        live = status_age is not None and status_age < 2.0

        return {
            "now": time.time(),
            "link": {
                "live": live,
                "status_age": round(status_age, 1) if status_age is not None else None,
                "domain_id": self.cfg.domain_id,
                "host": self.cfg.host,
                "rates": {k: round(v, 1) for k, v in ros.rates().items()},
                **ros.link_info(),
            },
            "flight": {
                "armed": ros.get("armed", 5.0),
                "in_control": status.get("in_control") if live else None,
                "reason": status.get("reason") if live else None,
                "override_on": status.get("override_on") if live else None,
                "arming_flags": status.get("arming_flags") if live else None,
                "roll_deg": _degrees(attitude[0]) if attitude else None,
                "pitch_deg": _degrees(attitude[1]) if attitude else None,
                "yaw_deg": _degrees(attitude[2]) if attitude else None,
                "altitude_m": ros.get("altitude", 2.0),
                "baro_altitude_m": ros.get("baro_altitude", 2.0),
                "pressure_hpa": ros.get("pressure", 2.0),
                "battery_v": battery[0] if battery else None,
                "battery_a": battery[1] if battery else None,
                "sticks_sent": status.get("sticks_sent") if live else None,
                "sticks_fc": status.get("sticks_fc") if live else None,
                "gyro": imu[0] if imu else None,
                "accel": imu[1] if imu else None,
                "mag_ut": list(mag) if mag else None,
                "sensors": status.get("rb3_sensors") if live else None,
            },
            # Where the drone thinks it is. Z comes from the flight controller (fused
            # baro+IMU); there is no X/Y from it at all, since Betaflight only knows position
            # with GPS and none is configured. X/Y has to come from the stereo estimator.
            "position": {
                # x/y/z as shown, picked by the same function the path recorder uses
                "shown": displayed_position(ros),
                "odom": ros.get("odom", ODOM_MAX_AGE),
                "vo": ros.get("vo_status", VO_MAX_AGE),
                "fc_altitude_m": ros.get("altitude", 2.0),
                "baro_altitude_m": ros.get("baro_altitude", 2.0),
            },
            "cameras": _cameras(self.cfg, ros),
            "board": {
                "online": health.online,
                "reason": health.reason,
                "uptime": health.uptime,
                "load": health.load,
                "temp_c": round(health.temp_c, 1) if health.temp_c else None,
                "mem_percent": health.mem_percent,
                "disk_percent": health.disk_percent,
                "fc_port": health.fc_port,
                "fc_port_holder": health.fc_port_holder,
                "see_streams": health.see_streams,
                "age": round(time.monotonic() - health.checked, 1) if health.checked else None,
            },
            "units": [
                {"name": name, "description": description, "controllable": controllable,
                 "warning": self.cfg.unit_warnings.get(name),
                 "active": health.units.get(name, ("unknown", "", ""))[0],
                 "sub": health.units.get(name, ("", "", ""))[1],
                 "enabled": health.units.get(name, ("", "", ""))[2]}
                for name, description, controllable in self.cfg.units
            ],
            "plots": plots,
            # Which position track is being recorded and how long it is; the page fetches the
            # samples themselves from /api/tracks/<id> only when this count grows.
            "track": self.tracks.status() if self.tracks else None,
            # The page scales the x axis to this, not to the oldest sample it happens to
            # have, so the traces slide instead of compressing while the buffer fills.
            "window_seconds": self.cfg.history_seconds,
            "events": self.events.items()[:60],
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "drone-gui"
    protocol_version = "HTTP/1.1"
    state: State = None            # set on the server instance

    def log_message(self, fmt, *args):      # quiet: one line per request is just noise here
        pass

    # --- helpers ----------------------------------------------------------
    def _send(self, code, body=b"", content_type="text/plain; charset=utf-8", headers=()):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in headers:
            self.send_header(key, value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, payload, code=200):
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _static(self, path):
        name = os.path.basename(path) or "index.html"
        full = os.path.join(WEB_DIR, name)
        if not os.path.isfile(full):
            return self._send(404, b"not found")
        extension = os.path.splitext(name)[1]
        with open(full, "rb") as handle:
            body = handle.read()
        self._send(200, body, CONTENT_TYPES.get(extension, "application/octet-stream"))

    def _topic(self, query):
        """The ?topic= of a camera request (default: the stereo preview), or None if invalid."""
        topic = (parse_qs(query).get("topic") or [""])[0] \
            or self.state.cfg.stereo_ns.rstrip("/") + "/preview/compressed"
        return topic if TOPIC.match(topic) else None

    def _client_gone(self):
        """True once the browser has closed the connection. A stream on a topic that sends
        nothing never writes, so it would never find out from a failed write - and the topic
        would stay watched, and keep coming over the link, for good."""
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(readable) and not self.connection.recv(1, socket.MSG_PEEK)
        except (OSError, ValueError):
            return True

    # --- routes -----------------------------------------------------------
    def do_GET(self):                                  # noqa: N802 - BaseHTTPRequestHandler API
        url = urlsplit(self.path)
        path = url.path
        if path in ("/", "/index.html"):
            return self._static("index.html")
        if path in ("/app.css", "/app.js"):
            return self._static(path)
        if path == "/api/state":
            return self._json(self.state.snapshot())
        if path == "/api/stream":
            return self._sse()
        if path in ("/camera/stream", "/camera/frame.jpg"):
            topic = self._topic(url.query)
            if topic is None:
                return self._send(400, b"topic must be a ROS topic name like /stereo/preview/compressed")
            if path == "/camera/stream":
                return self._mjpeg(topic)
            frame = self.state.ros.latest_frame(topic)
            if frame is None:
                return self._send(503, b"no frame (only topics an open stream shows are received)")
            return self._send(200, frame, "image/jpeg")
        if path == "/api/tracks":
            if not self.state.tracks:
                return self._json({"live": None, "tracks": []})
            return self._json(self.state.tracks.list())
        if path.startswith("/api/tracks/") and self.state.tracks:
            try:
                start = int(parse_qs(url.query).get("from", ["0"])[0])
            except ValueError:
                start = 0
            track = self.state.tracks.get(path.rsplit("/", 1)[-1], max(start, 0))
            if track is None:
                return self._json({"ok": False, "error": "unknown track"}, 404)
            return self._json(track)
        if path.startswith("/api/journal/"):
            unit = path.rsplit("/", 1)[-1]
            if unit not in [name for name, _, _ in self.state.cfg.units]:
                return self._json({"ok": False, "error": "unknown unit"}, 404)
            result = self.state.board.journal(unit)
            return self._json({"ok": result.ok, "text": result.out or result.err})
        return self._send(404, b"not found")

    def do_POST(self):                                 # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)

        if path.startswith("/api/unit/"):
            try:
                _, _, _, unit, action = path.split("/", 4)
            except ValueError:
                return self._json({"ok": False, "error": "bad request"}, 400)
            allowed = {name: controllable for name, _, controllable in self.state.cfg.units}
            if unit not in allowed:
                return self._json({"ok": False, "error": "unknown unit"}, 404)
            if not allowed[unit] and action != "restart":
                return self._json({"ok": False,
                                   "error": "%s is a board service; only restart is offered"
                                            % unit}, 403)
            self.state.events.add("%s %s requested from the panel" % (action, unit))
            result = self.state.board.unit_action(unit, action)
            if not result.ok and "cancel" in (result.err or "").lower():
                # systemd cancels a queued job when a conflicting one is started. Saying so
                # beats showing the raw "Job for X canceled."
                result.err += (" - a conflicting unit was started, or the job was still "
                               "waiting on a dependency. drone-stereo and drone-vo share the "
                               "cameras and cannot both run.")
            if not result.ok:
                self.state.events.add("%s %s failed: %s" % (action, unit,
                                                            result.err or "see journal"), "error")
            return self._json({"ok": result.ok, "out": result.out, "err": result.err})

        if path == "/api/reboot":
            self.state.events.add("reboot requested from the panel", "warning")
            result = self.state.board.reboot()
            return self._json({"ok": result.ok, "err": result.err})

        if path == "/api/tracks/split" and self.state.tracks:
            self.state.tracks.split()
            return self._json({"ok": True})

        if path == "/api/note":
            return self._json({"ok": True})
        return self._send(404, b"not found")

    # --- streams ----------------------------------------------------------
    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        period = 1.0 / STREAM_HZ
        try:
            while True:
                payload = json.dumps(self.state.snapshot())
                self.wfile.write(b"data: " + payload.encode("utf-8") + b"\n\n")
                self.wfile.flush()
                time.sleep(period)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass        # the tab was closed or reloaded

    def _mjpeg(self, topic):
        boundary = "dronegui"
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=%s" % boundary)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        period = 1.0 / MJPEG_HZ
        last = None
        ros = self.state.ros
        ros.watch(topic)
        try:
            while not self._client_gone():
                frame = ros.latest_frame(topic)
                # Only push when the frame actually changed: frames arrive at most at the
                # agent's 10 Hz and resending the same bytes faster buys nothing.
                if frame is not None and frame is not last:
                    self.wfile.write(("--%s\r\n" % boundary).encode())
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(("Content-Length: %d\r\n\r\n" % len(frame)).encode())
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    last = frame
                time.sleep(period)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.timeout):
            pass
        finally:
            ros.unwatch(topic)
            self.close_connection = True


class Server(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows, SO_REUSEADDR lets a second process bind a port that is already being
    # listened on, so two panels would both "start" and split the requests between them.
    # Linux reports EADDRINUSE either way; turning it off here makes both behave the same.
    allow_reuse_address = os.name != "nt"


def serve(config, state):
    Handler.state = state
    return Server((config.bind, config.port), Handler)
