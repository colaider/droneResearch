"""Telemetry over SSH: run agent.py on the board, read its JSON lines.

This is the default transport. It needs no DDS reachability (which WSL does not have) and
nothing installed on the board - agent.py is piped into `python3 -u -` each time the link
comes up, so it cannot drift from this repo.

The link reconnects on its own with a backoff, which is what makes "monitor the drone even if
it has been restarted" work: a reboot of the board, or of the flight controller node, just
shows up as a gap and then data again.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
import time

from .store import Store

AGENT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent.py")
RECONNECT_DELAY = 3.0
MAX_LINE = 4 * 1024 * 1024          # a preview frame in base64 is well under this


class SshLink:
    """Keeps one SSH-hosted agent running and feeds its output into a Store."""

    def __init__(self, config, on_event=None):
        self.cfg = config
        self.on_event = on_event or (lambda *a, **k: None)
        self.store = Store(config.history_seconds, on_event=self.on_event)
        self.connected = False
        self.last_line = 0.0
        self.attempts = 0
        self.last_error = ""
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="ssh-link")
        self.process = None

    # --- lifecycle --------------------------------------------------------
    def start(self):
        self.thread.start()

    def stop(self):
        self._stop.set()
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()

    @property
    def command(self):
        remote = ("source /opt/ros/jazzy/setup.bash && "
                  "source ~/ros2_ws/install/setup.bash 2>/dev/null; "
                  "export ROS_DOMAIN_ID=%d && exec python3 -u - %s %s %s"
                  % (self.cfg.domain_id, self.cfg.drone_ns, self.cfg.stereo_ns,
                     self.cfg.vo_ns))
        return [
            "ssh", "-i", self.cfg.ssh_key,
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=8",
            # Notice a dead link reasonably fast instead of hanging on a half-open socket.
            "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=2",
            self.cfg.ssh_target, "bash -lc %s" % _quote(remote),
        ]

    def _run(self):
        while not self._stop.is_set():
            self.attempts += 1
            try:
                with open(AGENT, "rb") as script:
                    self.process = subprocess.Popen(
                        self.command, stdin=script, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, bufsize=1, text=True)
            except FileNotFoundError as exc:
                self.last_error = "ssh not available: %s" % exc
                self.on_event(self.last_error, "error")
                return

            self._read(self.process)

            if self._stop.is_set():
                return
            if self.connected:
                self.connected = False
                self.on_event("telemetry link lost%s"
                              % (": " + self.last_error if self.last_error else ""), "warning")
            self._stop.wait(RECONNECT_DELAY)

    def _read(self, process):
        for line in process.stdout:
            if self._stop.is_set():
                return
            line = line.strip()
            if not line or len(line) > MAX_LINE:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                continue            # a stray print from a sourced setup script

            kind = message.get("type")
            if kind == "telemetry":
                self.store.apply(message.get("values") or {})
            elif kind == "frame":
                try:
                    self.store.apply_frame(base64.b64decode(message["jpeg"]))
                except (KeyError, ValueError):
                    pass
            elif kind == "hello":
                if not self.connected:
                    self.on_event("telemetry link up (agent on %s)" % self.cfg.host)
                self.connected = True
                self.last_error = ""
            elif kind == "error":
                self.last_error = message.get("message", "")
                self.on_event("agent: %s" % self.last_error, "error")

            self.last_line = time.monotonic()

        # stdout closed: collect why, so the panel can show something useful.
        stderr = (process.stderr.read() or "").strip() if process.stderr else ""
        if stderr:
            self.last_error = stderr.splitlines()[-1][:200]
        process.wait()

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
        return {"transport": "ssh", "connected": self.connected,
                "attempts": self.attempts, "error": self.last_error,
                "silent_for": round(time.monotonic() - self.last_line, 1) if self.last_line
                              else None}


def _quote(text):
    return "'" + text.replace("'", "'\\''") + "'"
