"""Telemetry over SSH: run agent.py on the board, read its JSON lines.

This is the default transport. It needs no DDS reachability (which WSL does not have) and
nothing installed on the board - agent.py is piped into python3 each time the link comes up,
so it cannot drift from this repo. Its stdin stays open afterwards: that is how the agent
learns which camera topics the open pages are showing.

The link reconnects on its own with a backoff, which is what makes "monitor the drone even if
it has been restarted" work: a reboot of the board, or of the flight controller node, just
shows up as a gap and then data again.
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import threading
import time

from .store import Store

AGENT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent.py")
# Reads the agent's source (its length in bytes on the first line, then the source) and runs
# it, leaving the rest of stdin for commands. `python3 -` would read stdin to EOF first.
BOOTSTRAP = ('import sys;b=sys.stdin.buffer;'
             'exec(compile(b.read(int(b.readline())),"agent.py","exec"))')
RECONNECT_DELAY = 3.0
MAX_LINE = 8 * 1024 * 1024          # a full-resolution colour frame in base64 fits


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
        self._stdin_lock = threading.Lock()

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
                  "export ROS_DOMAIN_ID=%d && exec python3 -u -c %s %s %s %s"
                  % (self.cfg.domain_id, _quote(BOOTSTRAP), self.cfg.drone_ns,
                     self.cfg.stereo_ns, self.cfg.vo_ns))
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
                    source = script.read()
                # Binary pipes: a text-mode stdin on Windows would turn \n into \r\n and the
                # byte count in the header would no longer match.
                self.process = subprocess.Popen(
                    self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
            except FileNotFoundError as exc:
                self.last_error = "ssh not available: %s" % exc
                self.on_event(self.last_error, "error")
                return

            with self._stdin_lock:
                try:
                    self.process.stdin.write(b"%d\n" % len(source) + source)
                    self.process.stdin.flush()
                except OSError:
                    pass                    # ssh already died; _read collects why
            self._send_watch()
            self._read(self.process)

            if self._stop.is_set():
                return
            if self.connected:
                self.connected = False
                self.on_event("telemetry link lost%s"
                              % (": " + self.last_error if self.last_error else ""), "warning")
            self._stop.wait(RECONNECT_DELAY)

    def _send_watch(self):
        """Tell the running agent which camera topics to subscribe to and send."""
        self._send({"type": "watch", "topics": self.store.watched()})

    def _send(self, command):
        """One command line to the running agent. False if no agent is running."""
        process = self.process
        if process is None or process.poll() is not None:
            return False                    # the next agent gets the watch list when it starts
        line = json.dumps(command) + "\n"
        with self._stdin_lock:
            try:
                process.stdin.write(line.encode("utf-8"))
                process.stdin.flush()
            except OSError:
                return False
        return True

    def _read(self, process):
        stdout = io.TextIOWrapper(process.stdout, encoding="utf-8", errors="replace")
        for line in stdout:
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
            elif kind == "attitude":
                message.pop("type")
                self.store.apply_attitude(message)
            elif kind == "frame":
                try:
                    self.store.apply_frame(message["topic"], base64.b64decode(message["jpeg"]))
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
        stderr = (process.stderr.read() or b"").decode("utf-8", "replace").strip() \
            if process.stderr else ""
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

    def latest_frame(self, topic, max_age=5.0):
        return self.store.latest_frame(topic, max_age)

    def frame_stats(self):
        return self.store.frame_stats()

    def watch(self, topic):
        if self.store.watch(topic):
            self._send_watch()

    def unwatch(self, topic):
        if self.store.unwatch(topic):
            self._send_watch()

    def plot(self, *names):
        return self.store.plot(*names)

    def attitude_plot(self):
        return self.store.attitude_plot()

    def zero_yaw(self):
        """The IMU attitude's current heading becomes yaw 0 (the filter runs in the agent)."""
        return self._send({"type": "zero_yaw"})

    def link_info(self):
        return {"transport": "ssh", "connected": self.connected,
                "attempts": self.attempts, "error": self.last_error,
                "silent_for": round(time.monotonic() - self.last_line, 1) if self.last_line
                              else None}


def _quote(text):
    return "'" + text.replace("'", "'\\''") + "'"
