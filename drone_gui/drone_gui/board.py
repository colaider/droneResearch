"""Process control and health for the board, over SSH.

Telemetry comes over DDS, but starting and stopping things does not, so the GUI shells out to
ssh. Nothing is installed on the board for this beyond two systemd units (see
service/install_board_units.sh); everything here is plain `systemctl` and reads from /sys and
/proc.

Every call has a timeout and returns a result object instead of raising, because the board
being off or rebooting is a normal state for this panel to display, not an error.
"""
from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass, field

# One process may hold /dev/ttyACM0: a second opener gets corrupted MSP replies and both die.
# The GUI therefore starts the flight controller through systemd, which keeps it to one.
PORT = "/dev/ttyACM0"

# Qualcomm's see_workhorse logs its protocol dumps to syslog from inside the flight
# controller's cgroup and buries the node's own messages. systemd 255 has no
# --exclude-identifier, so it is filtered out of the text output instead.
JOURNAL_NOISE = "see_workhorse"


@dataclass
class Result:
    ok: bool
    out: str = ""
    err: str = ""
    seconds: float = 0.0


@dataclass
class Health:
    online: bool = False
    reason: str = ""
    uptime: str = ""
    load: str = ""
    temp_c: float | None = None
    mem_percent: float | None = None
    disk_percent: float | None = None
    fc_port: bool = False
    fc_port_holder: str = ""
    see_streams: int = 0
    units: dict = field(default_factory=dict)       # name -> (active, sub_state, enabled)
    checked: float = 0.0


# Each probe prints a "###name" marker and then its value, so one SSH round trip covers the
# whole health panel. These are literal shell lines, not format strings: several contain
# their own % (awk formats, df --output=pcent), which rules out %-formatting the script.
PROBES = [
    'echo "###uptime"; uptime -p',
    'echo "###load"; cut -d" " -f1-3 /proc/loadavg',
    # thermal_zone0 on this SoC is not the CPU (it reads ~4 C), so take the hottest zone.
    'echo "###temp"; cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1',
    """echo "###mem"; free | awk '/Mem:/ {printf "%.0f\\n", 100*$3/$2}'""",
    'echo "###disk"; df --output=pcent / | tail -1 | tr -d " %"',
    'echo "###port"; test -e ' + PORT + ' && echo yes || echo no',
    'echo "###holder"; sudo fuser ' + PORT + ' 2>/dev/null | head -c 200; echo',
    # Each stream is a sudo parent plus the real process, so count distinct sensors. The
    # bracket keeps pgrep's own command line from matching the pattern and counting itself.
    '''echo "###see"; pgrep -fa "[s]ensor=" | grep -o "sensor=[a-z][a-z]*" | sort -u | wc -l''',
]


class Board:
    def __init__(self, config):
        self.cfg = config
        self.health = Health()

    # --- ssh --------------------------------------------------------------
    def run(self, command, timeout=None):
        timeout = timeout or self.cfg.ssh_timeout
        argv = [
            "ssh", "-i", self.cfg.ssh_key,
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=%d" % max(int(timeout) - 1, 2),
            self.cfg.ssh_target, command,
        ]
        start = time.monotonic()
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return Result(False, err="ssh timed out after %.0fs" % timeout,
                          seconds=time.monotonic() - start)
        except FileNotFoundError:
            return Result(False, err="ssh not found on this machine")
        return Result(done.returncode == 0, done.stdout.strip(), done.stderr.strip(),
                      time.monotonic() - start)

    # --- units ------------------------------------------------------------
    def unit_action(self, unit, action):
        """start / stop / restart a unit. Needs passwordless sudo, which the image has."""
        if action not in ("start", "stop", "restart"):
            raise ValueError(action)
        if not unit.replace("-", "").replace("_", "").replace(".", "").isalnum():
            raise ValueError("suspicious unit name: %r" % unit)
        return self.run("sudo systemctl %s %s" % (action, shlex.quote(unit)), timeout=25)

    def journal(self, unit, lines=60):
        command = ("journalctl -u %s -n %d --no-pager -o short-iso | grep -v ' %s' | tail -n %d"
                   % (shlex.quote(unit), int(lines) * 8, JOURNAL_NOISE, int(lines)))
        return self.run(command, timeout=15)

    # --- health -----------------------------------------------------------
    def poll(self):
        """One SSH round trip for everything the panel shows."""
        units = " ".join(name for name, _, _ in self.cfg.units)
        unit_probe = (
            'echo "###units"; for u in ' + units + '; do '
            'printf "%s %s %s %s\\n" "$u" "$(systemctl is-active $u)" '
            '"$(systemctl show -p SubState --value $u)" '
            '"$(systemctl is-enabled $u 2>/dev/null || echo -)"; done'
        )
        result = self.run("\n".join(PROBES + [unit_probe]),
                          timeout=max(self.cfg.ssh_timeout, 10))

        health = Health(checked=time.monotonic())
        if not result.ok:
            health.online = False
            health.reason = result.err.splitlines()[0] if result.err else "no answer"
            self.health = health
            return health

        sections = {}
        current = None
        for line in result.out.splitlines():
            if line.startswith("###"):
                current = line[3:]
                sections[current] = []
            elif current is not None:
                sections[current].append(line)

        def one(name, default=""):
            lines = [line for line in (sections.get(name) or []) if line.strip()]
            return lines[0].strip() if lines else default

        health.online = True
        health.uptime = one("uptime")
        health.load = one("load")
        try:
            health.temp_c = int(one("temp", "0")) / 1000.0 or None
        except ValueError:
            health.temp_c = None
        for key, attr in (("mem", "mem_percent"), ("disk", "disk_percent")):
            try:
                setattr(health, attr, float(one(key, "nan")))
            except ValueError:
                pass
        health.fc_port = one("port") == "yes"
        health.fc_port_holder = " ".join(sections.get("holder") or []).strip()
        try:
            health.see_streams = int(one("see", "0"))
        except ValueError:
            health.see_streams = 0
        for line in sections.get("units") or []:
            parts = line.split()
            if len(parts) >= 4:
                health.units[parts[0]] = (parts[1], parts[2], parts[3])
        self.health = health
        return health

    def reboot(self):
        return self.run("sudo systemctl reboot", timeout=10)
