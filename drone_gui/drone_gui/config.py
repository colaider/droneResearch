"""Where the drone is and how to reach it. Everything is overridable from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name, default):
    return os.environ.get(name, default)


@dataclass
class Config:
    # --- the board ---------------------------------------------------------
    host: str = _env("DRONE_HOST", "192.168.0.172")
    user: str = _env("DRONE_USER", "ubuntu")
    # Whichever candidate key this machine already has (see setup_key.CANDIDATES); if it has
    # none, this is where `--login` creates one.
    ssh_key: str = _env("DRONE_SSH_KEY", "")
    ssh_timeout: float = float(_env("DRONE_SSH_TIMEOUT", "6"))

    # --- ROS ---------------------------------------------------------------
    # Must match the board. The project uses 42 (see rb3_stereo/QUICKSTART.md); the value here
    # is only a default - the units installed on the board set the same one.
    domain_id: int = int(_env("ROS_DOMAIN_ID", "42"))
    drone_ns: str = _env("DRONE_NS", "/drone")
    stereo_ns: str = _env("STEREO_NS", "/stereo")
    # Visual-odometry status namespace (a VO node publishes its estimator output here).
    vo_ns: str = _env("VO_NS", "/vo")

    # --- this GUI ----------------------------------------------------------
    port: int = int(_env("DRONE_GUI_PORT", "8080"))
    # Loopback on purpose: the page can start and stop processes and reboot the board, and
    # WSL2 forwards localhost to Windows, so http://localhost:8080 in the Windows browser
    # works without exposing any of that to the LAN. Set DRONE_GUI_BIND=0.0.0.0 to reach it
    # from another machine, on a network you trust.
    bind: str = _env("DRONE_GUI_BIND", "127.0.0.1")
    history_seconds: float = float(_env("DRONE_GUI_HISTORY", "120"))
    state_dir: str = _env("DRONE_GUI_STATE",
                          os.path.join(os.path.expanduser("~"), ".local", "share", "drone_gui"))

    # Units the GUI can start and stop. The first two are installed by
    # service/install_board_units.sh; the rest ship with the board setup.
    units: list = field(default_factory=lambda: [
        ("drone-control", "Flight controller bridge (MSP over USB + RB3 sensors)", True),
        ("drone-stereo", "Stereo camera node (/stereo topics)", True),
        ("rb3-stereo-setup", "Camera kernel modules (camss combo + CAM0B)", False),
        ("rb3-usb-host", "Renesas USB host re-probe (brings up /dev/ttyACM0)", False),
        ("sscrpcd", "Qualcomm sensor core daemon (IMU/baro/mag)", False),
    ])

    def __post_init__(self):
        if not self.ssh_key:
            from .setup_key import default_key_path
            self.ssh_key = default_key_path()

    @property
    def ssh_target(self):
        return "%s@%s" % (self.user, self.host)


CONFIG = Config()
