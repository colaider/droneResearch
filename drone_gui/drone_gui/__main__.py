"""Entry point: python3 -m drone_gui  (with ROS 2 sourced).

    source /opt/ros/jazzy/setup.bash
    python3 -m drone_gui                       # then open http://localhost:8080
    python3 -m drone_gui --host 192.168.0.172 --port 8080
"""
from __future__ import annotations

import argparse
import sys
import webbrowser

from .board import Board
from .config import CONFIG
from .events import EventLog
from .server import State, serve
from .ssh_link import SshLink


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=CONFIG.host, help="the board's address")
    parser.add_argument("--port", type=int, default=CONFIG.port, help="port to serve on")
    parser.add_argument("--bind", default=CONFIG.bind,
                        help="interface to serve on (default loopback; 0.0.0.0 exposes the "
                             "process controls to the network)")
    parser.add_argument("--domain-id", type=int, default=CONFIG.domain_id)
    parser.add_argument("--ssh-key", default=CONFIG.ssh_key)
    parser.add_argument("--transport", choices=("ssh", "dds"), default="ssh",
                        help="ssh (default): run the collector on the board and read it back "
                             "over SSH - works from WSL and needs nothing installed there. "
                             "dds: subscribe directly, only usable where the board can route "
                             "to this machine.")
    parser.add_argument("--login", action="store_true",
                        help="first run on this machine: create an SSH key if there is none "
                             "and install it on the board (the board asks for its password), "
                             "then carry on and start the panel")
    parser.add_argument("--open", action="store_true", help="open a browser on start-up")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    CONFIG.host, CONFIG.port, CONFIG.bind = args.host, args.port, args.bind
    CONFIG.domain_id, CONFIG.ssh_key = args.domain_id, args.ssh_key

    if args.login:
        from .setup_key import ensure_access
        if not ensure_access(CONFIG):
            return 1
        print()

    events = EventLog(CONFIG.state_dir)
    events.add("panel started (domain %d, board %s)" % (CONFIG.domain_id, CONFIG.host))

    if args.transport == "dds":
        from .ros_link import RosLink
        ros = RosLink(CONFIG, on_event=events.add)
    else:
        ros = SshLink(CONFIG, on_event=events.add)
    try:
        ros.start()
    except Exception as exc:                            # noqa: BLE001 - report it plainly
        raise SystemExit("could not start the telemetry link: %s%s"
                         % (exc, "\nDid you source the ROS 2 setup?"
                            if args.transport == "dds" else ""))

    board = Board(CONFIG)
    state = State(CONFIG, ros, board, events)
    state.start()

    try:
        httpd = serve(CONFIG, state)
    except OSError as exc:
        # Almost always a panel that is still running - say so instead of a bare traceback.
        state.stop()
        ros.stop()
        raise SystemExit("\n".join([
            "cannot listen on %s:%d - %s" % (CONFIG.bind, CONFIG.port, exc),
            "Another panel is probably still running. Find it with:",
            "  Linux/WSL:  ss -ltnp | grep %d" % CONFIG.port,
            "  Windows:    netstat -ano | findstr :%d" % CONFIG.port,
            "or start this one on a different port: --port %d" % (CONFIG.port + 1),
        ]))
    url = "http://%s:%d" % ("localhost" if CONFIG.bind in ("127.0.0.1", "0.0.0.0") else CONFIG.bind,
                            CONFIG.port)
    print("drone control center on %s   (board %s, ROS_DOMAIN_ID %d, %s transport)"
          % (url, CONFIG.host, CONFIG.domain_id, args.transport), flush=True)
    if CONFIG.bind == "0.0.0.0":
        print("  note: bound to all interfaces - the process controls are reachable from the "
              "network")
    if args.open:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        state.stop()
        httpd.shutdown()
        ros.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
