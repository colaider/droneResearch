#!/bin/bash
# Start the control panel. Works on any machine with ROS 2 Jazzy and network access to the
# board - there is nothing to install, the panel is stdlib only.
#
#   ./run.sh                 # http://localhost:8080
#   ./run.sh --open          # ...and open a browser
#   ./run.sh --login         # first run here: make an SSH key and install it on the board
set -e
cd "$(dirname "$0")"
[ -z "$ROS_DISTRO" ] && source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
exec python3 -m drone_gui "$@"
