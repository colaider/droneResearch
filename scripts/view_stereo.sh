#!/bin/bash
# Open the drone's stereo camera from another Linux machine (or WSL2 with mirrored networking) on
# the same network and ROS_DOMAIN_ID. Needs ROS 2 (Jazzy) and python3-opencv; the package itself
# does not have to be installed - the viewer is a single file.
#   ./view_stereo.sh                                  preview, ROS_DOMAIN_ID 42
#   ./view_stereo.sh --domain 7 --fps 60
#   ./view_stereo.sh --exposure-us 3000 --gain 2      manual exposure
#   ./view_stereo.sh --auto-exposure on --ae-target 120
#   ./view_stereo.sh --help
source /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash 2>/dev/null
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
if ros2 pkg prefix rb3_stereo >/dev/null 2>&1; then
  exec ros2 run rb3_stereo viewer "$@"
fi
exec python3 "$(dirname "$(readlink -f "$0")")/../rb3_stereo/rb3_stereo/viewer.py" "$@"
