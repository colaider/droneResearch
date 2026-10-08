#!/bin/bash
# One-time setup of the drone's RB3 Gen 2 for this repo. Safe to run again; rerun it after a kernel
# update (the drivers are built for one kernel). On the drone, from the repo checkout:
#   ~/droneResearch/scripts/setup_rb3.sh
# - builds + installs the camera drivers and loads them at every boot (drivers/install_board.sh --enable-boot)
# - links the ROS 2 package rb3_stereo into ~/ros2_ws and builds it with --symlink-install, so edits
#   in the repo take effect without rebuilding (new files or entry points still need a rebuild)
# - adds ROS 2 Jazzy, the workspace and ROS_DOMAIN_ID=42 to ~/.bashrc
# A freshly flashed board needs the camss device tree first: drivers/devicetree/dtb_flash.sh
set -e
REPO=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
WS=${RB3_STEREO_WS:-$HOME/ros2_ws}
DOMAIN=${ROS_DOMAIN_ID:-42}

echo "== camera drivers"
sudo "$REPO/drivers/install_board.sh" --enable-boot

echo "== ROS 2 workspace $WS"
mkdir -p "$WS/src"
if [ -e "$WS/src/rb3_stereo" ] && [ ! -L "$WS/src/rb3_stereo" ]; then
  backup="$HOME/rb3_stereo.backup-$(date +%Y%m%d-%H%M%S)"
  echo "moving the old copy of the package out of the workspace: $backup"
  mv "$WS/src/rb3_stereo" "$backup"
fi
ln -sfn "$REPO/rb3_stereo" "$WS/src/rb3_stereo"
# a build made without --symlink-install cannot be switched over in place
rm -rf "$WS/build/rb3_stereo" "$WS/install/rb3_stereo"
source /opt/ros/jazzy/setup.bash
(cd "$WS" && colcon build --base-paths src --symlink-install --packages-select rb3_stereo)

echo "== ~/.bashrc"
if ! grep -q '>>> droneResearch >>>' "$HOME/.bashrc"; then
  cat >> "$HOME/.bashrc" <<EOF
# >>> droneResearch >>>  (scripts/setup_rb3.sh)
source /opt/ros/jazzy/setup.bash
[ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"
export ROS_DOMAIN_ID=$DOMAIN
# <<< droneResearch <<<
EOF
  echo "added ROS 2 + workspace + ROS_DOMAIN_ID=$DOMAIN (open a new shell)"
else
  echo "already set up"
fi
echo "done. Camera node: $REPO/scripts/run_stereo.sh   VO test: cd $REPO && python3 realCameraTest/testRB3Cameras.py"
