#!/bin/bash
# Start the stereo camera on the drone (loads the drivers first if needed):
#   ~/droneResearch/scripts/run_stereo.sh                      # 640x400 @ 30 fps
#   ~/droneResearch/scripts/run_stereo.sh fps:=60
#   ~/droneResearch/scripts/run_stereo.sh resolution:=1280x800 fps:=30
#   ~/droneResearch/scripts/run_stereo.sh auto_exposure:=false exposure_us:=3000 gain:=2.0
if [ "$(cat /sys/module/qcom_camss/parameters/combo_phy 2>/dev/null)" != 0 ] ||
   [ ! -d /sys/bus/platform/devices/ac4b000.cci ]; then
  sudo rb3-stereo-setup || exit 1
fi
source /opt/ros/jazzy/setup.bash
WS=${RB3_STEREO_WS:-$HOME/ros2_ws}
[ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
# bigger Fast DDS shared-memory segments: 1280x800 images are 1 MB (default segments: 512 KB)
export FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE:-$(ros2 pkg prefix rb3_stereo)/share/rb3_stereo/config/fastdds.xml}
# camera node + v4l2-ctl on the little A55 cores (0-3); the big A78 cores (4-7) stay free for VO
CPUS=${RB3_STEREO_CPUS:-0-3}
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID, camera on CPUs $CPUS"
exec taskset -c "$CPUS" ros2 launch rb3_stereo stereo.launch.py "$@"
