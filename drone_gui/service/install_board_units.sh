#!/bin/bash
# Install the units the control panel starts and stops. Run on the board, or from the
# repo on a workstation:  ./install_board_units.sh ubuntu@192.168.0.172
set -e
cd "$(dirname "$0")"
UNITS="drone-control.service drone-stereo.service drone-rb3.service drone-gnss-control.service"

# The panel's key (setup_key.py / --login) if this machine has it; DRONE_SSH_KEY overrides.
KEY="${DRONE_SSH_KEY:-$HOME/.ssh/rb3_drone}"
SSH_OPTS=()
[ -f "$KEY" ] && SSH_OPTS=(-i "$KEY")

if [ -n "$1" ]; then
  echo "installing on $1"
  scp "${SSH_OPTS[@]}" $UNITS "$1:/tmp/"
  # shellcheck disable=SC2029  # the unit list is ours, not user input
  ssh "${SSH_OPTS[@]}" "$1" "sudo mv $(for u in $UNITS; do printf '/tmp/%s ' "$u"; done) /etc/systemd/system/ && \
            sudo systemctl daemon-reload && \
            for u in $UNITS; do systemctl is-enabled \$u >/dev/null 2>&1 || true; done && \
            systemctl status --no-pager $UNITS 2>&1 | grep -E 'Loaded|●' || true"
  echo
  echo "installed, not enabled at boot. To start them on every boot:"
  echo "  ssh $1 sudo systemctl enable $UNITS"
  exit 0
fi

sudo cp $UNITS /etc/systemd/system/
sudo systemctl daemon-reload
echo "installed. 'sudo systemctl enable $UNITS' to start them at boot."
