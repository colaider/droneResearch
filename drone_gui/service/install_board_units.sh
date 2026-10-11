#!/bin/bash
# Install the two units the control panel starts and stops. Run on the board, or from the
# repo on a workstation:  ./install_board_units.sh ubuntu@192.168.1.243
set -e
cd "$(dirname "$0")"
UNITS="drone-control.service drone-stereo.service drone-vo.service"

if [ -n "$1" ]; then
  echo "installing on $1"
  scp $UNITS "$1:/tmp/"
  # shellcheck disable=SC2029  # the unit list is ours, not user input
  ssh "$1" "sudo mv $(for u in $UNITS; do printf '/tmp/%s ' "$u"; done) /etc/systemd/system/ && \
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
