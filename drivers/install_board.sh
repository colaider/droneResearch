#!/bin/bash
# One-time board install (rerun after every kernel update - the modules are built for one kernel):
#   sudo ./drivers/install_board.sh                 build + install drivers, I2C access, rb3-stereo-setup
#   sudo ./drivers/install_board.sh --enable-boot   ... and load the drivers automatically at every boot
# scripts/setup_rb3.sh runs this with --enable-boot.
# Requires the camss device tree from devicetree/dtb_flash.sh (CAM0A on camss) to be on the board already.
set -e
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)
KVER=$(uname -r)
KDIR=/lib/modules/$KVER/build
B=$(mktemp -d)
cp -r "$HERE"/ov9282 "$HERE"/camss "$HERE"/cam0b_bridge "$B"/

echo "== building kernel modules for $KVER"
make -s -C "$KDIR" M="$B/ov9282" modules
make -s -C "$KDIR" M="$B/camss" CONFIG_VIDEO_QCOM_CAMSS=m modules
make -s -C "$KDIR" M="$B/cam0b_bridge" modules

echo "== installing to /lib/modules/$KVER/updates/rb3_stereo"
DEST=/lib/modules/$KVER/updates/rb3_stereo
mkdir -p "$DEST"
install -m644 "$B/ov9282/ov9282.ko" "$B/camss/qcom-camss.ko" "$B/cam0b_bridge/cam0b_bridge.ko" "$DEST/"
rm -f "/lib/modules/$KVER/updates/ov9282.ko"            # older single-camera install
cat > /etc/modprobe.d/rb3-stereo.conf <<'EOF'
# rb3_stereo: OV9282 modules on the vision mezzanine run one MIPI lane; CSIPHY0 in 2+1 combo mode
options ov9282 lanes=1
options qcom_camss combo_phy=0 combo_alias=1
EOF
rm -f /etc/modprobe.d/ov9282-1lane.conf                 # superseded by rb3-stereo.conf
depmod -a "$KVER"
update-initramfs -u -k "$KVER" >/dev/null               # ov9282 can load from the initramfs

echo "== I2C access for ${SUDO_USER:-ubuntu} (log in again to take effect)"
usermod -aG i2c,video "${SUDO_USER:-ubuntu}"

install -m755 "$HERE/rb3-stereo-setup" /usr/local/sbin/rb3-stereo-setup
install -m644 "$HERE/rb3-stereo-setup.service" /etc/systemd/system/
systemctl daemon-reload
if [ "$1" = --enable-boot ]; then
  systemctl enable rb3-stereo-setup.service
  echo "== drivers load at boot (disable: sudo systemctl disable rb3-stereo-setup)"
else
  echo "== load the drivers after each boot with: sudo rb3-stereo-setup  (or rerun with --enable-boot)"
fi
rm -rf "$B"
echo "done"
