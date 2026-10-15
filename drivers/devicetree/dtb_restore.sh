#!/bin/bash
# Undo dtb_flash.sh: put the original firmware device tree back and re-enable Qualcomm CamX.
set -e
cd "$(dirname "$0")"
sudo dd if=dtb_a_backup.img of=/dev/disk/by-partlabel/dtb_a bs=1M conv=fsync status=none
sudo rm -f /etc/default/grub.d/90-camss.cfg /etc/default/grub.d/zz-camss.cfg /etc/modprobe.d/zz-no-camx.conf
sudo update-grub >/dev/null 2>&1
sudo update-initramfs -u >/dev/null 2>&1
sudo systemctl enable cam-server >/dev/null 2>&1 || true
echo "Original dtb_a restored, CamX re-enabled. Reboot to apply."
