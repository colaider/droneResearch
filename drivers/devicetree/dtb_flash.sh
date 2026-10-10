#!/bin/bash
# Bake the OV9282 overlay into the firmware's own device tree (dtb_a partition).
# GRUB `devicetree` crashes the Qualcomm UEFI and kexec resets the board, so this is the remaining route.
# Safeguards: full backup first, all edits are done on a COPY, and it asks before writing anything.
# Run from the folder that contains ov9282-cam0a.dtso.
set -e
cd "$(dirname "$0")"
PART=/dev/disk/by-partlabel/dtb_a
sudo apt-get install -y device-tree-compiler >/dev/null

if [ ! -f dtb_a_backup.img ]; then
  sudo dd if=$PART of=dtb_a_backup.img bs=1M status=none
  sudo chown "$USER" dtb_a_backup.img
fi
echo "backup: dtb_a_backup.img ($(stat -c %s dtb_a_backup.img) bytes)"

cp dtb_a_backup.img dtb_a_new.img
mkdir -p dtbmnt
sudo mount -o loop dtb_a_new.img dtbmnt || { echo "!! dtb_a is not a mountable image:"; file dtb_a_backup.img; exit 1; }
trap 'sudo umount dtbmnt 2>/dev/null || true' EXIT
echo "=== files in dtb_a"; ls -la dtbmnt
DTB=$(ls -S dtbmnt/*.dtb 2>/dev/null | head -n1)
[ -z "$DTB" ] && { echo "!! no .dtb file in the partition - send me this output"; exit 1; }
echo "using $DTB"

dtc -@ -I dts -O dtb -o ov9282-cam0a.dtbo ov9282-cam0a.dtso
sudo cat /sys/firmware/fdt > live.dtb
cp "$DTB" orig_multi.dtb

# The file is several DTBs back to back; patch the one(s) the firmware picks for this board.
python3 - orig_multi.dtb new_multi.dtb ov9282-cam0a.dtbo <<'PY'
import struct, subprocess, sys
src, dst, ovl = sys.argv[1:4]
def get(path, prop, t='x'):
    r = subprocess.run(['fdtget', '-t', t, path, '/', prop], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None
live_ids = (get('live.dtb', 'qcom,msm-id'), get('live.dtb', 'qcom,board-id'))
live_model = get('live.dtb', 'model', 's')
data = open(src, 'rb').read()
blobs, off = [], 0
while off + 8 <= len(data):
    magic, size = struct.unpack('>II', data[off:off + 8])
    if magic != 0xd00dfeed:
        break
    blobs.append(data[off:off + size]); off += size
tail = data[off:]
print(f'{len(blobs)} DTBs in file; running board: {live_model} ids={live_ids}')
paths = []
for i, b in enumerate(blobs):
    p = f'/tmp/dtb_{i}.dtb'; open(p, 'wb').write(b); paths.append(p)
sel = [i for i, p in enumerate(paths) if (get(p, 'qcom,msm-id'), get(p, 'qcom,board-id')) == live_ids]
if not sel:
    sel = [i for i, p in enumerate(paths) if get(p, 'model', 's') == live_model]
    print('no exact id match, falling back to model match')
for i, p in enumerate(paths):
    print(f'  [{i}] {len(blobs[i]):7d} B {"PATCH" if i in sel else "     "} {get(p, "model", "s")}')
if not sel:
    sys.exit('!! no DTB matches the running board - nothing changed')
for i in sel:
    r = subprocess.run(['fdtoverlay', '-i', paths[i], '-o', paths[i] + '.new', ovl], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f'!! fdtoverlay failed on [{i}]: {r.stderr.strip()}')
    new = paths[i] + '.new'
    # camss waits for EVERY sensor linked in its ports; the IMX577 link (port@3) never binds
    # (its CCI is disabled and the sensor doesn't answer), so camss never registers /dev/media.
    for args in (['-r', new, '/soc@0/camss@acaf000/ports/port@3/endpoint'],
                 ['-r', new, '/soc@0/cci@ac4b000/i2c-bus@1/camera@1a'],
                 ['-d', new, '/__symbols__', 'csiphy3_ep'],
                 ['-d', new, '/__symbols__', 'imx577_ep']):
        subprocess.run(['fdtput'] + args, capture_output=True)
    blobs[i] = open(new, 'rb').read()
    ok = subprocess.run(['fdtget', new, '/soc@0/cci@ac4a000/i2c-bus@0/camera@60', 'compatible'],
                        capture_output=True, text=True).stdout.strip()
    eps = subprocess.run(['fdtget', '-l', new, '/soc@0/camss@acaf000/ports/port@3'],
                         capture_output=True, text=True).stdout.split()
    print(f'  [{i}] patched, camera@60 compatible = {ok}, port@3 children = {eps or "none (good)"}')
open(dst, 'wb').write(b''.join(blobs) + tail)
PY

sudo cp new_multi.dtb "$DTB"
sync
sudo umount dtbmnt
trap - EXIT

echo
echo "Built dtb_a_new.img. BEFORE continuing, copy dtb_a_backup.img to your laptop (scp) - it's the recovery file."
read -r -p "Write dtb_a_new.img to the dtb_a partition now? Type YES: " a
[ "$a" = "YES" ] || { echo "Nothing written."; exit 0; }
sudo dd if=dtb_a_new.img of=$PART bs=1M conv=fsync status=none
cmp <(sudo head -c "$(stat -c %s dtb_a_new.img)" $PART) dtb_a_new.img && echo "dtb_a written and verified."

# Keep Qualcomm's CamX driver off the camera hardware so upstream camss owns it.
# (The grub.d route alone didn't take effect last time, so also block the module in modprobe + initramfs.)
sudo rm -f /etc/default/grub.d/90-camss.cfg
sudo mkdir -p /etc/default/grub.d
echo 'GRUB_CMDLINE_LINUX_DEFAULT="$GRUB_CMDLINE_LINUX_DEFAULT module_blacklist=camera_qcm6490"' \
  | sudo tee /etc/default/grub.d/zz-camss.cfg >/dev/null
sudo update-grub >/dev/null 2>&1
printf 'blacklist camera_qcm6490\ninstall camera_qcm6490 /bin/false\n' | sudo tee /etc/modprobe.d/zz-no-camx.conf >/dev/null
sudo update-initramfs -u >/dev/null 2>&1
sudo systemctl disable cam-server >/dev/null 2>&1 || true

echo "Done. Now: sudo reboot   then: ../../scripts/setup_rb3.sh  (drivers, loaded at every boot)"
echo "To undo everything: bash dtb_restore.sh"
