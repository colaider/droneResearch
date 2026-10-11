#!/bin/bash
# Bake one or more device-tree overlays into the firmware's own device tree (dtb_a partition).
#
# Generalised from dtb_flash.sh (which is specific to the OV9282 overlay). This one applies
# whatever .dtbo files you pass, on top of what is ALREADY in the partition, so earlier patches
# (e.g. the camera overlay) are preserved.
#
#   ./dtb_apply.sh icm42688-imu.dtbo              # build is up to you, or pass a .dtso and it compiles
#   ./dtb_apply.sh --dry-run icm42688-imu.dtso    # show what would change, write nothing
#
# Safeguards: keeps a pristine backup the first time it runs, works on a copy, prints the DTBs it
# would patch, and asks before writing to the partition. Restore with:
#   sudo dd if=dtb_a_backup.img of=/dev/disk/by-partlabel/dtb_a bs=1M
set -e
cd "$(dirname "$0")"
PART=/dev/disk/by-partlabel/dtb_a

DRY=0
OVERLAYS=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    *.dtso)
      out="${arg%.dtso}.dtbo"
      dtc -@ -I dts -O dtb -o "$out" "$arg"
      OVERLAYS+=("$out")
      ;;
    *.dtbo) OVERLAYS+=("$arg") ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
[ ${#OVERLAYS[@]} -eq 0 ] && { echo "usage: $0 [--dry-run] <overlay.dtso|overlay.dtbo> ..." >&2; exit 2; }

command -v fdtoverlay >/dev/null || sudo apt-get install -y device-tree-compiler >/dev/null

if [ ! -f dtb_a_backup.img ]; then
  sudo dd if=$PART of=dtb_a_backup.img bs=1M status=none
  sudo chown "$USER" dtb_a_backup.img
  echo "pristine backup written: dtb_a_backup.img ($(stat -c %s dtb_a_backup.img) bytes)"
fi

# Work from the CURRENT partition content so previous overlays stay applied.
sudo dd if=$PART of=dtb_a_current.img bs=1M status=none
sudo chown "$USER" dtb_a_current.img
cp dtb_a_current.img dtb_a_new.img

mkdir -p dtbmnt
sudo mount -o loop dtb_a_new.img dtbmnt || { echo "!! dtb_a is not a mountable image:"; file dtb_a_current.img; exit 1; }
trap 'sudo umount dtbmnt 2>/dev/null || true' EXIT
DTB=$(ls -S dtbmnt/*.dtb 2>/dev/null | head -n1)
[ -z "$DTB" ] && { echo "!! no .dtb file in the partition"; exit 1; }
echo "patching $DTB with: ${OVERLAYS[*]}"

sudo cat /sys/firmware/fdt > live.dtb
cp "$DTB" orig_multi.dtb

python3 - orig_multi.dtb new_multi.dtb "${OVERLAYS[@]}" <<'PY'
import struct, subprocess, sys
src, dst, *overlays = sys.argv[1:]

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
    current = paths[i]
    for ovl in overlays:
        out = current + '.new'
        r = subprocess.run(['fdtoverlay', '-i', current, '-o', out, ovl], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f'!! fdtoverlay failed on [{i}] with {ovl}: {r.stderr.strip()}')
        current = out
    blobs[i] = open(current, 'rb').read()
    print(f'  [{i}] patched, {len(blobs[i])} B')

open(dst, 'wb').write(b''.join(blobs) + tail)
PY

if [ "$DRY" = "1" ]; then
  echo "dry run: nothing written to $PART (result is in new_multi.dtb)"
  exit 0
fi

read -rp "write the patched device tree to $PART ? [y/N] " answer
[ "$answer" = "y" ] || { echo "aborted, partition untouched"; exit 0; }
sudo cp new_multi.dtb "$DTB"
sudo umount dtbmnt
trap - EXIT
sudo dd if=dtb_a_new.img of=$PART bs=1M status=none conv=fsync
echo "written. reboot to pick it up. If the board does not come back:"
echo "  sudo dd if=dtb_a_backup.img of=$PART bs=1M conv=fsync   (from a recovery shell / EDL)"
