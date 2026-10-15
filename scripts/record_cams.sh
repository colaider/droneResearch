#!/usr/bin/env bash
# Record two USB cameras at the same time on a Raspberry Pi.
#
# Each camera's own MJPEG stream is copied as-is (no re-encode, near-zero CPU) into
# Matroska (.mkv). MKV stays playable when recording is cut off (Ctrl+C, kill -9,
# pulled power): only the last ~1 s is lost, a few seconds on power loss.
# MP4/AVI would be unreadable because their index is written only at the end.
#
# Usage:  ./record_cams.sh [DEVICE1 DEVICE2]        (default /dev/video0 /dev/video2)
#   Settings:   RES=1920x1080 FPS=30 OUT_DIR=/media/usb ./record_cams.sh
#   Cam modes:  v4l2-ctl -d /dev/video0 --list-formats-ext
#   "No space left on device" in a log = USB bandwidth; lower RES/FPS.
# Stop with Ctrl+C or SIGTERM; both files are then finalized normally.

set -u

RES="${RES:-1280x720}"
FPS="${FPS:-30}"
OUT_DIR="${OUT_DIR:-$HOME/recordings}"
devices=("${1:-/dev/video0}" "${2:-/dev/video2}")

command -v ffmpeg >/dev/null || { echo "ffmpeg missing: sudo apt install ffmpeg" >&2; exit 1; }
mkdir -p "$OUT_DIR" || exit 1

stamp="$(date +%Y%m%d_%H%M%S)"
pids=() files=() logs=()

for i in 0 1; do
    files[i]="$OUT_DIR/${stamp}_cam$i.mkv"
    logs[i]="$OUT_DIR/${stamp}_cam$i.log"
    # -nostdin: a backgrounded ffmpeg reading the terminal gets stopped by SIGTTIN.
    # cluster_time_limit: MKV is written in self-contained clusters; this caps what a hard kill loses.
    ffmpeg -hide_banner -nostdin -loglevel warning \
        -f v4l2 -input_format mjpeg -video_size "$RES" -framerate "$FPS" -i "${devices[i]}" \
        -c:v copy -cluster_time_limit 1000 \
        "${files[i]}" 2>"${logs[i]}" &
    pids[i]=$!
    echo "cam$i: ${devices[i]} -> ${files[i]}"
done

stopping=0
stop() {
    stopping=1
    trap '' INT TERM
    printf '\nStopping, finalizing files...\n'
    # Background jobs of a script start with SIGINT ignored, so Ctrl+C may not reach
    # ffmpeg; SIGTERM makes it finish the file just the same.
    kill -TERM "${pids[@]}" 2>/dev/null
}
trap stop INT TERM

echo "Recording ${RES}@${FPS}. Ctrl+C to stop."
reported=(0 0)
while :; do
    alive=0 status=""
    for i in 0 1; do
        if kill -0 "${pids[i]}" 2>/dev/null; then
            alive=1
            status+="  cam$i $(du -h "${files[i]}" 2>/dev/null | cut -f1)"
        else
            status+="  cam$i STOPPED"
            if [ "$stopping" -eq 0 ] && [ "${reported[i]}" -eq 0 ]; then
                reported[i]=1
                printf '\ncam%d (%s) stopped unexpectedly:\n' "$i" "${devices[i]}" >&2
                tail -n 5 "${logs[i]}" >&2
            fi
        fi
    done
    [ "$alive" -eq 1 ] || break
    printf '\r%02d:%02d%s   ' $((SECONDS / 60)) $((SECONDS % 60)) "$status"
    sleep 1
done

for i in 0 1; do
    [ -s "${logs[i]}" ] || rm -f "${logs[i]}"
done
echo "Done. Files in $OUT_DIR"
exit $(( reported[0] || reported[1] ))
