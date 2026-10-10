# scripts – helper scripts

| Script | Where | What |
|---|---|---|
| `setup_rb3.sh` | drone | One-time setup: camera drivers (loaded at every boot), ROS 2 workspace, `~/.bashrc`. Rerun after a kernel update |
| `run_stereo.sh` | drone | Start the stereo camera ROS 2 node (`fps:=60`, `resolution:=1280x800`, ...) |
| `view_stereo.sh` | Linux / WSL | Live view + camera settings over ROS 2 (`--ns /vo` for `realCameraTest/testRB3Cameras.py`) |
| `view_stereo.ps1` | Windows | The same viewer with a native Windows ROS 2 install |
| `record_cams.sh` | Raspberry Pi | Record two USB cameras to MKV (flight-test footage) |

Details: `rb3_stereo/QUICKSTART.md`.
