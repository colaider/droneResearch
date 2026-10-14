# rb3_stereo – quick start

Synchronized OV9282 stereo camera (CAM0A + CAM0B) on the RB3 Gen 2 as a ROS 2 node.
Board login: `ubuntu@192.168.0.172`, ROS 2 Jazzy, `ROS_DOMAIN_ID=42`.

## 1. Set up the drone (once)

Already done on the current board. The repo is checked out at `~/droneResearch`; on a fresh board:

```bash
git clone https://github.com/colaider/droneResearch.git ~/droneResearch
~/droneResearch/scripts/setup_rb3.sh
```

`setup_rb3.sh` builds and installs the camera drivers and **loads them at every boot**. It links
this package into `~/ros2_ws` (symlink install: edits in the repo take effect without a rebuild)
and adds ROS 2 Jazzy, the workspace and `ROS_DOMAIN_ID=42` to `~/.bashrc`. It is safe to run again.

- **Kernel update:** the drivers are built for one kernel. After `apt upgrade` installs a new
  kernel, rerun `setup_rb3.sh`.
- **Fresh firmware flash:** the camss device tree must be on the board first. Run
  `drivers/devicetree/dtb_flash.sh` once. See drivers/README.md.
- **Hardware:** vision mezzanine switch **DIP2-1 must be OFF**.

## 2. Start the camera (on the drone)

```bash
~/droneResearch/scripts/run_stereo.sh                              # 640x400 @ 30 fps
~/droneResearch/scripts/run_stereo.sh fps:=60
~/droneResearch/scripts/run_stereo.sh fps:=120
~/droneResearch/scripts/run_stereo.sh resolution:=1280x800 fps:=30   # or fps:=60
~/droneResearch/scripts/run_stereo.sh auto_exposure:=false exposure_us:=3000 gain:=2.0
```

| Mode | fps |
|---|---|
| 640x400 | 30, 60, 120 |
| 1280x800 | 30, 60 |

Stop it with **Ctrl-C**. Run it in `tmux` to keep it alive after logout. Don't use `kill`: ROS 2
launch leaves its node running on SIGTERM.

To run the swarm estimator (VO) on the cameras instead, with a live view on `/vo`:

```bash
cd ~/droneResearch && python3 realCameraTest/testRB3Cameras.py        # --help for the options
```

Only one program can use the cameras at a time, so stop the camera node first.

## 3. Look at it from another computer

On any machine on the same network with ROS 2 Jazzy and python3-opencv, from the repo root:

```bash
./scripts/view_stereo.sh                               # live preview, ROS_DOMAIN_ID 42
./scripts/view_stereo.sh --fps 60 --exposure-us 3000 --gain 2
./scripts/view_stereo.sh --auto-exposure on --ae-target 120
./scripts/view_stereo.sh --ns /vo                      # the VO test (testRB3Cameras.py)
./scripts/view_stereo.sh --help
```

From WSL2 (mirrored networking) on the laptop, for example:
`bash "/mnt/c/Users/domku/Desktop/Drone Hackathon/Project repo/droneResearch/scripts/view_stereo.sh"`.
On Windows with ROS 2 Jazzy installed, use `.\scripts\view_stereo.ps1`. It takes the same options.

Keys in the window:

| Key | Action |
|---|---|
| `q` | quit |
| `s` | save picture |
| `a` | auto exposure on/off |
| `+` / `-` | exposure |
| `g` / `h` | gain |
| `1` `2` `3` | 30 / 60 / 120 fps |
| `r` | 640x400 ↔ 1280x800 |

The viewer shows the small JPEG preview, which is fine over Wi-Fi. `--raw` shows the full images,
but only use that on a wired link.

## 4. Change settings while it runs

```bash
ros2 param set /stereo/stereo_camera fps 60
ros2 param set /stereo/stereo_camera auto_exposure false
ros2 param set /stereo/stereo_camera exposure_us 2500
ros2 param set /stereo/stereo_camera gain 2.0
ros2 param set /stereo/stereo_camera resolution 1280x800      # restarts the cameras (~3 s)
ros2 topic echo /stereo/status                                # fps, sync offset, exposure, ...
```

## 5. Calibrate the cameras

Needs a printed checkerboard on a flat, rigid board: 9x7 squares of 29 mm (other boards:
`--squares` and `--square-mm`). The calibration window runs on the drone and shows on your screen
through SSH X forwarding. From WSL (WSLg shows the window) or any Linux machine:

```bash
ssh -Y -C ubuntu@192.168.0.172
```
```bash
python3 ~/droneResearch/scripts/calibrate_rb3.py
```

On Windows without WSL, use MobaXterm (it has an X server built in). The script asks to stop
`drone-stereo` / `drone-vo` while it uses the cameras and starts them again at the end.

1. Hold the board so **both** cameras see all of it (the corners light up in both images) and press
   **SPACE**, or press **A** once to capture automatically whenever the board is held still in a new
   place. Collect 20-30 pairs: near and far, tilted left/right/up/down, and in every part of the
   image (the green outlines show what is covered so far). **U** removes the last pair.
2. **C** calibrates and prints the result. Good: reprojection errors below 0.5 px, and a baseline
   that matches the spacing of the two lenses.
3. The view switches to the rectified images with guide lines: the board's corners must sit on the
   same line in both images ("board row error" well below 1 px). **R** goes back to recording more.
4. **S** saves `rb3_stereo/config/stereo_calib.npz`. When the script ends, the camera services start
   again and rectify every frame from then on (status: `"rectified": true`).

Options: `--check` shows the live rectified view with the saved calibration; `--images DIR`
recalibrates pairs saved earlier (they are kept in `~/rb3_calibration/`), without camera or window.
Commit the calibration file so the rest of the team gets it.

## 6. Use it in code: same interface as the simulation

```python
from rb3_stereo.client import StereoSubscriber
cams = StereoSubscriber("/stereo")
left, right = cams.get_two_frames()       # BGR, like DroneStruct.get_two_frames()
proc = VisulaAcEst(cams.res, cams.fov)    # res=(width, height), fov=vertical degrees
proc.camera_saperation = cams.camera_saperation
proc.disparity_direction = cams.disparity_direction   # from the calibration (right-minus-left shift)
```

The simulation can publish the same topics, so the vision code runs unchanged on both:

```python
from rb3_stereo.sim_publisher import StereoPublisher
pub = StereoPublisher(res=(500, 600), fov_deg=85, baseline_m=0.05)
pub.publish(*drone.get_two_frames())
```

Topics: `/stereo/{left,right}/image_raw`, `/stereo/{left,right}/camera_info`,
`/stereo/preview/compressed` and `/stereo/status`.

Until the cameras are calibrated (section 5), the images are raw and `camera_info` uses the
simulation's values (85°, 5 cm). After that they are rectified and `res`, `fov`,
`camera_saperation` and `disparity_direction` come from the calibration.
