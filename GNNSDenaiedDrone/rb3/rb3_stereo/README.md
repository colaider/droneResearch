# rb3_stereo

ROS 2 (Jazzy) driver for the two OV9282 global-shutter cameras on the Qualcomm RB3 Gen 2 vision
mezzanine (connectors CAM0A + CAM0B). It produces frame-synchronized stereo pairs at 30, 60 or 120 fps.
It also provides a subscriber and a publisher with the same interface as the swarm simulation, so
the vision code runs unchanged on simulated and real cameras.

- **QUICKSTART.md**: how to install, run, view and use it.
- **docs/HOW_IT_WORKS.md**: what was done to make the hardware work, hardware issues, and known
  risks.

## What runs where

```
 drone (RB3 Gen 2)                                                   any machine, same ROS_DOMAIN_ID
 ┌────────────────────────────────────────────────────────────┐     ┌──────────────────────────────┐
 │ kernel: ov9282 (1-lane patch), qcom-camss (combo patch),   │     │ viewer.py / view_stereo.sh   │
 │         cam0b_bridge  ── loaded by rb3-stereo-setup        │     │  preview + settings          │
 │ stereo_camera node (rb3_stereo/node.py)                    │ ──► └──────────────────────────────┘
 │   StereoCamera (stereo.py): capture, pairing, genlock, AE  │
 │   publishes /stereo/... ──► vision code (StereoSubscriber) │
 └────────────────────────────────────────────────────────────┘
```

## Topics

All topics are in the namespace `/stereo`.

| Topic | Type | Notes |
|---|---|---|
| `left/image_raw`, `right/image_raw` | sensor_msgs/Image | `mono8`; both images of a pair have the identical `header.stamp`; sensor-data QoS (best effort). **Rectified** once the cameras are calibrated (see Calibration) |
| `left/camera_info`, `right/camera_info` | sensor_msgs/CameraInfo | rectified: the shared pinhole model, no distortion; else a calibration YAML or the nominal model. Right `P[3] = -fx·baseline` |
| `preview/compressed` | sensor_msgs/CompressedImage | side-by-side JPEG, `preview_fps` (10 Hz), for remote viewing |
| `status` | std_msgs/String | JSON every 5 s: fps, mode, sync offset, exposure, gain, mean brightness, `rectified` |

Stamps are on the ROS clock. By default the stamp is the **centre of the exposure**:
frame-end interrupt time − readout time − exposure/2. Use `stamp:=frame_end` for the raw
end-of-frame time.

## Parameters

All can be set at launch (`name:=value`) or at runtime (`ros2 param set /stereo/stereo_camera ...`).

| Parameter | Default | |
|---|---|---|
| `resolution` | `640x400` | `640x400` or `1280x800`; changing it restarts the cameras |
| `fps` | `30` | 30 / 60 / 120 (120 only at 640x400), live |
| `auto_exposure` | `true` | software AE (there is no ISP on this path) |
| `exposure_us`, `gain` | `2000`, `1.0` | manual values; only used when `auto_exposure` is false; gain 1.0–15.9 |
| `ae_target` | `100` | mean brightness 0–255 |
| `ae_max_exposure_us`, `ae_max_gain` | `4000`, `8.0` | limits for motion blur and noise |
| `left_camera` | `cam0a` | which connector is the left camera |
| `flip` | `false` | rotate both images 180° (sensor readout) |
| `genlock` | `true` | keep both cameras' frames aligned |
| `calibration` | `auto` | stereo calibration that rectifies every pair: `auto` = `config/stereo_calib.npz` if it exists, `""` = raw images, or a path; restarts the cameras |
| `left_calibration`, `right_calibration` | `""` | without a stereo calibration: camera_calibration YAML files for camera_info only |
| `nominal_vfov_deg`, `baseline_m` | `85`, `0.05` | nominal model used until calibrated (the simulation's values) |
| `stamp` | `exposure_center` | or `frame_end` |
| `frame_id_left`, `frame_id_right` | `stereo_{left,right}_optical_frame` | |
| `preview_fps`, `preview_scale`, `preview_quality` | `10`, `0.5`, `75` | `preview_fps 0` turns the preview off |
| `qos_reliable` | `false` | publish images/camera_info reliable instead of best effort (startup only) |

## Measured performance (RB3 Gen 2, Ubuntu 24.04, kernel 6.8.0-1084-qcom)

| Mode | Pairs/s | Node CPU (of 1 core) | Subscriber on the drone |
|---|---|---|---|
| 640x400 @ 120 | 119.7 | ~68 % | 119.8 pairs/s |
| 1280x800 @ 60 | 60.0 | ~85 % | 60.4 pairs/s (needs `config/fastdds.xml`, set by `run_stereo.sh` and `StereoSubscriber`) |

- **Sync:** the median frame offset between the cameras is within ±40 µs; p5–p95 is mostly within
  ±80 µs.
- **Latency:** about 8.5 ms from the end of a frame to Python, plus ROS transport.
- **CPU placement:** `run_stereo.sh` pins the node to the little A55 cores 0–3 (`RB3_STEREO_CPUS`
  overrides that), and the camera interrupts run on cores 2 and 3. The big A78 cores 4–7 stay free
  for the vision code.

## Calibration

Until the pair is calibrated, the images are raw and `camera_info` is the nominal model (85°
vertical FOV, 5 cm baseline), which only matches the simulation. `scripts/calibrate_rb3.py`
records a checkerboard (9x7 squares of 29 mm by default) with a live view over SSH X forwarding,
calibrates, shows the rectified result and saves `config/stereo_calib.npz` (+ `left.yaml` /
`right.yaml`). See QUICKSTART.md → Calibrate the cameras.

From then on `StereoCamera` undistorts and rectifies **every pair** (on the GPU via OpenCL, ~11 %
of a little core at 30 fps), so the node, `testRB3Cameras.py` and anything else that opens the
cameras get rectified images. Both images then share one pinhole model; `cam.rectifier` and the
`camera_info` topics carry its focal length, baseline and disparity direction. One calibration
covers 640x400 and 1280x800 (the same view, 640x400 is binned 2×2). The file has the format of
`scripts/calibrate_stereo.py`, so `swarm.compVision.denseStereo.StereoRectifier.from_npz` reads it
too - but the images are already rectified, so don't rectify them again.
`calibration:=""` (node) or `--raw` (testRB3Cameras.py) gives raw images.
The calibration belongs to how the cameras are mounted: recalibrate after moving them, and when
`left_camera` or `flip` change (it is ignored then, with a warning).

## Files

```
rb3_stereo/
  sensor.py          V4L2/media-ctl pipelines, v4l2-ctl capture with kernel timestamps, I2C (CAM0B)
  stereo.py          StereoCamera: modes, pairing, exposure, software genlock; AutoExposure
  node.py            ROS 2 node
  calibration.py     checkerboard calibration, Rectifier (undistort + rectify), calibration file
  camera_info.py     camera_info: rectified model / calibration YAML / nominal model
  client.py          StereoSubscriber  (drop-in for the simulation's get_two_frames/res/fov)
  sim_publisher.py   StereoPublisher   (simulation -> same topics)
  viewer.py          remote viewer + settings (single file, also runs without the package)
launch/stereo.launch.py, config/stereo.yaml (defaults), config/fastdds.xml (large images)
docs/HOW_IT_WORKS.md
```

Outside the package, in the repo root:

```
drivers/             kernel side, see drivers/README.md
  ov9282/ camss/ cam0b_bridge/   sources: ov9282 (1-lane patch), camss (combo patch), CAM0B bring-up
  install_board.sh   build + install the modules, I2C group, rb3-stereo-setup (+ --enable-boot)
  rb3-stereo-setup   load the drivers (root); rb3-stereo-setup.service runs it at boot
  devicetree/        dtb_flash.sh / dtb_restore.sh + the camss overlay (once per firmware flash)
scripts/
  setup_rb3.sh       one-time drone setup: drivers (loaded at boot), ROS workspace, ~/.bashrc
  run_stereo.sh      start the node on the drone
  view_stereo.sh / view_stereo.ps1   viewer on Linux / Windows
realCameraTest/testRB3Cameras.py     the swarm estimator on these cameras, live view on /vo
```

## Troubleshooting

- **`stereo kernel modules not loaded`:** run `sudo rb3-stereo-setup`. If it says "no combo mode",
  rerun `drivers/install_board.sh`; this is needed after kernel updates.
- **`no frames from both cameras`:** check that DIP2-1 is OFF and both flex cables are seated.
- **The viewer shows "waiting":** check that `ROS_DOMAIN_ID` matches (42) and that both machines
  are on the same subnet. Allow UDP through the firewall. On WSL2, enable mirrored networking.
- **A subscriber on the drone gets 1280x800 at a low rate:** set
  `export FASTRTPS_DEFAULT_PROFILES_FILE=$(ros2 pkg prefix rb3_stereo)/share/rb3_stereo/config/fastdds.xml`.
- **The node restarts the pipelines** ("no stereo pair for 1 s"): a capture path stalled, usually
  because of a loose connector or a corrupted frame. It recovers on its own in about 3 s.
