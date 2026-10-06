# Real USB camera test

This folder drives the real `swarm/compVision/visualAccEst.py` estimator **directly** with two
USB cameras. There is no separate/duplicated estimator here anymore: `testRealCameras.py`
instantiates `swarm.compVision.visualAccEst.VisulaAcEst` and feeds it frames plus every input it
reads each step. Because there is no flight controller, all **drone/IMU inputs are supplied as
zero** (`drone_pos`, `drone_vel`, `drone_ang_vel`, `imu_att`, `previous_cmd_vel`, `drone_ang`);
only the camera calibration (focal length, baseline) comes from `camera_params.txt`.

Changes to the swarm feature detector, optical flow, filters, drawing, and velocity fitting apply
here automatically. Keep this folder inside the full repository. Python needs OpenCV, NumPy, and
SciPy, but the runner does not import Genesis or start a simulation.

Run from the repository root:

```sh
python realCameraTest/testRealCameras.py 0 1
# Equivalent package entry point:
python -m realCameraTest.testRealCameras 0 1
```

The single resizable window shows left/right filtered color images with tracking overlays above
left/right grayscale tracking inputs (reusing the estimator's cached `gray_frames`). Velocity and
depth text is drawn on the display only.

## Try filters

Without `--filters`, the runner inherits `image_filters` and `filter_kernel_size` from the swarm
estimator. Filter implementations live in its `preprocess_frame()` method.

```sh
# Gaussian blur followed by CLAHE:
python realCameraTest/testRealCameras.py 0 1 --filters gaussian clahe --kernel-size 5
# Median filter:
python realCameraTest/testRealCameras.py 0 1 --filters median
# No filters:
python realCameraTest/testRealCameras.py 0 1 --filters
```

## Camera setup and controls

- `q` or Escape exits. `s` swaps cameras, their focal calibration, and their mounting rotations,
  then restarts tracking with a fresh estimator.
- Calibration constants at the top of `testRealCameras.py`: 1920×1080, focal lengths
  1719.33 / 1176.10 px, baseline 0.125 m. They are applied to the estimator instance via
  `est.foc_l` / `est.camera_saperation` — the estimator itself is unmodified.
- Frames are resized to the calibration resolution. The left 90° CCW / right 90° CW mounting
  corrections are enabled; use `--no-rotate` if your images are already aligned.
- No IMU or commanded velocity: all drone inputs are zero, so rotation compensation is zero. The
  stereo disparity direction and the fallback scene depth are governed by the shared estimator and
  `swarm/config/cameraConfig.py` (its `disparity_direction()` / barometric height), not by this
  folder. Metric estimates depend on calibration and camera alignment; resizing/mounting rotation
  does not stereo-rectify the images.

## Headless checks

```sh
python -m unittest discover -s realCameraTest/tests -v
```

Tests drive the real estimator with mocked cameras and zeroed drone inputs; they do not access
physical hardware or start Genesis.
