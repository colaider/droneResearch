# Real USB camera test

This folder uses the current `swarm/compVision/visualAccEst.py` through a small USB adapter. Changes to the swarm feature detector, optical flow, filters, drawing, and velocity fitting also apply here. Keep this folder inside the full repository; it no longer contains a separate estimator copy. Python needs OpenCV, NumPy, and SciPy, but the runner does not import Genesis or start a simulation.

Run from the repository root:

```sh
python realCameraTest/testRealCameras.py 0 1
# Equivalent package entry point:
python -m realCameraTest.testRealCameras 0 1
```

The single resizable window shows left/right filtered color images with tracking overlays above left/right grayscale tracking inputs. Grayscale previews reuse the estimator's cached inputs; filtering is not repeated for display. Velocity/depth text is drawn on the display only.

## Try filters

Without `--filters`, the runner inherits `image_filters` and `filter_kernel_size` from the swarm estimator. Custom filter implementations belong in its `preprocess_frame()` method.

```sh
# Gaussian blur followed by CLAHE:
python realCameraTest/testRealCameras.py 0 1 --filters gaussian clahe --kernel-size 5
# Median filter:
python realCameraTest/testRealCameras.py 0 1 --filters median
# No filters:
python realCameraTest/testRealCameras.py 0 1 --filters
```

Filter order matters. Restart the runner after editing code or choosing different filter options. The old independent sharpening/rolling-shutter preprocessing has been replaced by the shared swarm pipeline.

## Camera setup and controls

- `q` or Escape exits. `s` swaps cameras, their focal calibration, and their mounting rotations, then starts fresh tracking.
- Calibration constants at the top of `testRealCameras.py` retain the previous USB values: 1920×1080, focal lengths 1719.33 / 1176.10 pixels, baseline 0.125 m. These are constants, not dynamically loaded from a calibration file.
- Frames are resized to the calibration resolution. The existing left 90° counterclockwise / right 90° clockwise mounting corrections remain enabled. Use `--no-rotate` if the incoming images are already aligned.
- The USB adapter accepts `disparity_dir`, `epipolar_tol`, `min_disparity`, and `default_depth`. The default disparity axis is horizontal in the images *after* rotation. Adjust these in the runner's `make_estimator()` for your camera arrangement.
- There is no IMU or commanded velocity: rotation compensation is zero and command-based rejection is skipped. When stereo depth is unavailable, the adapter uses `default_depth` (1 m by default). Metric estimates depend on calibration and camera alignment; resizing/mounting rotation does not stereo-rectify the images.

## Headless checks

```sh
python -m unittest discover -s realCameraTest/tests -v
```

Tests cover shared methods, filter/display consistency, synthetic temporal/stereo motion, depth fallback behavior, camera swapping, and resource cleanup. They do not access physical cameras.
