# Visual navigation pipeline

The stages follow the architecture sketch:

```text
FrameProcessor -> LargeFeatureDetector -> SmallFeatureTracker -> DepthEstimator -> VisulaAcEst
    frame.py       largeFeatures.py         smallFeatures.py       depth.py       visualAccEst.py
```

`VisulaAcEst` remains the public entry point and coordinates the stages. Its own
numerical work is motion compensation, velocity estimation, and velocity filtering.
The original import and simulator/camera-runner interfaces are retained:

```python
from swarm.compVision.visualAccEst import VisulaAcEst

estimator = VisulaAcEst(res=(1920, 1080), fov=60)
estimator.image_filters = ("clahe", "gaussian")
estimator.filter_kernel_size = 3
estimator.set_dt(1 / 30)
processed = estimator.processing([left_bgr, right_bgr], frame_index)
velocity = estimator.camera_velocity  # [vx, vy, supplied_sensor_vz, yaw_rate]
```

Use your actual calibrated field of view or set `foc_l` and `camera_saperation`
from calibration. The example values above are placeholders. Continue setting
the existing sensor attributes and calling `push_sensors()` as the controller does.

## Responsibilities and data

| Stage | Owns | Output |
| --- | --- | --- |
| `frame.py` | Ordered CLAHE/Gaussian/median filters, grayscale conversion, Gaussian pyramids | `Frame`: clean stereo BGR images, gray images, full/half/quarter pyramids, frame index |
| `largeFeatures.py` | Long-line detection at a coarse pyramid level; grouping nearby lines into padded regions | `LargeFeatures`: lines and region boxes in original-image pixels |
| `smallFeatures.py` | Existing GFTT + Canny proximity + quadrant selection, temporal LK, forward/backward checks, affine/neighbor filters, Delaunay mesh | `FeatureTracks`: matched previous/current left-image points, triangles, links |
| `depth.py` | Same-frame stereo LK, reverse checks, signed disparity/epipolar checks, depth reconstruction | `DepthResult`: aligned left/right matches, depths, validity, frame index |
| `visualAccEst.py` | Stage orchestration, sensor windows, vertical-flow compensation, predicted-motion filtering, planar velocity/yaw fit and Kalman filtering | Existing public frame, velocity, and depth attributes |

Only two clean stereo frames are retained. Drawing occurs on color copies, so
mesh and region overlays never enter subsequent feature detection. Yellow boxes
are current-frame structural proposals; blue points/lines are tracked features.
Right-image overlays use actual current stereo matches and omit invalid matches.

Feature coordinates always stay in original-image pixels. Large detection maps
`pyrDown` coordinates by `2**level`, including odd image sizes. Full-resolution
images remain the inputs to temporal and stereo LK. Each frame's large regions
are cached for replenishing its points when the next frame arrives.

## What is reused, and what changes

The fine detector retains the existing GFTT settings (`qualityLevel=0.005`,
`minDistance=100`, `blockSize=11`), seven-pixel existing-point exclusion, Canny
proximity check, and quadrant balancing. Large regions get first access to roughly
35% of available new-point slots (rounded up); global detection supplies the rest.
This reserves priority slots, not a cap on all points eventually inside regions.
Set `estimator.small_features.region_fraction = 0` for the original global
selection path. This split does not enable the separate experimental statistical
budget sampler supplied earlier. The detector/tracker settings are now explicit
attributes on `small_features` so that experiment can be added independently.

The existing LK, global affine, and mesh-neighbor filters are retained. Degenerate
meshes and failed LK calls return empty tracks safely. New points cannot overflow
the total point budget when existing tracks are concentrated in one quadrant.

Filter settings now actually control preprocessing; the default remains CLAHE.
The old implementation exposed these settings but applied CLAHE unconditionally.
In-session resolution changes are rejected because a different pixel scale needs
matching camera calibration. Create a fresh estimator with that calibration.

## Depth and velocity semantics

Stereo depth is now an active separate stage for the configured simulator camera
model. It matches both the previous and current stereo pairs independently.
Current right-image coordinates are **not** synthesized as old right + left flow.
Disparity uses the configured `right - left` direction, so horizontal and vertical
stereo baselines are supported.

`DepthEstimator` requires rectified or already aligned common-intrinsic pinhole
images with aligned principal points. It does not calibrate cameras. Raw USB
images in `realCameraTest` have different intrinsics and are only resized/rotated;
that runner explicitly sets `use_stereo_depth = False` until stereo calibration
and rectification are supplied. With stereo disabled, the stage uses a positive
supplied height for scaling if available and reports no measured stereo depth.

Inspect `current_depth.valid` to distinguish measured depth from fallback values.
Invalid matches use the median measured depth, or positive supplied height if no
depth matches survive, preserving the previous fallback policy. If neither is
available, depth is NaN and the velocity solver falls back to supplied sensor
velocity. Filled depths are not included in `depth_points` or `depth_valid_frac`.
Height as optical-axis depth remains an approximation when the drone is tilted.

The velocity solver still estimates horizontal velocity and yaw rate using the
existing model. It does not yet independently estimate vertical velocity.
`camera_velocity[2]` now contains supplied `drone_vel[2]`, matching the controller's
existing use of sensor vertical velocity. The old code incorrectly put a depth
value in that velocity slot. The largest-decile depth statistic remains available
as `estimated_height`, separately from `depth_median`.

The existing global affine/neighbor filters can reject points on surfaces with
different depths. This refactor does not replace the motion model with a full
visual-inertial optimizer. Validate navigation accuracy and timing with recorded
or simulated sequences before drawing conclusions from feature counts.

## Checks

Run from the repository root with its Python environment:

```sh
.venv-py313/bin/python -m unittest discover -s swarm/compVision/tests -v
.venv-py313/bin/python -m unittest discover -s realCameraTest/tests -v
```

These are headless module/integration checks. No Genesis session or physical
camera is required. They cover preprocessing, structural proposals, fine tracking,
analytic and synthetic stereo depth, current-frame correspondence, velocity units,
failure paths, and the existing camera runner interface. They are not a flight
accuracy benchmark.
