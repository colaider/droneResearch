# City from above: binocular depth sandbox

The city uses bundled **Kenney CC0 buildings and roads**. Run from the project
root with the environment used for the original swarm demo:

```sh
.venv311/bin/python -m swarm.city_stereo
```

The drone starts just above the road at `(8, 0, 0.6)` metres, climbs vertically
to **40 m**, settles, then surveys along world +X. Both cameras face **downward**
from underneath the drone, viewing rooftops and streets. The tallest building
is about 33 m. The initial climb takes about 50 simulated seconds; the default
run lasts 80 simulated seconds. Terminal messages show climb/settle/survey.
Use `--speed 0` to climb and hover over the city. Press Q or Escape in the image
window to exit. CPU simulation and image matching can run slower than real time.

To see the city from above immediately, skipping the climb:

```sh
.venv311/bin/python -m swarm.city_stereo --start-airborne
```

Save one overhead stereo pair without displaying windows:

```sh
.venv311/bin/python -m swarm.city_stereo --start-airborne --headless --steps 1
```

`--headless` still uses OpenGL. Options include `--altitude 40` (35 to 60 m),
`--baseline 0.5`, `--width 640 --height 480`, `--fov 65`, `--disparities 128`,
`--camera-fps 10`, `--steps 8000`, and `--output <directory>`.
Steps are 0.01 simulated seconds each. FOV is **vertical**, in degrees.
Disparity search must be a multiple of 16, below image width. The wider 0.5 m
baseline helps resolve depth at survey height; it is a simulated camera boom,
not a validated mechanical payload for this drone model.

The route is scripted; depth does not yet control obstacle avoidance. For a
stable long climb, the flight controller uses simulator position as a GPS-like
reference and also retains velocity correction. This is not vision-only
navigation. No simulator position, altitude or depth is used to match images
or triangulate points.

## Code to edit

- `swarm/scene_builders/city_scene.py`: city layout, model scale and conservative
  building box colliders. Road and street-detail meshes are visual only.
- `swarm/sensors/stereo_rig.py`: synchronized downward camera mounts. Left
  is at body +Y, right at body -Y, both looking along body -Z. Image right
  is body -Y; image up is body +X. Cameras sit 0.12 m below the body with a
  0.5 m baseline, identical intrinsics and parallel axes. They tilt together
  with the aircraft (no gimbal). The optional forward mount remains available
  through `StereoRig(..., direction='forward')`.
- `swarm/compVision/stereo_depth.py`: `StereoCalibration` defines intrinsics and
  projection matrices; `StereoDepth.compute` runs dense SGBM; `sparse_matches`
  finds mutual ORB feature correspondences; `triangulate_matches` reconstructs
  their XYZ coordinates using `cv2.triangulatePoints`.
- `swarm/city_stereo.py`: simulation loop, position reference, depth preview and
  capture export. This is the entry point for city experiments.

The existing `visualAccEst.py` estimates motion using downward imagery and a
flat-ground assumption. It is not the city depth algorithm. The city demo
adds its own downward stereo pair and uses `DronePositionCTRL` for flight.
Function docstrings explain argument units, image formats and coordinate frames.

## Geometry and data flow

One physics state -> synchronized left/right RGB images -> BGR images ->
grayscale matching -> disparity and sparse correspondences -> metric XYZ.
Simulator depth and drone altitude are **not inputs** to this reconstruction.

For a rectified pair, disparity `d = u_left - u_right` is measured in pixels:

```
f = height / (2 tan(vertical_fov / 2))
Z = f B / d
X = (u - cx) Z / f
Y = (v - cy) Z / f
```

`B` is baseline in metres, `f` focal length in pixels, `(cx,cy)` image centre,
and `(u,v)` a left-image pixel. XYZ uses the **left optical frame**: X right,
Y down **in the image**, Z along the camera viewing axis. For a level drone,
optical Z points toward the ground. A roof produces a shorter measured depth
than the street; there is no flat-ground assumption. Z is optical depth, not
Euclidean distance or world altitude. When tilted, use the exported transform
to recover world heights instead of simply subtracting depth from altitude.
The general sparse triangulation uses `P_left = K[I|0]` and
`P_right = K[I|(-B,0,0)]`. Real binocular cameras require measured intrinsics,
distortion coefficients, relative pose, synchronization and image rectification.

Dense results are checked for left/right consistency, local texture and depth
range. Sparse matches use mutual matching, a descriptor ratio test, vertical
alignment, positive disparity and reprojection checks. These checks cannot
eliminate all mistakes on repeated windows or uniform walls. Black pixels in
the depth preview mean **unknown**, never clear space. The minimum disparity
threshold and finite search range also limit measurable distance. A larger
baseline improves distant depth sensitivity but increases occlusion.

## Saved data

Each run creates `outputs/city_stereo/<timestamp>/` and updates its latest
capture (it does not store a video or every frame):

- `left.png`, `right.png`: synchronized images.
- `preview.png`: left/right, depth colors and triangulated features.
- `depth.npz`: `depth_m` (HxW, metres, NaN if unknown), `disparity_px`, `valid`,
  `left_uv`/`right_uv` (Nx2 pixels) and `points_m` (Nx3 sparse optical XYZ).
- `calibration.json`: intrinsics, projection matrices, simulated capture time,
  baseline and optical-to-world transform at capture time.
- `points.ply`: colored dense reconstruction sampled every four pixels.

For example:

```python
import numpy as np
capture = np.load('outputs/city_stereo/<timestamp>/depth.npz')
depth = capture['depth_m']
sparse_xyz = capture['points_m']
```

The world transform is simulator metadata for locating exported points; the
matcher never uses it. To transform an optical point, append 1 and multiply by
`world_from_left_camera`.

## Checks

`python -m pytest tests/test_stereo.py tests/test_aerial.py` checks metric triangulation, disparity,
texture matching, blank-image rejection, downward mount geometry and flight phases.
`python -m tests.smoke_stereo` renders a textured wall at a known distance and
checks the actual mounted camera pair and metric reconstruction together.

`python -m tests.smoke_stereo_down` renders textured ground and a roof 4 m
higher, checking distinct metric depths and reconstructed world heights.
