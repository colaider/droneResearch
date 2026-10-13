# Computer vision architecture

`VisulaEst.processing()` is the temporal tracking, velocity, and sparse stereo
entry point. In the simulator it is called by `DroneStruct.camera_step()`; camera
runners pass it a `[left, right]` pair.

```text
                    VisulaEst.processing([left, right], frame_idx)
                                      |
                update_frame -> FrameProcessor.process
                  normalize / filter / grayscale / pyramid
                                      |
                     aply_flow: temporal tracking
                  per-layer LK, previous LEFT -> current LEFT
                  replenish -> GFTT/Canny candidates (smallFeatures.py)
                     |                          |
              layer_tracks[layer]          all_points[layer]
              full-res left tracks         full-res left points
                     |                          |
                     +-> combine/filter -> estimate_velocities
                          flow + tracks + supplied height
                                                |
                         estimate_depth -> estimate_frame_depth
                         current LEFT + current RIGHT, per layer
                         patch search -> LK refinement/reverse check
                         -> signed disparity -> optical-axis depth
                                      |
                       depth_by_layer / depth_points
```

`all_points` is a list indexed by pyramid layer. Each entry contains that
layer's current **left-image** points in full-resolution pixel coordinates;
it does not contain points from both cameras. Temporal LK runs independently
for each layer, with its own cached seeds. Stereo matching happens afterward
between the current left and right images at the corresponding layer.

The simulator's active 3D map uses only the estimator's sparse stereo points:

```text
VisulaEst.depth_points -> PointMap (sparse tracked-feature map)
```

`largeFeatures.py` and `structureMap.py` are retained but isolated: the simulator
and camera runners do not invoke them. Their structural detections and dense
stereo points do not feed the active map.

The original estimator import and caller interface are retained:

```python
from swarm.compVision.visualEst import VisulaEst

estimator = VisulaEst(res=(1920, 1080), fov=60)
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
| `frame.py` | Normalize and filter each image; create grayscale images and pyramids | `Frame`: left/right BGR, grayscale images, per-camera pyramids, frame index |
| `smallFeatures.py` | Replenish corners with GFTT/Canny; build image-space links and filter neighboring tracks | Candidate points and per-layer track helpers |
| `visualEst.py` | Coordinate frame processing and stereo-depth coordination | Prepared frames, depth outputs, and public estimator interface |
| `visualVelEst.py` | Per-layer temporal LK, motion compensation/filtering, and velocity fit | `all_points[layer]`, `layer_tracks`, velocity and tracking display state |
| `visualDepthEst.py` | Match current left/right feature patches, validate matches, calculate and aggregate sparse depth | Per-layer `StereoDepth`, plus `depth_points` as full-resolution `(u, v, Z)` |
| `terrainMap.py` | Convert sparse stereo observations into world-frame voxels | `PointMap` used by the active simulator |
| `largeFeatures.py`, `denseStereo.py`, `structureMap.py` | Isolated structural detection and dense stereo mapping; not invoked by active runners | Future structure/height-map path |

The estimator retains recent prepared frames for temporal tracking. Drawing is
performed on color images, not the grayscale pyramids used for tracking. The
stereo depth stage draws only validated current-frame right-camera matches.

Tracked and public feature coordinates are expressed in full-resolution
left-image pixels. The layer-specific LK seeds are kept separately in native
layer pixels. `visualDepthEst` scales stereo search by `2**layer` and returns
right-image match coordinates and depth aligned to the left point rows.

## Tracking details

Each pyramid layer has its own cached point seeds and receives a share of
`max_feature_points` (default 1500 total). `replenish()` detects up to 500 GFTT
candidates per layer (`qualityLevel=0.005`, `blockSize=11`), keeps candidates
near Canny edges, and balances selection across image quadrants. The configured
`feature_min_distance` is in full-resolution pixels and is scaled for each layer.
New seeds are selected in the previous left image, then LK tracks them forward
to the current left image. Forward/backward, affine, and mesh-neighbor checks
filter tracks before they enter the velocity fit. Layer data stays separately
available for display and depth matching.

Image filters are applied by `FrameProcessor` in configured order; CLAHE is the
default. An empty filter tuple disables filtering. Resolution changes require a
new estimator with matching camera calibration.

## Current depth and velocity semantics

The current `processing()` order is `update_frame()`, `aply_flow()`,
`estimate_velocities()`, then `estimate_depth()`. `aply_flow()` tracks and filters
the temporal left-image flow, retaining aligned flow and point arrays across
layers. `processing()` passes the same filtered rows to `estimate_velocities()`.
For now, each track's Z comes from the supplied height; stereo depth is computed
afterward and does not feed velocity.

`camera_velocity` is `[vx, vy, supplied_sensor_vz, yaw_rate]`; `estimated_height`
stores the robust Z statistic used to scale image flow. If too few tracks
survive to fit velocity, all components fall back to supplied drone/IMU values.

`estimate_frame_depth()` then matches each current left-image point against the
same-frame right image. It requires rectified or aligned common-intrinsic images
and calibrated focal length, baseline, and disparity direction. It returns
per-layer matches, valid full-resolution `(u, v, Z)` map points, uncertainty,
and summary statistics. This depth is optical-axis depth—not world altitude.

The existing global affine/neighbor filters can reject points on surfaces with
different depths. This refactor does not replace the motion model with a full
visual-inertial optimizer. Validate navigation accuracy and timing with recorded
or simulated sequences before drawing conclusions from feature counts.

## Isolated large-structure mapping

This optional path is retained for future work but is not called by the simulator
or camera runners. GFTT on every pyramid layer lands mostly on grass: grass has strong gradients
everywhere while box tops and roofs are smooth. Coarse layers have already lost
the grass blades to `pyrDown`, so they look for *structure* instead:

```text
Frame.pyramids --levels 2,3--> LargeFeatureDetector --> lines, corners, smooth quads
Frame.pyramids --level 1-----> DenseStereo (SGBM) ----> DepthMap
                                    map_points() ------> TerrainMap.add(weight_scale=...)
```

| Module | Does |
| --- | --- |
| `largeFeatures.py` | Per layer: texture map `std / (mean + 16)` (grass high, faces low); LSD lines kept only when one side is smooth; right-angle corners; convex smooth polygons (`quads`). All coordinates in full-resolution pixels. `structure_points()` and `draw_large_features()` for later stages. |
| `denseStereo.py` | `StereoRectifier` (identity in the sim; `.npz` from `scripts/calibrate_stereo.py` or ROS `left.yaml`/`right.yaml` for real cameras). `DenseStereo` runs SGBM on an existing layer for any axis-aligned baseline and invalidates textureless pixels (SGBM invents depth there). `fill_quad_plane()` gives smooth faces depth from a plane fit or, when they have no texture at all, by sliding the face's silhouette along the baseline. |
| `structureMap.py` | `StructureMapper.step(frame, pos, quat)` ties it together and feeds `TerrainMap`; structure points get 4x weight so roof edges stay sharp. |

When re-enabled, `StructureMapper.step` can combine structural points with
dense depth and feed a `TerrainMap`. Its earlier measured cost at 1920x1080 was
~16 ms detection (levels 2-3; level 1 adds ~50 ms of LSD), ~25 ms SGBM, ~10 ms fusion.

If re-enabled, the mapper measures the real left-to-right image shift on its
first frame and warns (and uses the measurement) when
`cam_cfg.disparity_direction()` disagrees.

The active visual estimator continues to use the temporal GFTT/LK tracking and
sparse stereo depth path described above.

## Checks

Run from the repository root with its Python environment (the test folders are
not packages, so name the modules):

```sh
.venv-py313/bin/python -m unittest swarm.compVision.tests.test_large_features swarm.compVision.tests.test_dense_stereo swarm.compVision.tests.test_structure_map swarm.compVision.tests.test_frame
```

These are headless module/integration checks. No Genesis session or physical
camera is required. They cover preprocessing, structural proposals, fine tracking,
analytic and synthetic stereo depth, current-frame correspondence, velocity units,
failure paths, and the existing camera runner interface. They are not a flight
accuracy benchmark.
