# Sparse stereo depth and 3D map

Run the existing simulation (`python -m swarm.main`) with its normal environment.
The tracking windows still show each layer's features. **Stereo depth** shows
measured depth in metres, with a labelled per-frame colour range. **Stereo point
map — simulation pose** accumulates 3D observations, coloured by feature layer.

The first mapping baseline registers observations using the simulator's body
position and quaternion captured in the camera step. It is a reconstruction
validation map, not yet an independent visual odometry/SLAM solution. Ground-truth
pose does not enter stereo matching or determine point depth. The existing motion
controller and velocity estimator are unchanged.

## Data

- `est.all_points[layer]`: full-resolution `(u, v)` temporal tracks.
- `est.depth_by_layer[layer]`: row-aligned measured right points, depth, validity,
  patch score and approximate depth uncertainty. Invalid depth/right points are NaN.
- `est.depth_points`: valid, cross-layer deduplicated `(u, v, Z)` observations.
- `est.depth_colors`, `est.depth_sigma`: aligned BGR colours and depth uncertainties.
- `drone.point_map.points()`: accumulated world XYZ in metres; `.colors()` gives RGB floats.

`Z` is optical-axis depth, not distance along the ray and not drone altitude.
Ground and block roofs at the same image location have different depths. The old
altitude-minus-depth error plot was removed because it conflated these quantities.

## Matching and geometry

Temporal tracking is independent of stereo success. Each tracked point is searched
along the epipolar line in the current right image at the corresponding pyramid
resolution. Normalized patch correlation rejects weak and ambiguous matches; local
LK refines the match and reverse tracking checks consistency. Signed disparity,
epipolar error, image bounds, depth range and relative uncertainty reject bad depth.
Each level uses its correct pixel scale before `Z = focal * baseline / disparity`.
Predicted right points from temporal flow are never used as depth measurements.

The simulation requires equal focal lengths and parallel camera axes with transverse
baseline. Explicit camera up vectors make rendering and backprojection agree. The
USB runner keeps stereo disabled until real calibration and rectification are supplied.

## Parameters and limits

`SparseStereoDepth` defaults: 15-pixel patches, correlation >= 0.80, uniqueness
margin 0.06 outside a two-pixel peak neighbourhood, 512 full-resolution pixels maximum
disparity, depth 0.25–30 m, forward/backward and epipolar tolerances 1 full-resolution
pixel. Uncertainty assumes 0.25 native pixels disparity error; relative uncertainty
over 15% is rejected. This is an approximation, not calibrated confidence.

`PointMap` defaults: 5 cm 3D voxels and 50,000 retained voxels, evicting the least
recently observed when full. One weighted observation per voxel per camera frame
prevents duplicate layers multiplying confidence. Multiple heights remain separate.
Image triangles are tracking overlays only: the map does not invent surfaces between
sparse points. Flat roofs, occlusions, repeated grass and shadows can yield holes or
wrong matches; rejected observations are not filled with drone height.

Tests use known stereo shifts, separate 3 m/6 m surfaces, changing disparity over
time, horizontal/vertical baselines, invalid/ambiguous matches, camera offsets,
rotated poses, multiple surface heights and bounded map storage. Full navigation
accuracy and timing require evaluation on flight sequences, including pitch/roll,
vegetation, shadows and independent ground-truth depth/motion.

## Initial rendered scene check

A three-frame 640×480 Genesis sample used the current plain-ground/block scene,
three pyramid levels, a 1,500-point budget, 3 m camera height, 500 px focal length,
0.125 m baseline and 1 cm horizontal increments. The two frames with temporal
tracks produced 66 accumulated voxels. Accepted stereo depth had 0.0654 m mean
absolute error and 0.1155 m RMSE against the renderer's optical-axis depth at the
sampled pixels. Sparse tracking/depth processing took 15.2 ms and 14.5 ms on this
machine (rendering and the separate dense structure mapper excluded). This small
smoke check is not a full-resolution or flight benchmark. Textureless ground and
smooth roofs remain sparsely observed; the map does not fill those gaps.
