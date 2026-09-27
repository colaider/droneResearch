# Control and estimation corrections

## Run and test

Use Python 3.11 and install from the repository root:

```sh
python -m pip install -e '.[swarm]'
python -m pip install pytest
python -m pytest -q
python -m tests.smoke_simulation
python -m tests.smoke_swarm
python -m swarm.main
python -m scripts.eval.pos_ctrl_eval
python -m scripts.eval.se3_controller_eval --use-trajectory
```

The quick tests cover real NumPy/OpenCV/Torch calculations with lightweight drone
objects; the separate smoke test runs actual Genesis physics on the CPU without
a viewer. GUI camera testing requires a working OpenGL display.

Scripts use CUDA when available and CPU otherwise. The RL training configuration
still requests 12,000 environments: reduce `num_envs` before training on a CPU.
Assets/config paths are resolved relative to the source tree. Existing user path
fixes are preserved and extended to support another working directory.

## Behavioral changes

- A library action is applied before the corresponding physics step. Odometry is
  updated once afterwards. Resets teleport only the selected environments, clear
  their controller/history buffers, and synchronize position without advancing
  the scene. The next action overwrites motor commands before physics advances.
- RL reset and target-change observations are rebuilt consistently. Previous
  actions refer to the action just executed; reset rows contain zero. Timeouts
  are reported separately and angular termination thresholds are converted from
  configured degrees to radians. Observation normalization is opt-in so the
  existing checkpoint's input scales are not silently changed.
- Position mode uses hover collective plus its altitude correction; RC throttle
  stays normalized until conversion to RPM. Position PID integration/differentiation
  uses seconds, world errors are rotated by yaw, and angle errors wrap at pi.
- SE3 works without an obligatory randomization call, does not mutate input
  quaternions, returns matching quaternions/rotation matrices in both Hopf charts,
  expresses desired angular rate in the current body frame, and limits rotor
  speed. Allocation coefficients, rotor ordering/direction and lever arms match
  the supplied URDF's thrust coefficient and propeller inertial offsets.
- The swarm estimator uses world-frame velocity/position and quaternion body-rate
  integration. Low-level attitude calculations convert world velocity to body
  coordinates. Command clipping now takes effect; position damping has the
  stabilizing sign; integrators are bounded.
- Camera timing uses the actual elapsed simulation time (30 Hz on a 100 Hz
  simulator alternates 0.03/0.04 s). Only a fresh, valid estimate corrects velocity.
  Image history contains raw images, not previously drawn feature markers.

## Vision model and remaining limitations

The two cameras explicitly face down along body -z. OpenGL camera axes coincide
with the drone body axes; image x is body +x and image y is body -y when level.
The estimator uses Genesis's vertical field of view and each camera's lever arm.
Tracked rays are intersected with a static world z=0 plane at the supplied
altitude. Differences between successive intersections estimate world velocity
while compensating IMU attitude changes. Robust residual rejection removes bad
tracks; each camera has its own velocity filter. Blank frames, lost tracks,
near-ground views and rays near the horizon produce an invalid measurement and
fall back to the IMU estimate without visual fusion.

This is altitude-assisted planar optical-flow odometry, not stereo depth or a
full visual-inertial SLAM system. Non-planar terrain, moving objects dominating
the image, and attitude/altitude errors can still bias it. The noisy IMU retains
an explicit 5% simulator-velocity correction (`velocity_truth_weight`); set it to
zero for unassisted dead reckoning. `get_simulated_altitude()` is ground truth for
plotting, not a barometer. Velocity-based fusion does not eliminate absolute
position drift. The legacy class spellings are retained for existing imports.

These control, timing and observation-history changes alter the dynamics seen
by old learned policies. Existing checkpoints should be reevaluated and normally
retrained; preserving observation scales does not guarantee equivalent behavior.
Controller gains require flight-level tuning beyond these regression checks.

Swarm coordination, map loading and a live ROS bridge remain separate unfinished
features. Requests for the corresponding unimplemented capabilities now raise
clear errors instead of silently doing nothing. `ros_bridge.py` remains a
synthetic ROS example. The default RC configuration disables unused map loading;
set `render_cam: True` to enable its optional FPV view (batch rendering needs a
compatible GPU). RC hardware and long-duration training are not covered by tests.

## Verification

Validated with the local Python 3.11 environment: Genesis 1.4.2, Torch 2.14.0,
OpenCV 5.0.0 and rsl-rl-lib 3.1.3. Quick regression tests cover 19 cases. The
physics smoke check covers hover, isolated resets, closed-loop climb/descent,
and one small PPO update; the sensor smoke check renders both cameras and runs
12 IMU/visual-control steps. A wheel build verifies packaging and bundled assets.
These checks are short regression checks, not long-duration flight qualification.
