# axis_check - where does each part think forward, right and up are?

The IMU, the camera, the visual estimator and the fused filter each have their own idea of the
drone's axes, and a single flipped sign anywhere shows up as drift, jumps or a filter fighting
itself. This tool records all of them at once while you move the drone by hand through seven
simple motions. Then, for every source, it reports which of its own axes moved and with what
sign, and lists the ones that disagree.

It only subscribes and never publishes, so it cannot move the motors.

## Run it on the drone

On the board, after `git pull --no-rebase` in `~/droneResearch`:

```bash
ssh -t ubuntu@192.168.137.124
source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=42
cd ~/droneResearch/axis_check && python3 axis_check.py record
```

Use `-t`: the steps wait for Enter. What has to be running first:

| | needed for |
| --- | --- |
| `start_rb3.py` (or the `drone-rb3` unit) | `/drone/imu`, `/drone/attitude`, the cameras. Required |
| `main.py` | `/vo/status`: the estimator's camera velocity, world velocity and fused velocity. Optional. Start it with the drone resting still, because it calibrates first. Keep the drone **disarmed with props off**: main.py publishes motor commands |

Hold the drone with the cameras 0.5-1.5 m above a floor with texture (not a plain white
floor). For each step: press Enter, keep still until **GO**, do the motion, and hold until it
says **done**. After every step it prints a one-line check, so a bad motion can be redone on
the spot (`r` + Enter).

| step | motion |
| --- | --- |
| still | hold still and level, 3 s |
| roll_right | tilt the right side down 20-30 deg |
| pitch_up | tilt the nose up 20-30 deg |
| yaw_right | turn the nose right (clockwise seen from above) 45-90 deg |
| forward | slide forward 30-50 cm, level |
| right | slide right 30-50 cm, level |
| up | lift straight up 30-50 cm, level |

The recording is saved as `axis_check_<date>_<time>.json`, and the report is printed.

## Re-analyse anywhere

```bash
python3 axis_check.py analyze axis_check_20261016_150000.json
```

This needs only numpy, so it also works on the laptop. Attitude is recomputed from the recorded
IMU with the control loop's own `sensors/IMU/IMU.py` and drone_gui's filter from the current
checkout, and the `Rb3ReadSend` signs are the ones saved at recording time. After a change to
`IMU.py`, re-running `analyze` on the old recording shows the effect. After a change to the
estimator, record again: `/vo/status` comes from the running `main.py`.

## Reading the report

* **Per step**: each cell is the source's own axis that moved (`+x 24`), `-` if nothing
  moved, `?` if the motion was not clearly on one axis. Rotations are in degrees. Accel cells
  are peak velocity in m/s (sign = which way it says the drone went), and camera and estimator
  cells are metres.
* **Where each source puts forward / right / up**: the same as one row per source, plus what
  its axes therefore are ("x = forward, y = right, z = down"). For gyros, "forward" is the axis
  of a right roll, "right" the axis of nose-up, and "up" minus the axis of a right turn.
* **Findings**: `ok`, `!!` (a problem) or `..` (worth knowing). Every check is against the
  control loop's gyro frame, because the estimator and the FKF use that frame:
  * gyro handedness; accelerometer vs gyro (a negated accelerometer makes FKF integrate every
    acceleration backwards);
  * camera mounting vs `cameraConfig.py`;
  * the estimator's vx/vy/vz and yaw_rate vs the IMU, their scale against the camera image, and
    how much a pure tilt leaks into velocity: ~0 means the gyro compensation works, ~1 means it
    is missing, ~2 means its sign is wrong;
  * whether body -> world is a proper rotation or mirrored;
  * the attitude signs of the control loop, drone_gui and the FC.

Frames used: the body is **FRD** (x forward, y right, z down) and rotations follow the
right-hand rule, so a right roll, nose-up and a right turn are all positive. The camera image is
in OpenCV axes: x image right, y image down, z the optical axis. The camera's motion is measured
here with OpenCV (tracked corners), independently of the team's estimator.

## Self-test

```bash
python3 axis_check.py selftest -v
```

This needs no hardware. It simulates two recordings and checks that the analysis finds what
was built into them:
* one where the IMU reads exactly as `/drone/imu` does today, the camera is mounted as
  `cameraConfig.py` says, and the estimator is right;
* one with a camera turned 90 deg, a flipped estimator y, 10x scale, gyro compensation with
  the wrong sign, and a negated yaw_rate.

It also checks the image-motion measurement against a known warp.
