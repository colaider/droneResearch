"""Ground-plane optical-flow odometry for the downward cameras in CustomScene.

This is NOT stereo reconstruction. Scale comes from the supplied altitude and
rotation from the IMU. World z=0 must be a static plane. Both cameras independently
estimate world velocity; valid estimates are fused after rotation compensation.
"""
from collections import deque
import cv2
import numpy as np
from swarm.utilities.geometry import rotation


class EnumFrame:
    """Container for display images and their camera-sample index.

    ``frames`` is [left_image, right_image], each uint8 (height, width, 3)
    in OpenCV BGR order. ``idx`` counts camera samples, not physics steps.
    """
    def __init__(self):
        """Create an empty image list and initialize the sample index to zero."""
        self.frames = []
        self.idx = 0


class VisulaAcEst:
    """Estimate drone velocity from ground features seen in two downward cameras.

    Before processing a sample, DroneVisulaCTRL supplies these NumPy arrays:
        drone_pos: (3,) estimated world [x, y, z] in metres. Only altitude z
            is used for ray/ground intersections; x and y need not be known.
        drone_vel: (3,) estimated world [vx, vy, vz] in m/s, used as a filter
            prior and as the fallback when images are unusable.
        imu_att: (3,) body roll, pitch, yaw relative to world, in radians.
        drone_ang_vel: (3,) body-axis angular velocity [p, q, r] in rad/s.
        camera_saperation: distance between camera centres in metres; the
            legacy misspelling is retained to match DroneStruct.

    Outputs are stored on the object rather than returned as velocity:
        camera_velocity: (3,) world velocity in m/s. Check valid before fusion.
        valid: True if at least one camera supplied a usable measurement.
        expected_vel_err: mean inlier velocity residual in m/s, or infinity
            on failure. This is a consistency score, NOT a calibrated error
            bound or Kalman covariance.
        camera_yaw_rate: last accepted sample's body z gyro reading (rad/s),
            NOT an independent visual yaw-rate estimate or exact Euler yaw rate.

    Images are paired over time separately for each camera; there is no
    left-to-right feature matching. Metric scale relies on altitude above
    a stationary flat ground plane, so this is not general stereo odometry.
    """
    def __init__(self, res, fov, noise_sigma=0.0):
        """Set camera calibration and initialize estimator state.

        Args:
            res: (width, height) image resolution in pixels, matching both cameras.
                NumPy image arrays use the opposite dimension order: (height, width, 3).
            fov: Vertical field of view in degrees, as defined by Genesis.
                Square pixels and a centred principal point are assumed.
            noise_sigma: Optional Gaussian noise standard deviation in image
                intensity units (0..255). Zero leaves input intensities unchanged.

        The initial dt is 1/30 s; the caller must supply the actual interval with
        set_dt() before each later camera sample. Each camera owns its own filter.
        """
        self.width, self.height = res
        # Genesis defines fov vertically; pixels are square.
        self.foc_l = (self.height / 2) / np.tan(np.deg2rad(fov) / 2)
        # Principal point [u0, v0] in pixels; f=(height/2)/tan(vertical_fov/2).
        self.center = np.array([self.width / 2, self.height / 2])
        self.dt = 1 / 30
        self.noise_sigma = noise_sigma
        self.camera_saperation = 0.0  # Retained spelling for DroneStruct compatibility.
        self.drone_pos = np.zeros(3)
        self.drone_vel = np.zeros(3)
        self.imu_att = np.zeros(3)
        self.drone_ang_vel = np.zeros(3)
        self.camera_velocity = np.zeros(3)
        self.camera_yaw_rate = 0.0
        self.expected_vel_err = np.inf
        self.valid = False
        self.buffer = deque(maxlen=2)
        self.current_frame = EnumFrame()
        self.filters = [VelocityKalmanFilter(), VelocityKalmanFilter()]
        self.previous_pose = None

    def set_dt(self, dt):
        """Set elapsed simulation time between consecutive camera samples.

        Args:
            dt: Positive finite interval in seconds, not the physics timestep.
                At 100 Hz physics and nominal 30 Hz imaging, this can be 0.03 or
                0.04 s. The velocity calculation divides displacement by this value.

        Raises:
            ValueError: If dt is non-positive or non-finite.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Camera interval must be positive and finite")
        self.dt = float(dt)

    def processing(self, frames, idx):
        """Process one synchronized camera pair and update velocity/validity.

        Args:
            frames: [left, right] BGR uint8 arrays of shape (height, width, 3).
                Their resolution must match res passed to the constructor.
            idx: Camera sample counter used for display/bookkeeping; it does not
                determine elapsed time. Call set_dt() and update the pose fields first.

        Returns:
            List of annotated BGR images with tracked features drawn on them.
            Read self.camera_velocity and self.valid for the estimation result.

        On the first sample there is no earlier image, so valid is False. Later
        samples track features, convert tracks to ground-relative displacement,
        filter each usable camera estimate, and average accepted estimates.
        If neither camera works, return images and expose the IMU velocity fallback
        with valid=False so the caller does not treat it as a new visual measurement.
        """
        raw = self.add_noise(frames, self.noise_sigma)
        # Save a snapshot: the controller overwrites these fields next physics step.
        pose = (self.drone_pos.copy(), self.imu_att.copy())
        self.current_frame = EnumFrame()
        self.current_frame.idx = idx
        self.current_frame.frames = [f.copy() for f in raw]
        self.valid = False
        self.expected_vel_err = np.inf
        estimates = []
        errors = []
        if self.buffer and self.previous_pose is not None:
            for camera, (old_frame, new_frame) in enumerate(zip(self.buffer[-1], raw)):
                annotated, old, new = self.lucas_kanade_flow(old_frame, new_frame)
                self.current_frame.frames[camera] = annotated
                measurement = self.estimate_velocity(old, new, self.previous_pose, pose, camera)
                if measurement is not None:
                    velocity, error = measurement
                    self.filters[camera].predict(self.drone_vel, self.dt)
                    self.filters[camera].update(velocity)
                    estimates.append(self.filters[camera].get())
                    errors.append(error)
        if estimates:
            self.camera_velocity = np.mean(estimates, axis=0)
            self.camera_yaw_rate = float(self.drone_ang_vel[2])
            self.expected_vel_err = float(np.mean(errors))
            self.valid = True
        else:
            # A fallback is not a visual measurement and must not be fused as one.
            self.camera_velocity = np.nan_to_num(self.drone_vel.copy())
        self.buffer.append(raw)  # Never feed annotation graphics into optical flow.
        self.previous_pose = pose
        return self.current_frame.frames

    def lucas_kanade_flow(self, frame1, frame2):
        """Find corresponding feature pixels in two successive images from ONE camera.

        Args:
            frame1: Previous BGR uint8 image, shape (height, width, 3).
            frame2: Current image with the same shape, format, and camera viewpoint.

        Returns:
            (annotated, old, new): annotated is a copy of frame2 with markers.
            old and new are matching (N, 2) arrays of [u, v] pixel coordinates:
            u increases rightward, v downward, and row i refers to the same feature
            in both images. Pixel displacement is new - old, not yet metric velocity.
            If tracking fails, N=0; both point arrays retain shape (0, 2).

        Corner detection supplies textured points. Pyramidal Lucas-Kanade tracks
        their small image patches at multiple scales. Tracking back to frame1
        checks consistency and rejects lost or ambiguous matches.
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        annotated = frame2.copy()
        empty = np.empty((0, 2), dtype=float)
        old = cv2.goodFeaturesToTrack(gray1, maxCorners=500, qualityLevel=0.01,
                                      minDistance=7, blockSize=7)
        if old is None:
            return annotated, empty, empty
        # 21x21-pixel patches; pyramid levels 0..3; stop at 30 iterations
        # or when the point update is smaller than 0.01 pixels.
        options = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        new, forward, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, old, None, **options)
        if new is None or forward is None:
            return annotated, empty, empty
        back, backward, _ = cv2.calcOpticalFlowPyrLK(gray2, gray1, new, None, **options)
        if back is None or backward is None:
            return annotated, empty, empty
        old, new, back = (p.reshape(-1, 2) for p in (old, new, back))
        # forward/backward are OpenCV success flags. Keep tracks returning
        # within 0.5 px of their start and inside a central circular image region.
        mask = ((forward.ravel() == 1) & (backward.ravel() == 1)
                & np.isfinite(new).all(axis=1) & np.isfinite(back).all(axis=1)
                & (np.linalg.norm(old-back, axis=1) < 0.5)
                & (np.linalg.norm(new-self.center, axis=1) < min(self.width, self.height)*0.46))
        for point in new[mask]:
            cv2.circle(annotated, tuple(point.astype(int)), 3, (0, 100, 200), -1)
        return annotated, old[mask], new[mask]

    def ground_offsets(self, pixels, pose, camera):
        """Intersect camera rays with world z=0 and return drone-to-ground vectors.

        Args:
            pixels: (N, 2) array of [u, v] image coordinates in pixels.
            pose: (position, attitude), where position is world [x, y, z] in
                metres and attitude is [roll, pitch, yaw] in radians, both (3,).
                Only position[2] is needed; no absolute horizontal position is used.
            camera: Camera index: 0=left at body y=-baseline/2; 1=right at
                body y=+baseline/2. baseline is self.camera_saperation in metres.

        Returns:
            (offsets, valid): offsets is (N, 3), in world axes and metres, from
            the DRONE origin to the ground intersection. valid is an (N,) boolean
            mask. Ignore offsets for invalid rows (they are not real intersections).

        The cameras look along body -z; image right is body +x and image down is
        body -y when level. A camera's displacement from the drone origin (lever
        arm) rotates with the drone. Including it prevents pure drone rotation
        from being mistaken for translational motion of the drone origin.
        """
        position, attitude = pose
        R = rotation(attitude)  # (3, 3), maps body column vectors into world axes.
        lever = R @ np.array([0., (-1 if camera == 0 else 1)*self.camera_saperation/2, 0.])
        height = position[2] + lever[2]
        # Unnormalized pinhole rays: [(u-u0)/f, -(v-v0)/f, -1].
        rays = np.column_stack(((pixels-self.center)/self.foc_l, -np.ones(len(pixels))))
        rays[:, 1] *= -1
        rays = rays @ R.T  # Each ray is stored as a ROW, hence R transpose.
        # Reject near-horizontal/upward rays and camera height <= 5 cm.
        # -0.1 is a cutoff on this unnormalized ray z component, not an angle.
        valid = (np.isfinite(rays).all(axis=1) & (rays[:, 2] < -0.1)
                 & np.isfinite(height) & (height > 0.05))
        # Solve camera_z + scale * ray_z = 0. "depth" is this scale in metres;
        # it is not Euclidean distance because the rays are not unit vectors.
        depth = np.zeros(len(pixels))
        np.divide(-height, rays[:, 2], out=depth, where=valid)
        return lever + rays * depth[:, None], valid

    def estimate_velocity(self, old, new, old_pose, new_pose, camera):
        """Convert one camera's matched feature tracks into world velocity.

        Args:
            old, new: Matching (N, 2) arrays of [u, v] pixel coordinates returned
                by lucas_kanade_flow(); row i identifies one stationary ground point.
            old_pose, new_pose: Poses at the corresponding image times, each
                (world_position_metres, roll_pitch_yaw_radians), with (3,) arrays.
            camera: 0 for the left camera or 1 for the right camera.

        Returns:
            (velocity, error) on success: velocity is world [vx, vy, vz], shape
            (3,), in m/s; error is mean inlier residual in m/s. Returns None when
            there are too few reliable tracks or an implausible/non-finite result.

        For a stationary point G, G = p_old + offset_old = p_new + offset_new.
        Therefore p_new - p_old = offset_old - offset_new. Divide by self.dt to
        get velocity. Attitude-dependent ray projection compensates rotation.
        Vertical displacement comes from the supplied altitude change, so vz
        is not an independent visual estimate of altitude motion.
        """
        if len(old) < 4:
            return None
        old_offsets, good_old = self.ground_offsets(old, old_pose, camera)
        new_offsets, good_new = self.ground_offsets(new, new_pose, camera)
        velocities = (old_offsets-new_offsets)[good_old & good_new] / self.dt
        if len(velocities) < 4:
            return None
        median = np.median(velocities, axis=0)
        residual = np.linalg.norm(velocities-median, axis=1)
        # Reject velocity outliers around the median. The 0.05 m/s floor
        # avoids rejecting nearly identical tracks due to numerical noise.
        threshold = max(0.05, 3.0*np.median(residual))
        inliers = residual <= threshold
        if inliers.sum() < 4:
            return None
        velocity = np.mean(velocities[inliers], axis=0)
        # The 20 m/s cap is a plausibility limit for this small-flight demo.
        if not np.isfinite(velocity).all() or np.linalg.norm(velocity) > 20:
            return None
        return velocity, float(np.mean(residual[inliers]))

    @staticmethod
    def add_noise(frames, sigma=0.0):
        """Copy camera images, optionally adding independent pixel noise.

        Args:
            frames: Iterable of uint8 BGR images, each (height, width, 3).
            sigma: Gaussian standard deviation in pixel intensity units, not
                metres or sensor acceleration. Zero only copies the arrays.

        Returns:
            New list of uint8 images, clipped to [0, 255]; inputs are not modified.
        """
        if sigma == 0:
            return [frame.copy() for frame in frames]
        return [np.clip(frame.astype(float)+np.random.normal(0, sigma, frame.shape),
                        0, 255).astype(np.uint8) for frame in frames]


class VelocityKalmanFilter:
    """Filter one camera's world velocity, using IMU velocity as the prior.

    state is [vx, vy, vz] in m/s. P is a 3x3 state-error covariance, Q is a
    covariance-growth rate, and R is the measurement-error covariance.
    The observation matrix H is identity: measurements directly observe velocity.
    This small filter does not estimate attitude, position, or IMU biases.
    """
    def __init__(self, process_var=1.0, measurement_var=3.0):
        """Initialize three independent velocity axes with equal uncertainty.

        Args:
            process_var: Per-axis covariance growth per second, (m/s)^2/s.
                Larger values make predictions lose confidence faster.
            measurement_var: Per-axis optical-flow velocity variance, (m/s)^2.
                Larger values reduce the weight of visual measurements.

        These scalar tuning values form diagonal Q and R matrices; initial P
        is identity in (m/s)^2. Values should be non-negative, with positive
        measurement variance to keep the update well-conditioned.
        """
        self.state = np.zeros(3)
        self.P = np.eye(3)
        self.Q = np.eye(3) * process_var
        self.R = np.eye(3) * measurement_var

    def predict(self, velocity, dt=1.0):
        """Use the current IMU-based velocity as the prior for this camera sample.

        Args:
            velocity: (3,) estimated world [vx, vy, vz] in m/s; copied internally.
            dt: Elapsed seconds since the preceding camera sample (default 1 s).

        Replaces the prior mean and adds Q*dt to covariance. This is an externally
        supplied velocity prior, not acceleration integration inside this filter.
        """
        self.state = np.asarray(velocity, dtype=float).copy()
        self.P += self.Q * dt

    def update(self, measurement):
        """Correct the prior with a visual velocity observation.

        Args:
            measurement: (3,) world [vx, vy, vz] in m/s, in the same axes as state.

        Updates state and P in place; returns None. With H=I, the Kalman gain is
        K=P(P+R)^-1. The innovation measurement-state is weighted by K. The Joseph
        covariance update below helps preserve symmetry and non-negative covariance.
        """
        # Solve a linear system rather than explicitly forming a matrix inverse.
        K = np.linalg.solve((self.P+self.R).T, self.P.T).T
        self.state += K @ (measurement-self.state)
        I_K = np.eye(3)-K
        self.P = I_K @ self.P @ I_K.T + K @ self.R @ K.T

    def get(self):
        """Return a copy of the filtered world [vx, vy, vz] vector, shape (3,), in m/s."""
        return self.state.copy()
