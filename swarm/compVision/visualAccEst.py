import cv2
import numpy as np
from collections import deque
import time
import swarm.config.cameraConfig as camConfig
from swarm.compVision.visualDepthEst import triangulate_altitude
from swarm.compVision.smallFeatures import replenish, build_triangles, filter_by_neighbors
from swarm.compVision.frame import Frame, FrameProcessor

# Preserve the original public frame name.
EnumFrame = Frame


class VisulaAcEst:
    def __init__(self, res, fov):
        self.fov = fov
        self.foc_l = (res[1] / 2) / np.tan(np.deg2rad(fov) / 2)
        self.dt = 0.01
        self.camera_saperation = camConfig.STEREO_CAM.baseline

        # Only one grayscale level is needed by the current tracker.
        self.frame_processor = FrameProcessor(pyramid_levels=1)
        self.resolution = tuple(res)  # calibrated output (width, height)
        self.use_stereo_depth = True
        self.current_frame = EnumFrame()
        self.buffer = deque(maxlen=100)

        self.drone_pos = np.zeros(3)
        self.drone_vel = np.zeros(3)
        self.drone_ang_vel = np.zeros(3)

        self.imu_att = np.zeros(3)
        self.previous_cmd_vel = np.zeros(4)

        self.buf_ang_vel = deque(maxlen=20)
        self.buf_att = deque(maxlen=20)
        self.buf_pos = deque(maxlen=20)
        self.avg_ang_vel = np.zeros(3)
        self.avg_att = np.zeros(3)
        self.avg_pos = np.zeros(3)

        self.tracked_points = {}
        self.vkfs = {}

        self.camera_velocity = np.zeros(4)   # [vx, vy, vz, yaw_rate], body frame
        self.depth_median = np.nan       # median stereo depth of valid matches
        self.depth_valid_frac = 0.0      # share of points with a valid stereo match
        self.depth_points = np.empty((0, 3))   # (u, v, depth) of valid matches, left image
        self.depth_frame_idx = -1              # camera frame the depth_points belong to

    # ---------------- frames ----------------

 

    @property
    def image_filters(self):
        return self.frame_processor.image_filters

    @image_filters.setter
    def image_filters(self, value):
        self.frame_processor.image_filters = value

    @property
    def filter_kernel_size(self):
        return self.frame_processor.filter_kernel_size

    @filter_kernel_size.setter
    def filter_kernel_size(self, value):
        self.frame_processor.filter_kernel_size = value

    @property
    def clahe(self):
        return self.frame_processor.clahe

    @clahe.setter
    def clahe(self, value):
        self.frame_processor.clahe = value

    def preprocess_frame(self, frame):
        """Compatibility adapter for filtering an already-normalized BGR image."""
        return self.frame_processor.preprocess(frame)

    def update_frame(self, frame, idx):
        prepared = self.frame_processor.process(frame, idx)
        height, width = prepared.frames[0].shape[:2]
        if (width, height) != self.resolution:
            raise ValueError('Frame resolution differs from calibration; create a new estimator for this output size')
        self.frame_size = prepared.frames[0].shape
        self.buffer.append(prepared)
        self.current_frame = prepared.display_copy()

    def processing(self, frame, idx):
        self.update_frame(frame, idx)
        self.aply_flow()
        return self.current_frame


    # ---------------- sensors ----------------

    def push_sensors(self):
        self.buf_ang_vel.append(np.array(self.drone_ang_vel, dtype=float))
        self.buf_att.append(np.array(self.imu_att, dtype=float))
        self.buf_pos.append(np.array(self.drone_pos, dtype=float))

    def _take_sensor_window(self):
        if len(self.buf_ang_vel) > 0:
            self.avg_ang_vel = np.mean(np.array(self.buf_ang_vel), axis=0)
            self.avg_att = np.mean(np.array(self.buf_att), axis=0)
            self.avg_pos = np.mean(np.array(self.buf_pos), axis=0)
        else:
            self.avg_ang_vel = np.array(self.drone_ang_vel, dtype=float)
            self.avg_att = np.array(self.imu_att, dtype=float)
            self.avg_pos = np.array(self.drone_pos, dtype=float)
        self.buf_ang_vel.clear()
        self.buf_att.clear()
        self.buf_pos.clear()

    def _current_height(self):
        return float(self.avg_pos[2])

    # ---------------- pipeline ----------------

    def aply_flow(self):
        self._take_sensor_window()
        if len(self.buffer) < 2:
            return

        prev_data, curr_data = self.buffer[-2], self.buffer[-1]
        prev, curr = prev_data.frames, curr_data.frames
        if min(len(prev), len(curr)) < 2:
            self.camera_velocity = self._fallback_velocity()
            return

        processed, flow, olds, news, tris, links = self.lucas_kanade_flow(
            prev[0], curr[0], prev[1], curr[1],
            tracking_gray=(prev_data.gray_frames[0], curr_data.gray_frames[0], prev_data.gray_frames[1]),
        )
        self.current_frame.frames = processed
        if flow is None:
            self.camera_velocity = self._fallback_velocity()
            return

        # news = self.trinagulate_altitude(news, flow, olds, tris, links)

        keep = self.point_prediction_filtering(olds[0], news[0])
        flow, news = flow[keep], [news[0][keep], news[1][keep]]
        self.estimate_velocities(flow, news[0])

# ---------------- tracking ----------------

    def lucas_kanade_flow(self, frame1, frame2, frame_r, frame_r_curr, tracking_gray=None):
        gray1, gray2, gray_r = tracking_gray
        annotated_frame = frame2.copy()
        annotated_r = frame_r_curr.copy()

        lost = ([annotated_frame, annotated_r], None, None, None, None, None)
        empty = np.empty((0, 2), np.float32)

        lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        lk_stereo = dict(winSize=(100, 100), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        def bad_return():
            self.tracked_points[0] = empty
            return lost

        pts = self._replenish(gray1, self.tracked_points.get(0, empty), 0)
    
        if len(pts) < 3: return bad_return()
        old = pts.reshape(-1, 1, 2).astype(np.float32)
        new, st_f, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, old, None, **lk)

        if new is None: return bad_return()
        back, st_b, _ = cv2.calcOpticalFlowPyrLK(gray2, gray1, new, None, **lk)  # ✓ new still (N, 1, 2)
        old, new, back = old.reshape(-1, 2), new.reshape(-1, 2), back.reshape(-1, 2)
        good = (st_f.ravel() == 1) & (st_b.ravel() == 1) & (np.abs(old - back).max(axis=1) < 1.0)
        good_old, good_new = old[good], new[good]
       
        if len(good_old) >= 6:
            _, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=2000, confidence=0.99)
            inliers = inliers.ravel().astype(bool)
            good_old, good_new = good_old[inliers], good_new[inliers]

        for _ in range(2):
            if len(good_new) < 3: break
            _, _, links, _ = self._build_triangles(good_new)
            good_old, good_new = self.filter_by_neighbors(good_old, good_new, links)

        old_r = empty
        if len(good_old) >= 3:
            p = good_old.reshape(-1, 1, 2).astype(np.float32)
            old_r, st_r, _ = cv2.calcOpticalFlowPyrLK(gray1, gray_r, p, None, **lk_stereo)
            if old_r is None: return bad_return()
                
            old_r = old_r.reshape(-1, 2)
            good_old, good_new, old_r = good_old[st_r.ravel() == 1], good_new[st_r.ravel() == 1], old_r[st_r.ravel() == 1]

        if len(good_new) < 3: return bad_return()

        triangles, _, links, _ = self._build_triangles(good_new)
        self.tracked_points[0] = good_new.copy()
        flow, scale, div = self.compensate_vertical(good_old, good_new, triangles)
       
        new_r = old_r + flow

        z = self._current_height()
        olds = [np.hstack((good_old, np.full((len(good_old), 1), z))), np.hstack((old_r, np.full((len(old_r), 1), z)))]
        news = [np.hstack((good_new, np.full((len(good_new), 1), z))), np.hstack((new_r, np.full((len(new_r), 1), z)))]
        
        annotated_frame, annotated_r = self.drawing(annotated_frame, annotated_r, good_new, new_r, links)
        return [annotated_frame, annotated_r], flow, olds, news, triangles, links 


    def _replenish(self, gray, pts, cam, max_points=300):
        """Compatibility adapter for smallFeratures.replenish."""
        return replenish(gray, pts, cam, max_points)

    # ---------------- mesh ----------------

    def compensate_vertical(self, good_old, good_new, triangles, min_area=4.0):
        flow = good_new - good_old
        if len(triangles) == 0: return flow, 1.0, 0.0
        def area(p): return 0.5 * np.abs(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]))

        a_old = area(good_old[triangles])
        a_new = area(good_new[triangles])
        ok = a_old > min_area
        if not ok.any(): return flow, 1.0, 0.0

        s = np.sqrt(np.median(a_new[ok] / a_old[ok]))
        h, w = self.frame_size[:2]
        c = np.array([w / 2, h / 2])

        flow_comp = flow - (s - 1) * (good_old - c)
        div = np.log(s) / self.dt
        return flow_comp, s, div


    def _build_triangles(self, points, max_edge=200.0, min_area=20.0):
        """Compatibility adapter for smallFeratures.build_triangles."""
        return build_triangles(points, max_edge, min_area)


    def filter_by_neighbors(self, good_old, good_new, links, thresh=2.0, min_links=1, neighbor_min_links=3):
        """Compatibility adapter for smallFeratures.filter_by_neighbors."""
        return filter_by_neighbors(good_old, good_new, links, thresh, min_links, neighbor_min_links)


    def trinagulate_altitude(self, news, flow, olds, triangles, links,
                            min_disparity_px=0.3, max_epipolar_error_px=1.0) -> list:
        """Compatibility adapter; stereo geometry lives in depth.py."""
        direction = getattr(self, 'disparity_direction', None)
        if direction is None:
            direction = camConfig.STEREO_CAM.disparity_direction()
        triangulate_altitude(
            news, olds, focal_px=self.foc_l, baseline=self.camera_saperation,
            disparity_direction=direction, frame_size=getattr(self, 'frame_size', None),
            use_stereo_depth=getattr(self, 'use_stereo_depth', True),
            min_disparity_px=min_disparity_px,
            max_epipolar_error_px=max_epipolar_error_px,
        )
        valid = np.isfinite(news[0][:, 2]) & (news[0][:, 2] > 0)
        self.depth_valid_frac = float(valid.mean()) if len(valid) else 0.0
        self.depth_median = float(np.median(news[0][valid, 2])) if valid.any() else np.nan
        self.depth_points = news[0][valid].copy()  # image coordinates (u, v, Z)
        self.depth_frame_idx = self.current_frame.idx
        return news


    def estimate_velocities(self, flow, points):
        if len(points) < 2:
            self.camera_velocity = self._fallback_velocity()
            return None

        ang_vel, att = self.avg_ang_vel, self.avg_att
        h, w = self.frame_size[:2]
        c = np.array([w / 2, h / 2])

        n = len(points)
        z = points[:, 2]

        k = max(1, int(len(z) * 0.10))
        z_biggest = np.partition(z, -k)[-k:]
        z_estimated = np.median(z_biggest) 

        v = flow * (z / (self.foc_l * self.dt))[:, None]
        w = ang_vel[[1, 0]] * np.array([1, 1])   # swapped
        v_rot = z[:, None] / np.cos(att[[1, 0]]) ** 2 * w

        B = v - v_rot
        r = (z[:, None] / self.foc_l) * (points[:, :2] - c) + self.camera_saperation / 2
        A = np.vstack([
            np.column_stack([np.ones(n), np.zeros(n), -r[:, 1]]),
            np.column_stack([np.zeros(n), np.ones(n),  r[:, 0]])
        ])
        x, *_ = np.linalg.lstsq(A, np.concatenate([B[:, 0], B[:, 1]]), rcond=None)
        x = -x

        kf = self.vkfs.setdefault(0, VelocityKalmanFilter())
        if not kf.started:
            kf.state = x.copy()
            kf.started = True

        kf.predict(kf.state)
        kf.update(x)
        
        self.camera_velocity = kf.get()
        self.camera_velocity = np.insert(self.camera_velocity, 2, z_estimated)
        

    @staticmethod
    def drawing(annotated_frame, annotated_r, good_new, new_r, links):
        """Centralized drawing: triangulation and points on left and right frames."""
        color = (255, 0, 0)      # Blue
         
        for a, b in links:
            cv2.line(annotated_frame, tuple(good_new[a].astype(int)), tuple(good_new[b].astype(int)), color, 2)
        for x, y in good_new.astype(int):
            cv2.circle(annotated_frame, (x, y), 2, color, -1)
        

        for a, b in links:
            cv2.line(annotated_r, tuple(new_r[a].astype(int)), tuple(new_r[b].astype(int)), color, 2)
        for x, y in new_r.astype(int):
            cv2.circle(annotated_r, (x, y), 2, color, -1)
        
        return annotated_frame, annotated_r


    def point_prediction_filtering(self, old, new):
        v = self.previous_cmd_vel[:3]
        ang = self.avg_ang_vel
        h, w = self.frame_size[:2]
        c = np.array([w / 2, h / 2])

        u = old[:, 0] - c[0]
        v_pix = old[:, 1] - c[1]
        z = old[:, 2]

        du = (-self.foc_l * v[0] + u * v[2]) / z * self.dt - v_pix * ang[2] * self.dt
        dv = (-self.foc_l * v[1] + v_pix * v[2]) / z * self.dt + u * ang[2] * self.dt
        predicted = np.column_stack([old[:, 0] + du, old[:, 1] + dv])

        residuals = np.clip(np.linalg.norm(new[:, :2] - predicted, axis=1), 0, 150)
        threshold = 1.5
        if residuals.sum() > 0:
            threshold = np.min(residuals) + 0.75 * (np.max(residuals) - np.min(residuals))
        return residuals < threshold
    
    # ---------------- utils ----------------

    def set_dt(self, dt):
        self.dt = dt

    def _fallback_velocity(self):
        print("fallback velocity used")
        # [vx, vy, vz, yaw_rate] -- same layout as the estimated camera_velocity
        return np.array([self.drone_vel[0], self.drone_vel[1], self.drone_vel[2], self.drone_ang_vel[2]])


class VelocityKalmanFilter:
    def __init__(self, process_var=0.6, measurement_var=4.5, yaw_process_var=5.0, yaw_measurement_var=1):
        self.state = np.zeros(3)
        self.P = np.eye(3)
        self.Q = np.diag([process_var, process_var, yaw_process_var])
        self.R = np.diag([measurement_var, measurement_var, yaw_measurement_var])
        self.started = False

    def predict(self, prior):
        self.state = np.asarray(prior, dtype=float)
        self.P = self.P + self.Q

    def update(self, measured):
        K = self.P @ np.linalg.inv(self.P + self.R)
        self.state = self.state + K @ (measured - self.state)
        self.P = (np.eye(3) - K) @ self.P

    def get(self):
        return self.state.copy()