import cv2
import numpy as np
from collections import deque
from scipy.spatial import Delaunay
import time

class EnumFrame:
    def __init__(self):
        self.frames = []
        self.idx = 0
        self.time = 0.0


class VisulaAcEst:
    def __init__(self, res, fov):
        self.fov = fov
        self.foc_l = (res[1] / 2) / np.tan(np.deg2rad(fov) / 2)
        self.dt = 0.01
        self.camera_saperation = 0

        self.current_frame = EnumFrame()
        self.buffer = deque(maxlen=100)

        self.drone_pos = np.zeros(3)
        self.drone_vel = np.zeros(3)
        self.drone_ang_vel = np.zeros(3)
        self.drone_ang = np.zeros(2)
        self.imu_att = np.zeros(3)
        self.previous_cmd_vel = np.zeros(4)

        self.buf_ang_vel = deque(maxlen=20)
        self.buf_att = deque(maxlen=20)
        self.buf_pos = deque(maxlen=20)
        self.avg_ang_vel = np.zeros(3)
        self.avg_att = np.zeros(3)
        self.avg_pos = np.zeros(3)

        self.tracked_points = {}
        self.prev_med_flow = {}
        self.vkfs = {}

        self.camera_velocity = np.zeros(3)
        self.camera_angle = np.zeros(2)
        self.expected_vel_err = 0

    # ---------------- frames ----------------

    def update_frame(self, frame, idx):
        new_frame_data = EnumFrame()
        new_frame_data.frames = self.add_noise([f.copy() for f in frame])
        
        new_frame_data.idx = idx
        new_frame_data.time = time.time()

        if not hasattr(self, 'frame_size'):
            self.frame_size = np.shape(frame[0])

        self.buffer.append(new_frame_data)
        self.current_frame = new_frame_data

    def processing(self, frame, idx):
        self.update_frame(frame, idx)
        self.aply_flow()
        return self.current_frame.frames

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

        prev, curr = self.buffer[-2].frames, self.buffer[-1].frames
        if min(len(prev), len(curr)) < 2:
            self.camera_velocity = self._fallback_velocity()
            return

        processed, flow, olds, news, tris, links = self.lucas_kanade_flow(prev[0], curr[0], prev[1], curr[1])
        self.current_frame.frames = processed
        if flow is None:
            self.camera_velocity = self._fallback_velocity()
            return




        olds, news = self.trinagulate_altitude(news, flow, olds, tris, links)

        keep = self.point_prediction_filtering(olds[0], news[0])
        flow, news = flow[keep], [news[0][keep], news[1][keep]]
        self.estimate_velocities(flow, news[0])

# ---------------- tracking ----------------

    def lucas_kanade_flow(self, frame1, frame2, frame_r, frame_r_curr):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        annotated_frame = frame2.copy()
        annotated_r = frame_r_curr.copy()
        lost = ([annotated_frame, annotated_r], None, None, None, None, None)
        empty = np.empty((0, 2), np.float32)

        lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        pts = self._replenish(gray1, self.tracked_points.get(0, empty), 0)
        if len(pts) < 3:
            self.tracked_points[0] = empty
            return lost

        old = pts.reshape(-1, 1, 2).astype(np.float32)
        new, st_f, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, old, None, **lk)

        if new is None:
            self.tracked_points[0] = empty
            return lost

        back, st_b, _ = cv2.calcOpticalFlowPyrLK(gray2, gray1, new, None, **lk)

        old, new, back = old.reshape(-1, 2), new.reshape(-1, 2), back.reshape(-1, 2)
        good = (st_f.ravel() == 1) & (st_b.ravel() == 1) & (np.abs(old - back).max(axis=1) < 1.0)
        good_old, good_new = old[good], new[good]

        if len(good_old) >= 6:
            _, inliers = cv2.estimateAffinePartial2D(good_old, good_new,
                                                    method=cv2.RANSAC,
                                                    ransacReprojThreshold=2.0,
                                                    maxIters=2000,
                                                    confidence=0.99)
            if inliers is not None:
                inliers = inliers.ravel().astype(bool)
                good_old, good_new = good_old[inliers], good_new[inliers]

        for _ in range(2):
            if len(good_new) < 3:
                break
            _, _, links, _ = self._build_triangles(good_new)
            good_old, good_new = self.filter_by_neighbors(good_old, good_new, links)

        # match the left old points onto the right camera, reject unmatched ones
        old_r = empty
        if len(good_old) >= 3:
            p = good_old.reshape(-1, 1, 2).astype(np.float32)
            old_r, st_r, _ = cv2.calcOpticalFlowPyrLK(gray1, gray_r, p, None, **lk)
            if old_r is None:
                self.tracked_points[0] = empty
                return lost
            
            back_r, st_rb, _ = cv2.calcOpticalFlowPyrLK(gray_r, gray1, old_r, None, **lk)
            old_r, back_r = old_r.reshape(-1, 2), back_r.reshape(-1, 2)
            matched = (st_r.ravel() == 1) & (st_rb.ravel() == 1) & (np.abs(good_old - back_r).max(axis=1) < 1.0)
            good_old, good_new, old_r = good_old[matched], good_new[matched], old_r[matched]

        if len(good_new) < 3:
            self.tracked_points[0] = empty
            return lost

        triangles, corners, links, _ = self._build_triangles(good_new)
        for a, b in links: cv2.line(annotated_frame, tuple(corners[a].astype(int)), tuple(corners[b].astype(int)), (255, 255, 0), 1)
        for x, y in corners.astype(int): cv2.circle(annotated_frame, (x, y), 2, (0, 0, 255), -1)

        self.tracked_points[0] = good_new.copy()

        flow, scale, div = self.compensate_vertical(good_old, good_new, triangles)
        self.prev_med_flow[0] = np.median(flow, axis=0)

        # transport the left flow to the matching right points
        new_r = old_r + flow
        for a, b in links: cv2.line(annotated_r, tuple(new_r[a].astype(int)), tuple(new_r[b].astype(int)), (255, 255, 0), 1)
        for x, y in new_r.astype(int): cv2.circle(annotated_r, (x, y), 2, (0, 0, 255), -1)

        z = self._current_height()
        olds = [np.hstack((good_old, np.full((len(good_old), 1), z))),
                np.hstack((old_r, np.full((len(old_r), 1), z)))]
        news = [np.hstack((good_new, np.full((len(good_new), 1), z))),
                np.hstack((new_r, np.full((len(new_r), 1), z)))]
        
        
        return [annotated_frame, annotated_r], flow, olds, news, triangles, links 

    def _replenish(self, gray, pts, cam, max_points=100):
        h, w = gray.shape
        mask = np.full((h, w), 255, np.uint8)
        for x, y in pts.astype(int):
            cv2.circle(mask, (x, y), 7, 0, -1)

        cand = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.005, minDistance=15, blockSize=11, mask=mask)
        if cand is None: return pts
        cand = cand.reshape(-1, 2)
        edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((5, 5), np.uint8))

        # Keep only corners that are near an edge
        xs = np.clip(cand[:, 0].astype(int), 0, w - 1)
        ys = np.clip(cand[:, 1].astype(int), 0, h - 1)
        keep = edges[ys, xs] > 0
        cand = cand[keep]
        if len(cand) == 0: return pts

        # Quadrant balancing (unchanged)
        def quad(p):
            return (p[:, 0] >= w / 2).astype(int) + 2 * (p[:, 1] >= h / 2).astype(int)

        pt_q = quad(pts) if len(pts) else np.array([], dtype=int)
        cand_q = quad(cand)

        need = max_points - len(pts)
        if need <= 0: return pts
        existing = np.bincount(pt_q, minlength=4)
        target = (len(pts) + need) // 4
        quota = np.maximum(target - existing, 0)

        take = []
        for q in range(4):
            in_q = np.flatnonzero(cand_q == q)
            take.extend(in_q[:quota[q]])

        remaining_need = need - len(take)
        if remaining_need > 0:
            remaining = np.setdiff1d(np.arange(len(cand)), take)
            take.extend(remaining[:remaining_need])

        if not take: return pts
        return np.vstack([pts, cand[take]]).astype(np.float32)

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
        h, w = self.frame_size[:2]
        c = np.array([w / 2, h / 2])

        flow_comp = flow - (s - 1) * (good_old - c)
        div = np.log(s) / self.dt
        return flow_comp, s, div


    def _build_triangles(self, points, max_edge=200.0, min_area=20.0):
        points = np.asarray(points, dtype=np.float32)
        n = len(points)
        empty = (np.zeros((0, 3), int), points, np.zeros((0, 2), int), np.zeros((n, n), bool))
        if n < 3: return empty
        tri = Delaunay(points).simplices
       
        p = points[tri]
        edge_len = np.linalg.norm(p - np.roll(p, 1, axis=1), axis=2)
        area = 0.5 * np.abs(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]))
        triangles = tri[(edge_len.max(axis=1) < max_edge) & (area > min_area)]

        e = np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
        links = np.unique(np.sort(e, axis=1), axis=0)

        adjacency = np.zeros((n, n), dtype=bool)
        adjacency[links[:, 0], links[:, 1]] = True
        adjacency[links[:, 1], links[:, 0]] = True

        return triangles, points, links, adjacency


    def filter_by_neighbors(self, good_old, good_new, links, thresh=2.0, 
                            min_links=1, neighbor_min_links=3):
        """
        Filter points by flow consistency with neighbors.
        A point survives if:
        - It has enough direct neighbors AND flow agrees, OR
        - At least one of its neighbors is well-connected (structurally supported).
        """
        if len(links) == 0:
            return good_old[:0], good_new[:0]

        flow = good_new - good_old
        n = len(flow)
        diff_sum = np.zeros(n)
        count = np.zeros(n)
        d = np.linalg.norm(flow[links[:, 0]] - flow[links[:, 1]], axis=1)
        np.add.at(diff_sum, links[:, 0], d)
        np.add.at(diff_sum, links[:, 1], d)
        np.add.at(count, links[:, 0], 1)
        np.add.at(count, links[:, 1], 1)

        mean_diff = diff_sum / np.maximum(count, 1)

        # Primary rule: direct connectivity + flow agreement
        keep = (count > min_links) & (mean_diff < thresh)

        # Structural support rule: survives if a well-connected neighbor exists
        # For each point, find the MAX degree among its direct neighbors
        neighbor_max_degree = np.zeros(n, dtype=int)
        np.maximum.at(neighbor_max_degree, links[:, 0], count[links[:, 1]].astype(int))
        np.maximum.at(neighbor_max_degree, links[:, 1], count[links[:, 0]].astype(int))

        supported = (neighbor_max_degree >= neighbor_min_links) & (mean_diff < thresh)

        keep = keep | supported

        return good_old[keep], good_new[keep]


    def trinagulate_altitude(self, news, flow, olds, triangles, links) -> list:
        """
        Estimate altitude per feature using stereo disparity.
        
        Parameters
        ----------
        news : list of two ndarrays, each shape (N, 3)
            Current-frame feature positions with altitude placeholder as third column.
            news[0] : (N, 3) - left camera (x_px, y_px, z_placeholder)
            news[1] : (N, 3) - right camera (x_px, y_px, z_placeholder)
            Index i in news[0] corresponds to the SAME physical feature as index i in news[1].
            The third column currently holds the drone's barometric altitude for all points.
            This function should REPLACE that column with per-point altitude estimated 
            from stereo disparity: z = f * B / disparity.
        
        flow : ndarray, shape (N, 2)
            Per-point optical flow in the LEFT camera, pixels per frame.
            flow[i] = (dx, dy) for feature i, computed as points_new - points_old.
            Same length and index correspondence as points[0].
            After this function, flow is used by estimate_velocities to compute
            drone translational velocity: v_ground = flow * z / (f * dt).
            More accurate z per point → more accurate velocity per point.
        
        olds : list of two ndarrays, each shape (N, 3)
            Previous-frame feature positions, same format as points.
            olds[0] : (N, 3) - left camera positions in previous frame
            olds[1] : (N, 3) - right camera positions in previous frame
            Same index correspondence as points.
            Can be used to:
            - Verify stereo consistency over time (features should maintain 
            similar disparity if altitude doesn't change rapidly).
            - Reject features whose disparity changes implausibly between frames
            (likely bad tracks).
            - Smooth altitude estimates temporally per feature.
        
        triangles : ndarray, shape (T, 3), dtype int
            Delaunay triangulation of features. Each row is 3 indices into points
            (same indices work for left and right because of 1:1 correspondence).
            Can be used to:
            - Enforce altitude consistency within each triangle (triangle vertices
            lying on ground should have similar altitudes).
            - Reject triangles with wildly inconsistent per-vertex altitudes
            (indicates one vertex is on an obstacle, or a bad stereo match).
            - Compute per-triangle altitude as median of its 3 vertex altitudes
            for smoothing.
            - Detect scene structure: triangles with varying altitudes span 
            non-planar surfaces (obstacles, terrain variation).
        
        links : ndarray, shape (L, 2), dtype int
            Unique undirected edges of the triangulation. Each row [a, b] means
            points a and b are triangulation neighbors.
            Can be used to:
            - Smooth altitude between neighbors (Laplacian smoothing on altitude field).
            - Detect altitude discontinuities (large |z[a] - z[b]| → scene edge).
            - Reject features whose altitude disagrees heavily with all neighbors
            (likely bad stereo match).
            - Build a graph where edge weights = altitude difference, for 
            downstream segmentation of scene by depth.
        
        Returns
        -------
        list of two ndarrays, each shape (N, 3)
            points list with third column replaced by estimated altitude per feature.
            Index correspondence preserved. Same shape as input points.
        
        Notes
        -----
        Stereo disparity formula:
            disparity_i = points[0][i, 0] - points[1][i, 0]
            altitude_i  = focal_length_pixels * baseline_meters / disparity_i
        
        For narrow-baseline rectified stereo, use horizontal disparity (x difference).
        For unrectified stereo, use cv2.triangulatePoints with calibrated projection
        matrices for full 3D position recovery.
        
        Current implementation returns input unchanged — altitude logic to be added.
        """
        for k in range(2):
            olds[k][:, 2] = self._current_height()
            news[k][:, 2] = self._current_height()
        return olds, news


    def estimate_velocities(self, flow, points):
        if len(points) < 2:
            self.camera_velocity = self._fallback_velocity()
            self.camera_angle = self.drone_ang[:2]
            return None

        ang_vel, att = self.avg_ang_vel, self.avg_att
        h, w = self.frame_size[:2]
        c = np.array([w / 2, h / 2])

        n = len(points)
        z = points[:, 2]

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


    def estimate_ang_from_vel(self, v, p):
        c_img = np.array(self.frame_size[:2]) / 2
        u_pix = p[:, 0] - c_img[0]
        v_pix = p[:, 1] - c_img[1]

        A = np.column_stack([u_pix, v_pix, np.ones(len(u_pix))])
        v_mag = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)
        (a, b, c), *_ = np.linalg.lstsq(A, v_mag, rcond=None)

        pitch = -self.foc_l * a / (2 * c)
        roll = -self.foc_l * b / (2 * c)
        return pitch, roll



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
        self.expected_vel_err = np.mean(residuals)
        threshold = 1.5
        if residuals.sum() > 0:
            threshold = np.min(residuals) + 0.65 * (np.max(residuals) - np.min(residuals))
        return residuals < threshold
    
    # ---------------- utils ----------------

    @staticmethod
    def add_noise(frames, sigma=5):
        out = []
        for frame in frames:
            # Simple blur with small kernel
            # Add noise
            noisy = frame.astype(np.float32) + np.random.normal(0, sigma, frame.shape)
            out.append(np.clip(noisy, 0, 255).astype(np.uint8))
        return out
    
    
    @staticmethod
    def sharpen_kernel(image, strength=0.5):
        laplacian = cv2.Laplacian(image, cv2.CV_32F, ksize=3)
        sharpened = image.astype(np.float32) - strength * laplacian
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def set_dt(self, dt):
        self.dt = dt

    def _fallback_velocity(self):
        print("fallback velocity used")
        return np.array([self.drone_vel[0], self.drone_vel[1], self.drone_ang_vel[2]])


class VelocityKalmanFilter:
    def __init__(self, process_var=0.6, measurement_var=3, yaw_process_var=5.0, yaw_measurement_var=0.1):
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