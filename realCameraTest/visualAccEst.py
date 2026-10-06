"""Standalone copy of the visual velocity estimator for REAL two-camera (USB) testing.

Differences from swarm/compVision/visualAccEst.py:
  * No Genesis / swarm.config dependency -- focal length, baseline and the stereo
    disparity direction are passed in directly (from camera_params.txt).
  * No IMU: angular velocity and attitude are held at ZERO (no rotation compensation),
    so the estimator does not break without a flight controller feeding it.
  * The command-velocity `point_prediction_filtering` step is removed (there is no
    commanded velocity on a hand-held rig), along with the push_sensors / sensor-window
    machinery and the unused crop/rotate helpers.
"""
import cv2
import numpy as np
from collections import deque
from scipy.spatial import Delaunay


class EnumFrame:
    def __init__(self):
        self.frames = []
        self.idx = 0


class VisulaAcEst:
    def __init__(self, focal_px, baseline, res=None, dt=1.0 / 30.0,
                 disparity_dir=(1.0, 0.0), default_depth=1.0,
                 epipolar_tol=2.0, min_disparity=0.3,
                 sharpen_amount=0.6, rs_readout=0.5):
        self.foc_l = float(focal_px)                 # left-camera focal length (px), from the TXT
        self.camera_saperation = float(baseline)     # stereo baseline (m), from the TXT
        self.dt = float(dt)
        self.disp_dir = np.asarray(disparity_dir, float)   # baseline axis in the image (horizontal pair: x)
        self.default_depth = float(default_depth)    # assumed depth (m) when stereo is unavailable
        # stereo acceptance thresholds (loosen epipolar_tol for unrectified USB cameras)
        self.epipolar_tol = float(epipolar_tol)      # px, max offset perpendicular to the baseline
        self.min_disparity = float(min_disparity)    # px, reject near-zero disparity
        # frame preprocessing
        self.sharpen_amount = float(sharpen_amount)  # unsharp-mask strength; 0 disables
        self.rs_readout = float(rs_readout)          # rolling-shutter readout fraction (0..1); 0 disables

        self.current_frame = EnumFrame()
        self.buffer = deque(maxlen=100)

        # No IMU on a USB rig: angular velocity / attitude are zero -> no rotation compensation.
        self.avg_ang_vel = np.zeros(3)
        self.avg_att = np.zeros(3)

        self.tracked_points = {}
        self.prev_med_flow = {}
        self.vkfs = {}

        self.camera_velocity = np.zeros(3)           # [vx, vy, yaw_rate]
        self.depth_median = np.nan
        self.depth_valid_frac = 0.0

    # ---------------- frames ----------------

    def update_frame(self, frame, idx):
        new_frame_data = EnumFrame()
        left, right = frame[0], frame[1]
        # The two USB cameras may deliver different resolutions; optical flow between them
        # requires equal sizes, so resample the right (non-dominant) eye onto the left's grid.

        if right.shape[:2] != left.shape[:2]:
            right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
        # preprocess: rolling-shutter deskew (uses last frame's motion) then sharpen
        left = self._sharpen(left)
        right = self._sharpen(right)
        new_frame_data.frames = [left, right]
        new_frame_data.idx = idx
        if not hasattr(self, 'frame_size'):
            self.frame_size = np.shape(new_frame_data.frames[0])
        self.buffer.append(new_frame_data)
        self.current_frame = new_frame_data

    def processing(self, frame, idx):
        self.update_frame(frame, idx)
        self.aply_flow()
        return self.current_frame

    def _sharpen(self, frame):
        """Unsharp mask: frame + amount*(frame - blur). amount<=0 is a no-op (returns a copy)."""
        if self.sharpen_amount <= 0:
            return frame.copy()
        blur = cv2.GaussianBlur(frame, (0, 0), 1.0)
        return cv2.addWeighted(frame, 0.5 + self.sharpen_amount, blur, -self.sharpen_amount, 0)

    def _deskew_rolling_shutter(self, frame):
        """Undo rolling-shutter skew with a per-row shear from the last frame's median flow.

        A rolling shutter reads rows sequentially, so while the camera moves each row is
        displaced a bit more than the one above it. Using last frame's median optical flow
        `mf` (px/frame) as the motion estimate, row y is displaced by (y/H)*rs_readout*mf;
        we apply the inverse affine to straighten it. Approximate but cheap.
        """
        mf = self.prev_med_flow.get(0)
        if mf is None or self.rs_readout <= 0:
            return frame
        h, w = frame.shape[:2]
        kx = -self.rs_readout * float(mf[0]) / h     # x shear per row (from horizontal motion)
        ky = -self.rs_readout * float(mf[1]) / h     # y scale per row (from vertical motion)
        M = np.array([[1.0, kx, 0.0],
                      [0.0, 1.0 + ky, 0.0]], dtype=np.float32)
        return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _current_depth(self):
        return self.default_depth

    # ---------------- pipeline ----------------

    def aply_flow(self):
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

        news = self.trinagulate_altitude(news, flow, olds, tris, links)
        # prediction filter removed: use every tracked point directly.
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

        lk = dict(winSize=(51, 51), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

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

        z = self._current_depth()
        olds = [np.hstack((good_old, np.full((len(good_old), 1), z))),
                np.hstack((old_r, np.full((len(old_r), 1), z)))]
        news = [np.hstack((good_new, np.full((len(good_new), 1), z))),
                np.hstack((new_r, np.full((len(new_r), 1), z)))]

        return [annotated_frame, annotated_r], flow, olds, news, triangles, links

    def _replenish(self, gray, pts, cam, max_points=1000):
        h, w = gray.shape
        mask = np.full((h, w), 255, np.uint8)
        for x, y in pts.astype(int):
            cv2.circle(mask, (x, y), 7, 0, -1)

        cand = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.005, minDistance=15, blockSize=11, mask=mask)
        if cand is None: return pts
        cand = cand.reshape(-1, 2)
        edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((5, 5), np.uint8))

        xs = np.clip(cand[:, 0].astype(int), 0, w - 1)
        ys = np.clip(cand[:, 1].astype(int), 0, h - 1)
        keep = edges[ys, xs] > 0
        cand = cand[keep]
        if len(cand) == 0: return pts

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
        c = np.array([w / 2, h / 2])

        flow_comp = flow - (s - 1) * (good_old - c)
        div = np.log(s) / self.dt
        return flow_comp, s, div

    def _build_triangles(self, points, max_edge=2000.0, min_area=20.0):
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

    def filter_by_neighbors(self, good_old, good_new, links, thresh=5.0, min_links=1, neighbor_min_links=3):
        if len(links) == 0: return good_old[:0], good_new[:0]

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
        keep = (count > min_links) & (mean_diff < thresh)

        neighbor_max_degree = np.zeros(n, dtype=int)
        np.maximum.at(neighbor_max_degree, links[:, 0], count[links[:, 1]].astype(int))
        np.maximum.at(neighbor_max_degree, links[:, 1], count[links[:, 0]].astype(int))
        supported = (neighbor_max_degree >= neighbor_min_links) & (mean_diff < thresh)
        keep = keep | supported

        return good_old[keep], good_new[keep]

    def trinagulate_altitude(self, news, flow, olds, triangles, links):
        """Replace the depth column of news/olds with stereo depth (f * baseline / disparity)."""
        L, R = olds[0][:, :2], olds[1][:, :2]
        delta = R - L
        disp_dir = self.disp_dir
        perp = np.array([-disp_dir[1], disp_dir[0]])
        d = np.abs(delta @ disp_dir)               # disparity magnitude along the baseline (sign-agnostic)
        off = np.abs(delta @ perp)                 # epipolar error perpendicular to the baseline
        keep = (off < self.epipolar_tol) & (d > self.min_disparity)

        depth = np.full(len(d), self._current_depth())
        if keep.any():
            depth[keep] = self.foc_l * self.camera_saperation / d[keep]
            depth[~keep] = np.median(depth[keep])

        self.depth_valid_frac = keep.mean() if len(keep) else 0.0
        self.depth_median = np.median(depth[keep]) if keep.any() else np.nan

        for k in range(2):
            olds[k][:, 2] = depth
            news[k][:, 2] = depth
        return news

    def estimate_velocities(self, flow, points):
        if len(points) < 2:
            self.camera_velocity = self._fallback_velocity()
            return None

        ang_vel, att = self.avg_ang_vel, self.avg_att   # both zero on a USB rig
        h, w = self.frame_size[:2]
        c = np.array([w / 2, h / 2])

        n = len(points)
        z = points[:, 2]

        v = flow * (z / (self.foc_l * self.dt))[:, None]
        w_ang = ang_vel[[1, 0]] * np.array([1, 1])       # zero -> no rotation term
        v_rot = z[:, None] / np.cos(att[[1, 0]]) ** 2 * w_ang

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

    # ---------------- utils ----------------

    def set_dt(self, dt):
        self.dt = dt

    def _fallback_velocity(self):
        print("fallback velocity used")
        return np.zeros(3)


class VelocityKalmanFilter:
    def __init__(self, process_var=0.6, measurement_var=3.5, yaw_process_var=5.0, yaw_measurement_var=0.1):
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
