import cv2
import numpy as np
from collections import deque
from scipy.spatial import Delaunay


class EnumFrame:
    def __init__(self):
        self.frames = []
        self.idx = 0


class VisulaAcEst:
    def __init__(self, res, fov):
        self.foc_l = (res[0] / 2) / np.tan(np.deg2rad(fov) / 2)
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
        processed, flows, points, olds, news = [], [], [], [], []

        for cam in range(min(len(prev), len(curr))):
            fr, flow, p, old, new = self.lucas_kanade_flow(prev[cam], curr[cam], cam)
            if old is None:
                self.camera_velocity = self.drone_vel
                return
            processed.append(fr)
            flows.append(flow)
            points.append(p)
            olds.append(old)
            news.append(new)

        h = self._current_height()
        for k in range(len(points)):
            points[k][:, 2] = h
            olds[k][:, 2] = h
            news[k][:, 2] = h

        points = self.trinagulate_altitude(points, flows, olds)

        for k in range(len(flows)):
            keep = self.point_prediction_filtering(olds[k], news[k])
            flows[k] = flows[k][keep]
            points[k] = points[k][keep]

        self.current_frame.frames = processed
        self.estimate_velocities(flows, points)

    # ---------------- tracking ----------------

    def lucas_kanade_flow(self, frame1, frame2, cam):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        annotated_frame = frame2.copy()
        lost = (annotated_frame, None, None, None, None)
        empty = np.empty((0, 2), np.float32)

        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        pts = self._replenish(gray1, self.tracked_points.get(cam, empty), cam)
        if len(pts) < 3:
            self.tracked_points[cam] = empty
            return lost

        old = pts.reshape(-1, 1, 2).astype(np.float32)
        new, st_f, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, old, None, **lk)
        if new is None:
            self.tracked_points[cam] = empty
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

        if len(good_new) < 3:
            self.tracked_points[cam] = empty
            return lost

        triangles, corners, links, _ = self._build_triangles(good_new)
        for a, b in links:
            cv2.line(annotated_frame, tuple(corners[a].astype(int)), tuple(corners[b].astype(int)),
                     (255, 255, 0), 1)
        for x, y in corners.astype(int):
            cv2.circle(annotated_frame, (x, y), 2, (0, 0, 255), -1)

        self.tracked_points[cam] = good_new.copy()

        flow, scale, div = self.compensate_vertical(good_old, good_new, triangles)
        flow = self.smooth_flow_by_neighbors(good_new, good_new - good_old, links)
        self.prev_med_flow[cam] = np.median(flow, axis=0)

        z = self._current_height()
        good_old = np.hstack((good_old, np.full((len(good_old), 1), z)))
        good_new = np.hstack((good_new, np.full((len(good_new), 1), z)))
        avg_pos = (good_old + good_new) / 2
        return annotated_frame, flow, avg_pos, good_old, good_new


    def _replenish(self, gray, pts, cam, max_points=100, grid=(4, 4), per_empty=3):
        h, w = gray.shape
        gy, gx = grid
        ch, cw = h // gy, w // gx

        def cell_of(p):
            return (np.clip((p[:, 1] // ch).astype(int), 0, gy - 1) * gx +
                    np.clip((p[:, 0] // cw).astype(int), 0, gx - 1))

        mask = np.full((h, w), 255, np.uint8)
        for x, y in pts.astype(int):
            cv2.circle(mask, (x, y), 7, 0, -1)

        cand = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01,
                                       minDistance=7, blockSize=7, mask=mask)
        if cand is None:
            return pts
        cand = cand.reshape(-1, 2)

        occupied = np.zeros(gy * gx, bool)
        if len(pts):
            occupied[cell_of(pts)] = True

        cand_cells = cell_of(cand)
        take = []
        for cell in np.flatnonzero(~occupied):
            take.extend(np.flatnonzero(cand_cells == cell)[:per_empty])

        need = max_points - len(pts) - len(take)
        if need > 0:
            take.extend(np.setdiff1d(np.arange(len(cand)), take)[:need])

        if not take:
            return pts
        return np.vstack([pts, cand[take]]).astype(np.float32)

    # ---------------- mesh ----------------

    def compensate_vertical(self, good_old, good_new, triangles, min_area=5.0):
        flow = good_new - good_old
        if len(triangles) == 0:
            return flow, 1.0, 0.0

        def area(p):
            return 0.5 * np.abs(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]))

        a_old = area(good_old[triangles])
        a_new = area(good_new[triangles])
        ok = a_old > min_area
        if not ok.any():
            return flow, 1.0, 0.0

        s = np.sqrt(np.median(a_new[ok] / a_old[ok]))
        h, w = self.frame_size[:2]
        h, w = self.frame_size[:2]
        c = np.array([w / 2, h / 2])

        flow_comp = flow - (s - 1) * (good_old - c)
        div = np.log(s) / self.dt
        return flow_comp, s, div


    def _build_triangles(self, points, max_edge=80.0, min_area=2.0):
        points = np.asarray(points, dtype=np.float32)
        n = len(points)
        empty = (np.zeros((0, 3), int), points, np.zeros((0, 2), int), np.zeros((n, n), bool))
        if n < 3:
            return empty

        try:
            tri = Delaunay(points).simplices
        except Exception:
            return empty

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

    def filter_by_neighbors(self, good_old, good_new, links, thresh=2.0, min_links=1):
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
        keep = (count > min_links) & (mean_diff < thresh)
        return good_old[keep], good_new[keep]


    def smooth_flow_by_neighbors(self, points, flow, links, self_weight=0.7, iterations=3, sigma=None):
        flow = flow.astype(np.float64).copy()
        n = len(flow)
        if len(links) == 0:
            return flow

        i, j = links[:, 0], links[:, 1]
        if sigma is not None:
            d2 = np.sum((points[i] - points[j]) ** 2, axis=1)
            w = np.exp(-d2 / (2 * sigma ** 2))
        else:
            w = np.ones(len(links))

        for _ in range(iterations):
            acc = flow * self_weight
            wsum = np.full(n, self_weight, dtype=np.float64)
            np.add.at(acc, i, flow[j] * w[:, None])
            np.add.at(acc, j, flow[i] * w[:, None])
            np.add.at(wsum, i, w)
            np.add.at(wsum, j, w)
            flow = acc / wsum[:, None]

        return flow


    def trinagulate_altitude(self, points, flow, old) -> list:
        return list(points)

    def estimate_velocities(self, flow, points):
        if any(len(p) < 2 for p in points):
            self.camera_velocity = self.drone_vel
            self.camera_angle = self.drone_ang[:2]
            return None

        ang_vel, att = self.avg_ang_vel, self.avg_att
        h, w = self.frame_size[:2]
        c = np.array([w / 2, h / 2])
        v_out = []

        for cam, (f, p) in enumerate(zip(flow, points)):
            n = len(p)
            z = p[:, 2]

            v = f * (z / (self.foc_l * self.dt))[:, None]
            w = ang_vel[[1, 0]] * np.array([ 1,  1])   # swapped
            v_rot = z[:, None] / np.cos(att[[1, 0]]) ** 2 * w
            B = v - v_rot

            r = (z[:, None] / self.foc_l) * (p[:, :2] - c) + self.camera_saperation / 2
            A = np.vstack([np.column_stack([np.ones(n), np.zeros(n), -r[:, 1]]),
                           np.column_stack([np.zeros(n), np.ones(n), r[:, 0]])])
            x, *_ = np.linalg.lstsq(A, np.concatenate([B[:, 0], B[:, 1]]), rcond=None)
            x = -x

            kf = self.vkfs.setdefault(cam, VelocityKalmanFilter())
            if not kf.started:
                kf.state = x.copy()
                kf.started = True
            kf.predict(kf.state)
            kf.update(x)
            v_out.append(kf.get())

        self.camera_velocity = np.mean(v_out, axis=0)


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
            threshold = np.min(residuals) + 0.35 * (np.max(residuals) - np.min(residuals))
        return residuals < threshold

    # ---------------- utils ----------------

    @staticmethod
    def add_noise(frames, sigma=5):
        out = []
        for frame in frames:
            noisy = frame.astype(np.float32) + np.random.normal(0, sigma, frame.shape)
            out.append(np.clip(noisy, 0, 255).astype(np.uint8))
        return out

    def set_dt(self, dt):
        self.dt = dt


class VelocityKalmanFilter:
    def __init__(self, process_var=1, measurement_var=6):
        self.state = np.zeros(3)
        self.P = np.eye(3)
        self.Q = np.eye(3) * process_var
        self.R = np.eye(3) * measurement_var
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