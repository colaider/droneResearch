import cv2
import numpy as np
from collections import deque
import time
import swarm.config.cameraConfig as camConfig
from swarm.compVision.smallFeatures import (
    replenish, build_triangles, filter_by_neighbors, LayerTracks, LAYER_COLORS, unique_point_mask,
)
from swarm.compVision.frame import Frame, FrameProcessor, compose_layer_display

# Preserve the original public frame name.
EnumFrame = Frame


class VisulaAcEst:
    def __init__(self, res, fov):
        self.fov = fov
        self.foc_l = (res[1] / 2) / np.tan(np.deg2rad(fov) / 2)
        self.dt = 0.01
        self.camera_saperation = camConfig.STEREO_CAM.baseline

        self.frame_processor = FrameProcessor(pyramid_levels=3, image_filters=("clahe",), filter_kernel_size=3)
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

        self.tracked_points_by_layer = {0: {}}  # separate native-pixel seeds for each layer
        self.tracked_points = self.tracked_points_by_layer[0]  # legacy camera-0/full-resolution view
        self.layer_tracks = {}  # layer -> LayerTracks in full-resolution pixels
        self._layer_frame_idx = {}
        self.max_feature_points = 1500  # total, shared among all pyramid layers
        self.feature_min_distance = 30.0  # base spacing in full-resolution pixels
        self.cross_layer_min_distance = 5.0  # full-resolution pixels, velocity fit only
        self.all_points = [np.empty((0, 2), np.float32) for _ in range(self.frame_processor.pyramid_levels)]
        self.point_colors = [np.empty((0, 3), np.uint8) for _ in self.all_points]
        self.point_layers = np.empty(0, int)  # flattened layer IDs aligned with velocity_point_mask
        self.velocity_point_mask = np.empty(0, bool)
        self.layer_views = []  # one annotated left-image view per pyramid layer
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
        self.layer_views = [
            compose_layer_display(gray, self.all_points[layer] / (2 ** layer),
                                  self.layer_tracks[layer].links, layer,
                                  LAYER_COLORS[layer % len(LAYER_COLORS)])
            for layer, gray in enumerate(self.current_frame.pyramids[0])
        ]
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
        num_layers = len(self.current_frame.pyramids[0]) if self.current_frame.pyramids else 0
        self.layer_tracks = {layer: LayerTracks.empty(layer, self.current_frame.idx)
                             for layer in range(num_layers)}
        # Stable layer indices, including empty arrays for layers without valid tracks.
        # Every Nx2 point array uses full-resolution left-image coordinates.
        self.all_points = [np.empty((0, 2), np.float32) for _ in range(num_layers)]
        self.point_colors = [np.empty((0, 3), np.uint8) for _ in range(num_layers)]
        self.point_layers = np.empty(0, int)
        self.velocity_point_mask = np.empty(0, bool)
        if len(self.buffer) < 2:
            return
        prev_data, curr_data = self.buffer[-2], self.buffer[-1]
        prev, curr = prev_data.frames, curr_data.frames
        if min(len(prev), len(curr)) < 2:
            self.camera_velocity = self._fallback_velocity()
            return
        num_layers = len(curr_data.pyramids[0])
        if not num_layers or any(len(pyramid) != num_layers for pyramid in
                                 (prev_data.pyramids[0], prev_data.pyramids[1])):
            raise ValueError('Previous and current image pyramids must have the same layers')

        annotated = [curr[0].copy(), curr[1].copy()]
        flows, previous_points, current_points, layer_ids = [], [], [], []
        budget, remainder = divmod(self.max_feature_points, num_layers)
        for layer, (gray_prev, gray_curr, gray_right) in enumerate(zip(
                prev_data.pyramids[0], curr_data.pyramids[0], prev_data.pyramids[1])):
            annotated, flow, olds, news, _, _ = self.lucas_kanade_flow(
                prev[0], annotated[0], prev[1], annotated[1],
                tracking_gray=(gray_prev, gray_curr, gray_right),
                layer=layer, max_points=budget + (layer < remainder),
            )
            if flow is not None:
                self.all_points[layer] = news[0][:, :2].astype(np.float32, copy=True)
                self.point_colors[layer] = np.tile(
                    np.asarray(LAYER_COLORS[layer % len(LAYER_COLORS)], np.uint8), (len(flow), 1))
                flows.append(flow)
                previous_points.append(olds[0])
                current_points.append(news[0])
                layer_ids.append(np.full(len(flow), layer, int))
        for layer, tracks in self.layer_tracks.items():
            colour = LAYER_COLORS[layer % len(LAYER_COLORS)]
            text = f'L{layer} 1/{2 ** layer} | {len(tracks.new_points)} points'
            for frame in annotated:
                cv2.putText(frame, text, (8, 20 + 20 * layer), cv2.FONT_HERSHEY_SIMPLEX,
                            .5, colour, 1, cv2.LINE_AA)
        cv2.putText(annotated[1], 'Right overlay predicted from flow',
                    (8, 20 + 20 * num_layers), cv2.FONT_HERSHEY_SIMPLEX,
                    .45, (220, 220, 220), 1, cv2.LINE_AA)
        self.current_frame.frames = annotated
        if not flows:
            self.camera_velocity = self._fallback_velocity()
            return
        flow, old, new = np.vstack(flows), np.vstack(previous_points), np.vstack(current_points)
        # Velocity fitting flattens successful tracks; public point arrays stay grouped by layer.
        self.point_layers = np.concatenate(layer_ids)
        keep = (np.isfinite(flow).all(axis=1) & np.isfinite(old).all(axis=1) &
                np.isfinite(new).all(axis=1) & (old[:, 2] > 0) & (new[:, 2] > 0))
        ids = np.flatnonzero(keep)
        if len(ids):
            keep[ids] = self.point_prediction_filtering(old[ids], new[ids])
        # Duplicate observations may remain visible in their own layer, but may
        # not count twice toward the combined motion estimate.
        ids = np.flatnonzero(keep)
        keep[ids] = unique_point_mask(new[ids, :2], self.cross_layer_min_distance)
        self.velocity_point_mask = keep
        self.estimate_velocities(flow[keep], new[keep])

# ---------------- tracking ----------------

    def lucas_kanade_flow(self, frame1, frame2, frame_r, frame_r_curr, tracking_gray=None,
                         *, layer, max_points):
        """Process one iteration of the pyramid loop, returning full-resolution pixels."""
        if tracking_gray is None or len(tracking_gray) < 3:
            raise ValueError('Supply previous-left, current-left and previous-right grayscale layers')
        gray1, gray2, gray_r = tracking_gray[:3]
        # pyrDown keeps even pixel centres: factor 2**layer, also for odd sizes.
        pixel_scale = float(2 ** layer)
        h, w = frame1.shape[:2]
        expected = ((h + int(pixel_scale) - 1) // int(pixel_scale),
                    (w + int(pixel_scale) - 1) // int(pixel_scale))
        if any(gray.shape != expected for gray in (gray1, gray2, gray_r)):
            raise ValueError('Tracking images do not match the pyramid layer')
        annotated_frame, annotated_r = frame2.copy(), frame_r_curr.copy()
        lost = ([annotated_frame, annotated_r], None, None, None, None, None)
        empty = np.empty((0, 2), np.float32)
        cache = self.tracked_points_by_layer.setdefault(layer, {})
        previous_idx = self.buffer[-2].idx if len(self.buffer) >= 2 else self.current_frame.idx - 1
        if layer in self._layer_frame_idx and self._layer_frame_idx[layer] != previous_idx:
            cache.clear()

        def bad_return():
            cache[0] = empty.copy()
            self._layer_frame_idx[layer] = self.current_frame.idx
            self.layer_tracks[layer] = LayerTracks.empty(layer, self.current_frame.idx)
            return lost

        lk = dict(winSize=(21, 21), maxLevel=1,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        lk_stereo = dict(winSize=(100, 100), maxLevel=4,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        pts = self._replenish(gray1, cache.get(0, empty), 0, max_points=max_points, pixel_scale=pixel_scale)
        if len(pts) < 3:
            return bad_return()
        old = pts.reshape(-1, 1, 2).astype(np.float32)
        new, st_f, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, old, None, **lk)
        if new is None or st_f is None:
            return bad_return()
        valid = (st_f.ravel() == 1) & np.isfinite(new.reshape(-1, 2)).all(axis=1)
        if not valid.any():
            return bad_return()
        old, new = old[valid], new[valid]
        back, st_b, _ = cv2.calcOpticalFlowPyrLK(gray2, gray1, new, None, **lk)
        if back is None or st_b is None:
            return bad_return()
        old, new, back = old.reshape(-1, 2), new.reshape(-1, 2), back.reshape(-1, 2)
        good = ((st_b.ravel() == 1) & np.isfinite(back).all(axis=1) &
                (np.abs((old - back) * pixel_scale).max(axis=1) < 1.0) &
                (new[:, 0] >= 0) & (new[:, 0] * pixel_scale < w) &
                (new[:, 1] >= 0) & (new[:, 1] * pixel_scale < h))
        good_old, good_new = old[good] * pixel_scale, new[good] * pixel_scale
        if len(good_old) >= 6:
            _, inliers = cv2.estimateAffinePartial2D(
                good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=2.0,
                maxIters=2000, confidence=0.99)
            if inliers is not None:
                mask = inliers.ravel().astype(bool)
                good_old, good_new = good_old[mask], good_new[mask]
        mesh_args = dict(max_edge=200.0 * pixel_scale, min_area=20.0 * pixel_scale ** 2)
        for _ in range(2):
            if len(good_new) < 3:
                break
            _, _, links, _ = self._build_triangles(good_new, **mesh_args)
            good_old, good_new = self.filter_by_neighbors(good_old, good_new, links)
        if len(good_old) < 3:
            return bad_return()
        p = (good_old / pixel_scale).reshape(-1, 1, 2).astype(np.float32)
        old_r, st_r, _ = cv2.calcOpticalFlowPyrLK(gray1, gray_r, p, None, **lk_stereo)
        if old_r is None or st_r is None:
            return bad_return()
        old_r = old_r.reshape(-1, 2) * pixel_scale
        valid = ((st_r.ravel() == 1) & np.isfinite(old_r).all(axis=1) &
                 (old_r[:, 0] >= 0) & (old_r[:, 0] < w) &
                 (old_r[:, 1] >= 0) & (old_r[:, 1] < h))
        good_old, good_new, old_r = good_old[valid], good_new[valid], old_r[valid]
        if len(good_new) < 3:
            return bad_return()
        triangles, _, links, _ = self._build_triangles(good_new, **mesh_args)
        cache[0] = (good_new / pixel_scale).astype(np.float32)
        self._layer_frame_idx[layer] = self.current_frame.idx
        flow, _, _ = self.compensate_vertical(good_old, good_new, triangles)
        new_r = old_r + flow  # retained legacy overlay, not a stereo depth measurement
        self.layer_tracks[layer] = LayerTracks(
            layer, self.current_frame.idx, good_old.copy(), good_new.copy(), old_r.copy(),
            new_r.copy(), flow.copy(), triangles.copy(), links.copy(),
        )
        z = self._current_height()
        olds = [np.column_stack((points, np.full(len(points), z))) for points in (good_old, old_r)]
        news = [np.column_stack((points, np.full(len(points), z))) for points in (good_new, new_r)]
        annotated_frame, annotated_r = self.drawing(
            annotated_frame, annotated_r, good_new, new_r, links,
            color=LAYER_COLORS[layer % len(LAYER_COLORS)],
        )
        return [annotated_frame, annotated_r], flow, olds, news, triangles, links


    def _replenish(self, gray, pts, cam, max_points=300, *, pixel_scale=1.0):
        """Scale full-resolution feature spacing into native pyramid pixels."""
        min_distance = max(1.0, self.feature_min_distance / pixel_scale)
        return replenish(gray, pts, cam, max_points, min_distance=min_distance)

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
        """Compatibility adapter for smallFeatures.build_triangles."""
        return build_triangles(points, max_edge, min_area)


    def filter_by_neighbors(self, good_old, good_new, links, thresh=2.0, min_links=1, neighbor_min_links=3):
        """Compatibility adapter for smallFeatures.filter_by_neighbors."""
        return filter_by_neighbors(good_old, good_new, links, thresh, min_links, neighbor_min_links)


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
    def drawing(annotated_frame, annotated_r, good_new, new_r, links, color=(255, 0, 0)):
        """Centralized drawing: triangulation and points on left and right frames."""
         
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
        # print("fallback velocity used")
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