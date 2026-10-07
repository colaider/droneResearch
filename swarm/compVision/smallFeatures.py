"""GFTT points, temporal Lucas--Kanade tracking, and the existing point mesh.

This stage never computes stereo depth or converts pixels to velocity. Large
regions only guide replenishment; point coordinates remain full-resolution.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import Delaunay, QhullError


@dataclass
class FeatureTracks:
    old_points: np.ndarray
    new_points: np.ndarray
    triangles: np.ndarray
    links: np.ndarray

    @classmethod
    def empty(cls):
        return cls(np.empty((0, 2), np.float32), np.empty((0, 2), np.float32),
                   np.empty((0, 3), int), np.empty((0, 2), int))


class SmallFeatureTracker:
    def __init__(self, max_points=300, region_fraction=0.35):
        self.max_points = int(max_points)
        self.region_fraction = float(region_fraction)
        if self.max_points < 0 or not 0 <= self.region_fraction <= 1:
            raise ValueError("Invalid feature budget")
        self.tracked_points = {}
        # Retain the current detector and tracker settings.
        self.quality_level = 0.005
        self.min_distance = 100
        self.block_size = 11
        self.exclusion_radius = 7
        self.lk = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def track(self, previous_gray, current_gray, regions=(), cam=0):
        empty = FeatureTracks.empty()
        pts = self.replenish(previous_gray, self.tracked_points.get(cam, empty.old_points),
                             cam, self.max_points, regions=regions)

        def lost():
            self.tracked_points[cam] = empty.new_points.copy()
            return empty

        if len(pts) < 3:
            return lost()
        old = pts.reshape(-1, 1, 2).astype(np.float32)
        new, st_f, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, old, None, **self.lk)
        if new is None or st_f is None:
            return lost()
        # Do not pass failed/non-finite forward outputs into a second LK call.
        forward = (st_f.ravel() == 1) & np.isfinite(new.reshape(-1, 2)).all(axis=1)
        if not forward.any():
            return lost()
        old, new = old[forward], new[forward]
        back, st_b, _ = cv2.calcOpticalFlowPyrLK(current_gray, previous_gray, new, None, **self.lk)
        if back is None or st_b is None:
            return lost()
        old, new, back = old.reshape(-1, 2), new.reshape(-1, 2), back.reshape(-1, 2)
        h, w = current_gray.shape
        good = ((st_b.ravel() == 1) & np.isfinite(back).all(axis=1)
                & (np.abs(old - back).max(axis=1) < 1.0)
                & (new[:, 0] >= 0) & (new[:, 0] < w)
                & (new[:, 1] >= 0) & (new[:, 1] < h))
        old, new = old[good], new[good]
        if len(old) >= 6:
            _, inliers = cv2.estimateAffinePartial2D(
                old, new, method=cv2.RANSAC, ransacReprojThreshold=2.0,
                maxIters=2000, confidence=0.99)
            if inliers is not None:
                keep = inliers.ravel().astype(bool)
                old, new = old[keep], new[keep]
        for _ in range(2):
            if len(new) < 3:
                break
            _, _, links, _ = self.build_triangles(new)
            old, new = self.filter_by_neighbors(old, new, links)
        if len(new) < 3:
            return lost()
        triangles, _, links, _ = self.build_triangles(new)
        self.tracked_points[cam] = new.copy()
        return FeatureTracks(old, new, triangles, links)

    def replenish(self, gray, pts, cam=0, max_points=300, regions=()):
        """Existing GFTT/Canny/quadrant method, with optional structural priority.

        Up to region_fraction of new slots get ROI candidates first. Global
        candidates still supply the rest. This is not the optional statistical
        budget experiment: the original fine detector parameters are retained.
        """
        pts = np.asarray(pts, np.float32).reshape(-1, 2)
        h, w = gray.shape
        inside = (np.isfinite(pts).all(axis=1) & (pts[:, 0] >= 0) & (pts[:, 0] < w)
                  & (pts[:, 1] >= 0) & (pts[:, 1] < h))
        pts = pts[inside]
        need = max(0, int(max_points) - len(pts))
        if not need:
            return pts.copy()
        mask = np.full((h, w), 255, np.uint8)
        for x, y in pts.astype(int):
            cv2.circle(mask, (x, y), self.exclusion_radius, 0, -1)
        edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((5, 5), np.uint8))

        def detect(search_mask, count):
            if count <= 0 or not search_mask.any():
                return np.empty((0, 2), np.float32)
            cand = cv2.goodFeaturesToTrack(
                gray, maxCorners=count, qualityLevel=self.quality_level,
                minDistance=self.min_distance, blockSize=self.block_size, mask=search_mask)
            if cand is None:
                return np.empty((0, 2), np.float32)
            cand = cand.reshape(-1, 2)
            xy = np.rint(cand).astype(int)
            return cand[edges[np.clip(xy[:, 1], 0, h - 1), np.clip(xy[:, 0], 0, w - 1)] > 0]

        preferred = np.empty((0, 2), np.float32)
        if regions and self.region_fraction > 0:
            roi_mask = np.zeros_like(mask)
            for x0, y0, x1, y1 in regions:
                x0, x1 = np.clip([int(x0), int(x1)], 0, w)
                y0, y1 = np.clip([int(y0), int(y1)], 0, h)
                if x1 > x0 and y1 > y0:
                    roi_mask[y0:y1, x0:x1] = 255
            preferred = detect(cv2.bitwise_and(mask, roi_mask), int(np.ceil(need * self.region_fraction)))
        # Enforce the same new-candidate spacing between ROI and global points.
        global_mask = mask.copy()
        for point in preferred:
            cv2.circle(global_mask, tuple(np.rint(point).astype(int)), self.min_distance, 0, -1)
        cand = detect(global_mask, 500)
        chosen = list(preferred[:need])
        if len(cand):
            def quad(p):
                return (p[:, 0] >= w / 2).astype(int) + 2 * (p[:, 1] >= h / 2).astype(int)
            seeded = np.vstack((pts, preferred))
            existing = np.bincount(quad(seeded), minlength=4)
            candidate_quads = quad(cand)
            quota = np.maximum(max_points // 4 - existing, 0)
            take = []
            for q in range(4):
                room = need - len(chosen) - len(take)
                take.extend(np.flatnonzero(candidate_quads == q)[:min(int(quota[q]), room)])
            room = need - len(chosen) - len(take)
            if room > 0:
                take.extend(np.setdiff1d(np.arange(len(cand)), take)[:room])
            chosen.extend(cand[take])
        return np.vstack((pts, np.asarray(chosen, np.float32).reshape(-1, 2))).astype(np.float32)

    @staticmethod
    def build_triangles(points, max_edge=200.0, min_area=20.0):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        n = len(points)
        empty = (np.zeros((0, 3), int), points, np.zeros((0, 2), int), np.zeros((n, n), bool))
        if n < 3 or not np.isfinite(points).all():
            return empty
        try:
            tri = Delaunay(points).simplices
        except QhullError:
            return empty
        p = points[tri]
        edge_len = np.linalg.norm(p - np.roll(p, 1, axis=1), axis=2)
        a, b = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
        area = 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        triangles = tri[(edge_len.max(axis=1) < max_edge) & (area > min_area)]
        e = np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
        links = np.unique(np.sort(e, axis=1), axis=0)
        adjacency = np.zeros((n, n), bool)
        adjacency[links[:, 0], links[:, 1]] = True
        adjacency[links[:, 1], links[:, 0]] = True
        return triangles, points, links, adjacency

    @staticmethod
    def filter_by_neighbors(good_old, good_new, links, thresh=2.0, min_links=1, neighbor_min_links=3):
        if len(links) == 0:
            return good_old[:0], good_new[:0]
        flow = good_new - good_old
        n = len(flow)
        diff_sum, count = np.zeros(n), np.zeros(n)
        difference = np.linalg.norm(flow[links[:, 0]] - flow[links[:, 1]], axis=1)
        for column in (0, 1):
            np.add.at(diff_sum, links[:, column], difference)
            np.add.at(count, links[:, column], 1)
        mean_diff = diff_sum / np.maximum(count, 1)
        keep = (count > min_links) & (mean_diff < thresh)
        neighbor_max_degree = np.zeros(n, int)
        np.maximum.at(neighbor_max_degree, links[:, 0], count[links[:, 1]].astype(int))
        np.maximum.at(neighbor_max_degree, links[:, 1], count[links[:, 0]].astype(int))
        supported = (neighbor_max_degree >= neighbor_min_links) & (mean_diff < thresh)
        return good_old[keep | supported], good_new[keep | supported]
