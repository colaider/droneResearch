"""Small-feature selection and image-space neighbourhood helpers.

replenish detects GFTT corners, selects those near Canny edges and balances
new points across image quadrants. build_triangles builds pixel-space links;
these triangles do not establish physical surfaces or depth.
filter_by_neighbors is a separate track-quality filter using supplied motion.
Temporal optical flow, stereo matching and metric reconstruction live elsewhere.
"""
import cv2
import numpy as np
from scipy.spatial import Delaunay


def replenish(gray, pts, cam, max_points=300):
    h, w = gray.shape
    mask = np.full((h, w), 255, np.uint8)
    for x, y in pts.astype(int):
        cv2.circle(mask, (x, y), 7, 0, -1)

    cand = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.005, minDistance=100, blockSize=11, mask=mask)
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


def build_triangles(points, max_edge=200.0, min_area=20.0):
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


def filter_by_neighbors(good_old, good_new, links, thresh=2.0, min_links=1, neighbor_min_links=3):
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

    # Structural support rule: survives if a well-connected neighbor exists
    # For each point, find the MAX degree among its direct neighbors
    neighbor_max_degree = np.zeros(n, dtype=int)
    np.maximum.at(neighbor_max_degree, links[:, 0], count[links[:, 1]].astype(int))
    np.maximum.at(neighbor_max_degree, links[:, 1], count[links[:, 0]].astype(int))
    supported = (neighbor_max_degree >= neighbor_min_links) & (mean_diff < thresh)
    keep = keep | supported

    return good_old[keep], good_new[keep]
