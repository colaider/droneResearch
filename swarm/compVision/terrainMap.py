import numpy as np


def quat_to_R(q):
    """(w, x, y, z) quaternion -> 3x3 rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


class TerrainMap:
    """
    World-frame height map built from stereo points over time.

    Each grid cell keeps an inverse-variance weighted mean of the heights that fell into it.
    Stereo depth noise grows as Z^2 / (f * B) * sigma_d, so weight = 1 / sigma^2 ~ 1 / Z^4:
    points seen from close by dominate the ones seen from far away.
    """

    def __init__(self, cam_cfg, size=30.0, cell=0.05, center=(0.0, 0.0), sigma_d=0.2, min_hits=2):
        self.cam_cfg = cam_cfg
        self.cell = cell
        self.n = int(round(size / cell))
        self.origin = np.array(center, dtype=float) - size / 2   # world xy of cell (0, 0) corner
        self.sigma_d = sigma_d
        self.min_hits = min_hits

        self.sum_w = np.zeros((self.n, self.n))
        self.sum_wz = np.zeros((self.n, self.n))
        self.hits = np.zeros((self.n, self.n), dtype=np.int32)
        self.last_frame_idx = None

    def add(self, pts_uvz, drone_pos, drone_quat, frame_idx=None):
        """
        pts_uvz    : (N, 3) oriented left-image pixels + depth along the optical axis
        drone_pos  : (3,) world position of the drone body
        drone_quat : (4,) world orientation of the drone body, (w, x, y, z)
        frame_idx  : camera frame index, the same frame is only added once
        """
        if frame_idx is not None:
            if frame_idx == self.last_frame_idx:
                return
            self.last_frame_idx = frame_idx
        if len(pts_uvz) == 0:
            return

        depth = pts_uvz[:, 2]
        p_body = self.cam_cfg.backproject(pts_uvz[:, :2], depth)
        p_world = np.einsum('ij,nj->ni', quat_to_R(drone_quat), p_body) + np.asarray(drone_pos, dtype=float)

        ij = np.floor((p_world[:, :2] - self.origin) / self.cell).astype(int)
        inside = ((ij >= 0) & (ij < self.n)).all(axis=1)
        ij, z, depth = ij[inside], p_world[inside, 2], depth[inside]

        fB = self.cam_cfg.focal_px * self.cam_cfg.baseline
        sigma_z = depth ** 2 / fB * self.sigma_d
        w = 1.0 / np.maximum(sigma_z, 1e-6) ** 2

        np.add.at(self.sum_w, (ij[:, 0], ij[:, 1]), w)
        np.add.at(self.sum_wz, (ij[:, 0], ij[:, 1]), w * z)
        np.add.at(self.hits, (ij[:, 0], ij[:, 1]), 1)

    def height(self):
        """(n, n) height grid, NaN where the cell was not observed often enough."""
        h = np.full((self.n, self.n), np.nan)
        ok = self.hits >= self.min_hits
        h[ok] = self.sum_wz[ok] / self.sum_w[ok]
        return h

    def mesh(self):
        """
        Triangle mesh over observed cells: (V, 3) vertices at cell centres and (F, 3) faces.
        A quad is only built where all four corner cells are observed.
        """
        h = self.height()
        ok = ~np.isnan(h)
        if not ok.any():
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)

        index = np.full(h.shape, -1, dtype=np.int32)
        ii, jj = np.nonzero(ok)
        index[ii, jj] = np.arange(len(ii))
        xy = self.origin + (np.column_stack([ii, jj]) + 0.5) * self.cell
        verts = np.column_stack([xy, h[ii, jj]])

        a, b = index[:-1, :-1], index[1:, :-1]
        c, d = index[:-1, 1:], index[1:, 1:]
        quad = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
        a, b, c, d = a[quad], b[quad], c[quad], d[quad]
        faces = np.vstack([np.column_stack([a, b, d]), np.column_stack([a, d, c])])
        return verts, faces
