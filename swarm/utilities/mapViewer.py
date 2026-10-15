from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets


def triangle_lines(points, max_edge=0.75, max_points=6000):
    """Display-only local triangles in XYZ; not reconstructed surface faces.

    Each point chooses the shortest non-degenerate triangle among its five
    nearest neighbours. Shared edges are drawn once. Long edges are discarded
    in 3D, so separated roofs/ground are not joined merely because XY overlaps.
    Sampling bounds display cost; it never changes the stored map points.
    """
    if not np.isfinite(max_edge) or max_edge <= 0 or max_points < 3:
        raise ValueError('Triangle edge limit must be positive and point budget at least 3')
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > max_points:
        points = points[np.linspace(0, len(points)-1, max_points, dtype=int)]
    points = np.unique(points, axis=0)
    empty = np.empty((0, 3), dtype=np.float32)
    if len(points) < 3:
        return empty
    _, neighbours = cKDTree(points).query(points, k=min(6, len(points)))
    pairs = np.array(list(combinations(range(1, neighbours.shape[1]), 2)))
    a = neighbours[:, pairs[:, 0]]
    b = neighbours[:, pairs[:, 1]]
    va = points[a] - points[:, None]
    vb = points[b] - points[:, None]
    lengths = np.stack((np.linalg.norm(va, axis=2), np.linalg.norm(vb, axis=2),
                        np.linalg.norm(points[a]-points[b], axis=2)), axis=2)
    # Normalized triangle quality: 1 for equilateral, 0 for collinear.
    quality = (2*np.sqrt(3)*np.linalg.norm(np.cross(va, vb), axis=2)
               / np.maximum(np.sum(lengths**2, axis=2), 1e-20))
    good = (lengths.max(axis=2) <= max_edge) & (quality >= 0.15)
    cost = np.where(good, lengths.sum(axis=2), np.inf)
    best = cost.argmin(axis=1)
    rows = np.flatnonzero(good.any(axis=1))
    if not len(rows):
        return empty
    faces = np.column_stack((rows, a[rows, best[rows]], b[rows, best[rows]]))
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return points[edges].reshape(-1, 3).astype(np.float32)


class MapViewer:
    """
    3D measured point-map viewer; legacy TerrainMap surfaces are also supported.

    Usage:
        viewer = MapViewer(title='Terrain Map')
        for step in range(10000):
            terrain.add(...)
            viewer.update(terrain, drone_pos)
    """

    def __init__(self, title='Terrain Map', size=(900, 700), update_every=10, distance=6.0,
                 show_triangles=True, triangle_max_edge=0.75, triangle_max_points=6000):
        """
        update_every : refresh the map every N calls (higher = faster sim, choppier map)
        distance     : initial camera distance from the scene, m
        show_triangles : overlay local triangular connections on measured points
        triangle_max_edge : maximum displayed connection length in metres
        triangle_max_points : maximum points used for lines (all points stay visible)
        """
        # Validate display settings before creating the window.
        triangle_lines([], triangle_max_edge, triangle_max_points)
        self.show_triangles = show_triangles
        self.triangle_max_edge = triangle_max_edge
        self.triangle_max_points = triangle_max_points
        self._triangle_revision = None
        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        self._win = gl.GLViewWidget()
        self._win.setWindowTitle(title)
        self._win.resize(*size)
        self._win.setCameraPosition(distance=distance, elevation=35, azimuth=-60)
        self._win.show()

        grid = gl.GLGridItem()
        grid.setSize(30, 30)
        grid.setSpacing(1, 1)
        self._win.addItem(grid)

        self._surface = None  # create a mesh only when actual faces exist
        self._points = gl.GLScatterPlotItem(pos=np.empty((0, 3)), size=3, pxMode=True)
        self._win.addItem(self._points)
        self._triangle_edges = gl.GLLinePlotItem(
            pos=np.empty((0, 3)), color=(0.45, 0.75, 1.0, 0.55),
            width=1, mode='lines', antialias=True, glOptions='translucent')
        self._win.addItem(self._triangle_edges)
        self._axes = gl.GLAxisItem()
        self._axes.setSize(1, 1, 1)
        self._win.addItem(self._axes)
        self._drone = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(1, 0, 0, 1), size=12)
        self._win.addItem(self._drone)

        self._cmap = pg.colormap.get('viridis')
        self._update_every = update_every
        self._counter = 0

    def update(self, terrain, drone_pos=None, force=False):
        QtWidgets.QApplication.processEvents()
        self._counter += 1
        if not force and (self._counter % self._update_every != 0):
            return

        if hasattr(terrain, 'points'):
            if self._surface is not None:
                self._surface.setVisible(False)
            self._points.setVisible(True)
            points = terrain.points()
            colors = np.column_stack((terrain.colors(), np.ones(len(points))))
            self._points.setData(pos=points, color=colors, size=3)
            self._triangle_edges.setVisible(self.show_triangles)
            frame_idx = getattr(terrain, 'last_frame_idx', None)
            revision = (id(terrain), frame_idx, self.triangle_max_edge, self.triangle_max_points)
            if self.show_triangles and (frame_idx is None or revision != self._triangle_revision):
                lines = triangle_lines(points, self.triangle_max_edge, self.triangle_max_points)
                self._triangle_edges.setData(pos=lines)
                self._triangle_revision = revision
        else:
            self._triangle_edges.setVisible(False)
            self._triangle_revision = None
            self._points.setVisible(False)
            verts, faces = terrain.mesh()
            if self._surface is not None:
                self._surface.setVisible(bool(len(faces)))
            if len(faces):
                if self._surface is None:
                    self._surface = gl.GLMeshItem(smooth=False, drawEdges=False, shader='shaded', glOptions='opaque')
                    self._win.addItem(self._surface)
                z = verts[faces].mean(axis=1)[:, 2]
                lo, hi = np.percentile(z, [2, 98])
                colors = self._cmap.map(np.clip((z - lo) / max(hi - lo, 1e-3), 0, 1), mode='float')
                self._surface.setMeshData(vertexes=verts, faces=faces, faceColors=colors)

        if drone_pos is not None:
            self._drone.setData(pos=np.asarray(drone_pos, dtype=float).reshape(1, 3))

        QtWidgets.QApplication.processEvents()

    def close(self):
        self._win.close()
