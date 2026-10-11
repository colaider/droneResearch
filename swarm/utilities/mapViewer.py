import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets


class MapViewer:
    """
    3D measured point-map viewer; legacy TerrainMap surfaces are also supported.

    Usage:
        viewer = MapViewer(title='Terrain Map')
        for step in range(10000):
            terrain.add(...)
            viewer.update(terrain, drone_pos)
    """

    def __init__(self, title='Terrain Map', size=(900, 700), update_every=10, distance=6.0):
        """
        update_every : refresh the map every N calls (higher = faster sim, choppier map)
        distance     : initial camera distance from the scene, m
        """
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
        else:
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
