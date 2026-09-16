import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import numpy as np
from collections import deque


class LivePlotter:
    """
    Real-time plotter for streaming numpy arrays.

    Usage:
        plotter = LivePlotter(
            plots={
                'Accelerometer': ['ax', 'ay', 'az'],
                'Gyroscope':     ['gx', 'gy', 'gz'],
            },
            buffer_size=1000,
            title='IMU Live',
        )

        for step in range(10000):
            # get some numpy arrays
            acc  = np.array([...])   # shape (3,)
            gyro = np.array([...])   # shape (3,)

            plotter.push('Accelerometer', acc)
            plotter.push('Gyroscope', gyro)
            plotter.update()   # refresh window
    """

    _COLORS = ['r', 'g', 'b', 'y', 'c', 'm', 'w']

    def __init__(self, plots: dict, buffer_size: int = 1000,
                 title: str = 'Live Plot', size=(900, 600),
                 update_every: int = 3):
        """
        plots         : dict mapping plot_name → list of series labels
                        e.g. {'Accel': ['ax','ay','az'], 'Gyro': ['gx','gy','gz']}
        buffer_size   : how many points to keep on screen
        title         : window title
        size          : (w, h) window size
        update_every  : redraw every N pushes (higher = faster sim, choppier plot)
        """
        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        self._win = pg.GraphicsLayoutWidget(title=title)
        self._win.resize(*size)
        self._win.show()

        self._buffer_size = buffer_size
        self._update_every = update_every
        self._counter = 0

        self._data = {}       # {plot_name: {label: deque}}
        self._curves = {}     # {plot_name: {label: curve}}
        self._plots = {}      # {plot_name: PlotItem}

        for i, (plot_name, labels) in enumerate(plots.items()):
            if i > 0:
                self._win.nextRow()
            plot = self._win.addPlot(title=plot_name)
            plot.addLegend()
            plot.showGrid(x=True, y=True, alpha=0.3)
            self._plots[plot_name] = plot

            self._data[plot_name] = {}
            self._curves[plot_name] = {}
            for j, label in enumerate(labels):
                color = self._COLORS[j % len(self._COLORS)]
                self._data[plot_name][label] = deque(maxlen=buffer_size)
                self._curves[plot_name][label] = plot.plot(pen=color, name=label)

    def push(self, plot_name: str, values):
        """
        Append values to a plot's series.

        values can be:
          - numpy array of length N (matches number of series)
          - list of N values
          - single scalar (if only one series in plot)
        """
        if plot_name not in self._data:
            raise KeyError(f"Unknown plot '{plot_name}'. Known: {list(self._data)}")

        values = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
        labels = list(self._data[plot_name].keys())

        if len(values) != len(labels):
            raise ValueError(
                f"'{plot_name}' expects {len(labels)} values, got {len(values)}"
            )

        for label, v in zip(labels, values):
            self._data[plot_name][label].append(v)

    def update(self, force: bool = False):
        """
        Redraw the plot. Call once per simulation step.
        Only actually redraws every `update_every` calls unless force=True.
        """
        self._counter += 1
        if not force and (self._counter % self._update_every != 0):
            return

        for plot_name, curves in self._curves.items():
            for label, curve in curves.items():
                curve.setData(list(self._data[plot_name][label]))

        QtWidgets.QApplication.processEvents()

    def close(self):
        self._win.close()