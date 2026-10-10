"""Large structures on coarse pyramid layers + dense stereo -> TerrainMap height grid.

Runs alongside ``VisulaAcEst`` on the ``Frame`` it returns (its pyramids are
reused, nothing is recomputed). Used by the simulator ``DroneStruct`` and by
the real-camera runners.
"""

import numpy as np

from swarm.compVision.denseStereo import DenseStereo, estimate_disparity_direction, map_points
from swarm.compVision.largeFeatures import LargeFeatureDetector, STRUCTURE_COLORS, draw_large_features
from swarm.compVision.terrainMap import TerrainMap
from swarm.config.cameraConfig import StereoCameraConfig


class StructureMapper:
    """Detect structures, measure their depth and feed a ``TerrainMap``.

    cam_cfg      : ``StereoCameraConfig``-like (focal_px, baseline, res,
                   backproject, disparity_direction) of the (rectified) pair
    levels       : pyramid levels for large features (coarse = grass-free)
    depth_level  : pyramid level for dense SGBM; None disables depth/map
                   (2D detection only, e.g. unrectified real cameras)
    check_direction : measure the stereo shift on the first frame and warn /
                   correct when it disagrees with ``cam_cfg.disparity_direction()``
    """

    def __init__(self, cam_cfg, levels=(2, 3), depth_level=1, terrain=None,
                 min_depth=2.0, check_direction=True, detector=None):
        self.cam_cfg = cam_cfg
        self.levels = tuple(levels)
        self.depth_level = depth_level
        self.min_depth = min_depth
        self.check_direction = check_direction
        self.detector = detector or LargeFeatureDetector()
        self.terrain = terrain if terrain is not None else TerrainMap(cam_cfg)
        self.dense = None
        self.features = {}
        self.depth_map = None
        self.points = np.empty((0, 4))

    def step(self, frame, drone_pos=None, drone_quat=None):
        """Process one ``Frame``; returns the (u, v, Z, weight) map points.

        Without a pose the map is built in the drone body frame at the origin.
        """
        left_pyr, right_pyr = frame.pyramids[0], frame.pyramids[1]
        self.features = self.detector.detect(left_pyr, levels=self.levels)
        self.depth_map, self.points = None, np.empty((0, 4))
        if self.depth_level is None:
            return self.points
        lv = min(self.depth_level, len(left_pyr) - 1)
        if self.dense is None:
            self.dense = self._make_dense(left_pyr[lv], right_pyr[lv], lv)
        self.depth_map = self.dense.compute(left_pyr[lv], right_pyr[lv], lv)
        self.points = map_points(self.features, self.depth_map)
        if len(self.points):
            pos = np.zeros(3) if drone_pos is None else np.asarray(drone_pos, float).reshape(-1)[:3]
            quat = (1.0, 0.0, 0.0, 0.0) if drone_quat is None else np.asarray(drone_quat, float).reshape(-1)[:4]
            self.terrain.add(self.points[:, :3], pos, quat, frame.idx, weight_scale=self.points[:, 3])
        return self.points

    def draw(self, image, scale=1.0):
        """Overlay this frame's structures on a BGR display image (in place)."""
        for level, features in self.features.items():
            draw_large_features(image, features, STRUCTURE_COLORS[level % len(STRUCTURE_COLORS)], scale)
        return image

    def stats(self):
        structure = self.points[self.points[:, 3] > 1] if len(self.points) else self.points
        return {
            'quads': int(sum(len(f.quads) for f in self.features.values())),
            'lines': int(sum(len(f.lines) for f in self.features.values())),
            'corners': int(sum(len(f.corners) for f in self.features.values())),
            'map_points': int(len(self.points)),
            'structure_depth_m': (round(float(np.median(structure[:, 2])), 2) if len(structure) else None),
        }

    def _make_dense(self, left, right, level):
        configured = np.asarray(self.cam_cfg.disparity_direction(), float)
        direction = configured
        if self.check_direction:
            measured, shift = estimate_disparity_direction(left, right)
            if measured is not None and shift * 2 ** level > 2.0 and measured @ configured < 0.7:
                # Snap to the nearest image axis; SGBM needs an axis-aligned baseline.
                axis = int(np.argmax(np.abs(measured)))
                direction = np.zeros(2)
                direction[axis] = np.sign(measured[axis])
                print(f"[structure] disparity_direction() = {configured.round(2)} but the images "
                      f"shift along {measured.round(2)} ({shift * 2 ** level:.1f}px); using {direction}")
        return DenseStereo(self.cam_cfg.focal_px, self.cam_cfg.baseline, direction,
                           min_depth=self.min_depth)


def camera_for(rectifier, res):
    """Camera config for images rectified by ``rectifier`` (or raw, when None).

    ``res`` is the (width, height) the estimator sees. The rectified pinhole
    focal length / baseline / disparity direction replace the nominal ones.
    """
    if rectifier is None or rectifier.is_identity:
        return StereoCameraConfig(res=tuple(res))
    cfg = StereoCameraConfig(res=tuple(res), left_focal=rectifier.focal_px,
                             right_focal=rectifier.focal_px, baseline=rectifier.baseline)
    direction = np.asarray(rectifier.disparity_direction, float)
    cfg.disparity_direction = lambda: direction   # measured by calibration, not mount geometry
    return cfg
