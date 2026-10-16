"""The 3D reconstruction: depth points from /vo/points accumulated into a voxel map.

The control loop (GNNSDenaiedDrone/controll/communication/PointsPublisher.py) sends each new
depth frame's points already in the world frame, placed with the fused pose. Here they are
snapped to VOXEL_M cubes and averaged per cube, as swarm/mapping/terrainMap.py's PointMap does in
the simulator: repeated looks at the same surface settle on one point instead of piling up, and
the map is bounded (the least recently seen cubes go first beyond MAX_POINTS).

Per flight run (the same runs as the position tracks, tracks.py) the raw frames are appended to
<state_dir>/tracks/<run id>.cloud.jsonl, one line per frame: [t, [[x, y, z], ...]], t in seconds
since the run started. Replaying a run rebuilds its map from that file.
"""
from __future__ import annotations

import collections
import json
import math

VOXEL_M = 0.05
MAX_POINTS = 50000


def decode_points(value):
    """The link's vo_points value -> (stamp, [[x, y, z], ...]) of finite points, or None."""
    if not isinstance(value, dict) or not isinstance(value.get("points"), list):
        return None
    points = []
    for p in value["points"]:
        try:
            x, y, z = float(p[0]), float(p[1]), float(p[2])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
            points.append([x, y, z])
    return value.get("stamp"), points


class CloudMap:
    """Voxel-averaged points; version counts the frames added, so readers can tell it changed."""

    def __init__(self, voxel=VOXEL_M, max_points=MAX_POINTS):
        self.voxel = voxel
        self.max_points = max_points
        self._cells = collections.OrderedDict()     # (i, j, k) -> [sum_x, sum_y, sum_z, count]
        self.frames = 0
        self.raw_points = 0

    def add(self, points):
        for x, y, z in points:
            key = (math.floor(x / self.voxel), math.floor(y / self.voxel), math.floor(z / self.voxel))
            cell = self._cells.get(key)
            if cell is None:
                self._cells[key] = [x, y, z, 1]
            else:
                cell[0] += x
                cell[1] += y
                cell[2] += z
                cell[3] += 1
                self._cells.move_to_end(key)
        while len(self._cells) > self.max_points:
            self._cells.popitem(last=False)
        self.frames += 1
        self.raw_points += len(points)

    def points(self):
        return [[round(c[0] / c[3], 3), round(c[1] / c[3], 3), round(c[2] / c[3], 3)]
                for c in self._cells.values()]

    def summary(self):
        return {"frames": self.frames, "points": len(self._cells), "raw_points": self.raw_points,
                "voxel_m": self.voxel}

    @classmethod
    def from_file(cls, path):
        cloud = cls()
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        _t, points = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    cloud.add(points)
        except OSError:
            return None
        return cloud
