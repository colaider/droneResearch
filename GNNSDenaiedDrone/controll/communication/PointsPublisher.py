"""Publish the visual estimator's depth points in the world frame on /vo/points, for drone_gui.

Each new depth frame's points (estimator.depth_points: left-image pixel + depth along the optical
axis, up to ~300 tracked features) are back-projected with the camera config, rotated by the
attitude and moved by the fused position the estimator recorded for that frame (avg_att,
avg_pos: the same pose the visual velocity is rotated with in Sensors.process_visual_frame),
and sent as sensor_msgs/PointCloud2 with float32 x, y, z in that world frame. drone_gui
accumulates them into a voxel map (its 3D reconstruction window), as swarm/mapping/terrainMap.py's
PointMap does in the simulator.

Like /vo/status this is display only: nothing on the drone subscribes to it.
"""

import math
import time

import numpy as np


def body_to_world(roll, pitch, yaw):
    """Z-Y-X rotation, the same matrix Sensors.process_visual_frame builds."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def world_points(estimator):
    """(N, 3) world points of the estimator's latest depth frame (finite ones only)."""
    pts = np.asarray(estimator.depth_points, float).reshape(-1, 3)
    good = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
    pts = pts[good]
    if not len(pts):
        return pts
    body = estimator.camera_config.backproject(pts[:, :2], pts[:, 2])
    rotation = body_to_world(*np.asarray(estimator.avg_att, float))
    world = body @ rotation.T + np.asarray(estimator.avg_pos, float)
    return world[np.isfinite(world).all(axis=1)]


class PointsPublisher:
    def __init__(self, node, ns="/vo", rate_hz=5.0, frame_id="world"):
        from sensor_msgs.msg import PointCloud2, PointField

        self._cloud_type, self._field_type = PointCloud2, PointField
        self._publisher = node.create_publisher(PointCloud2, "/" + ns.strip("/") + "/points", 10)
        self.node = node
        self.frame_id = frame_id
        self.period = 1.0 / rate_hz
        self._next = 0.0
        self._last_frame_idx = None

    def publish(self, sensors):
        """Call every control step; sends each new depth frame once, at most rate_hz."""
        estimator = sensors.visual_estimator
        frame_idx = getattr(estimator, "depth_frame_idx", None)
        now = time.monotonic()
        if frame_idx is None or frame_idx < 0 or frame_idx == self._last_frame_idx or now < self._next:
            return
        self._last_frame_idx = frame_idx
        self._next = now + self.period
        points = world_points(estimator)
        if len(points):
            self._publisher.publish(self._cloud(points))

    def _cloud(self, points):
        field = self._field_type
        msg = self._cloud_type()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height, msg.width = 1, len(points)
        msg.fields = [field(name=n, offset=4 * i, datatype=field.FLOAT32, count=1)
                      for i, n in enumerate("xyz")]
        msg.is_bigendian = False
        msg.point_step, msg.row_step = 12, 12 * len(points)
        msg.is_dense = True
        msg.data = np.ascontiguousarray(points, dtype="<f4").tobytes()
        return msg

    def close(self):
        self.node.destroy_publisher(self._publisher)
