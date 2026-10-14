"""Render triangulated visual tracks and publish the processed stereo view."""

from collections.abc import Mapping

import cv2
import numpy as np


_LAYER_COLORS = (
    (255, 0, 0),
    (0, 220, 0),
    (0, 165, 255),
    (255, 0, 255),
    (255, 255, 0),
)


def render_visual_frame(frame, layer_tracks, depth_by_layer):
    """Render left feature meshes and right stereo matches side by side."""
    if len(frame.gray_frames) != 2:
        raise ValueError("Processed stereo frame must contain two grayscale images")

    left_gray, right_gray = frame.gray_frames
    if left_gray.shape != right_gray.shape or left_gray.ndim != 2:
        raise ValueError("Stereo display images must be equal-size grayscale images")

    left = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)
    track_items = (
        layer_tracks.items()
        if isinstance(layer_tracks, Mapping)
        else enumerate(layer_tracks)
    )
    for layer, tracks in track_items:
        color = _LAYER_COLORS[layer % len(_LAYER_COLORS)]
        points = np.asarray(tracks.new_points, dtype=np.float32).reshape(-1, 2)
        triangles = np.asarray(tracks.triangles, dtype=int).reshape(-1, 3)

        for triangle in triangles:
            if (np.all((triangle >= 0) & (triangle < len(points)))
                    and np.isfinite(points[triangle]).all()):
                polygon = np.rint(points[triangle]).astype(np.int32)
                cv2.polylines(left, [polygon], True, color, 1, cv2.LINE_AA)
        for point in points:
            if np.isfinite(point).all():
                cv2.circle(left, tuple(np.rint(point).astype(int)), 2, color, -1)

        if layer >= len(depth_by_layer):
            continue
        stereo = depth_by_layer[layer]
        right_points = np.asarray(stereo.right_points, dtype=np.float32).reshape(-1, 2)
        valid = np.asarray(stereo.valid, dtype=bool).reshape(-1)
        count = min(len(points), len(right_points), len(valid))
        valid = valid[:count] & np.isfinite(right_points[:count]).all(axis=1)

        for triangle in triangles:
            if (np.all((triangle >= 0) & (triangle < count))
                    and valid[triangle].all()):
                polygon = np.rint(right_points[triangle]).astype(np.int32)
                cv2.polylines(right, [polygon], True, color, 1, cv2.LINE_AA)
        for point in right_points[:count][valid]:
            cv2.circle(right, tuple(np.rint(point).astype(int)), 2, color, -1)

    cv2.putText(left, "Left: tracked feature mesh", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(right, "Right: stereo matches", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.hconcat((left, right))


class VisualFrameBridge:
    """Publish rendered stereo tracking and depth visualization as a JPEG topic."""

    def __init__(self, node, publish_ns="/stereoVisualReturn"):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage

        self.node = node
        self._message_type = CompressedImage
        namespace = "/" + publish_ns.strip("/")
        self.topic = f"{namespace}/processed/compressed"
        self._publisher = node.create_publisher(
            CompressedImage, self.topic, qos_profile_sensor_data
        )

    def publish(self, frame, layer_tracks, depth_by_layer):
        """Render the completed estimator outputs and publish one side-by-side JPEG."""
        visual = render_visual_frame(frame, layer_tracks, depth_by_layer)
        ok, encoded = cv2.imencode(".jpg", visual)
        if not ok:
            raise RuntimeError("OpenCV failed to encode the visualized stereo frame")

        message = self._message_type()
        message.format = "jpeg"
        message.data = encoded.tobytes()
        message.header.frame_id = "stereo_visual_return"
        if frame.timestamp is not None:
            seconds = int(frame.timestamp)
            message.header.stamp.sec = seconds
            message.header.stamp.nanosec = int(
                round((frame.timestamp - seconds) * 1_000_000_000)
            )
        self._publisher.publish(message)
        return visual

    def close(self):
        self.node.destroy_publisher(self._publisher)
