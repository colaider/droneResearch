"""Render triangulated visual tracks and publish the processed stereo view."""

import cv2
import numpy as np

def render_visual_frame(frame, tracks, stereo):
    """Render tracked features and stereo matches side by side."""
    if len(frame.gray_frames) != 2:
        raise ValueError("Processed stereo frame must contain two grayscale images")

    left_gray, right_gray = frame.gray_frames
    if left_gray.shape != right_gray.shape or left_gray.ndim != 2:
        raise ValueError("Stereo display images must be equal-size grayscale images")

    left = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)
    points = np.asarray(tracks.new_points, dtype=np.float32).reshape(-1, 2)
    triangles = np.asarray(tracks.triangles, dtype=int).reshape(-1, 3)
    color = (255, 0, 0)
    for triangle in triangles:
        if (np.all((triangle >= 0) & (triangle < len(points)))
                and np.isfinite(points[triangle]).all()):
            polygon = np.rint(points[triangle]).astype(np.int32)
            cv2.polylines(left, [polygon], True, color, 1, cv2.LINE_AA)
    for point in points:
        if np.isfinite(point).all():
            cv2.circle(left, tuple(np.rint(point).astype(int)), 2, color, -1)

    right_points = np.asarray(stereo.right_points, dtype=np.float32).reshape(-1, 2)
    count = min(len(points), len(right_points))
    matched = np.isfinite(right_points[:count]).all(axis=1)
    for triangle in triangles:
        if (np.all((triangle >= 0) & (triangle < count))
                and matched[triangle].all()):
            polygon = np.rint(right_points[triangle]).astype(np.int32)
            cv2.polylines(right, [polygon], True, color, 1, cv2.LINE_AA)
    for point in right_points[:count][matched]:
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

    def publish(self, frame, tracks, stereo):
        """Render the completed estimator outputs and publish one side-by-side JPEG."""
        visual = render_visual_frame(frame, tracks, stereo)
        ok, encoded = cv2.imencode(".jpg", visual)
        if not ok:
            raise RuntimeError("OpenCV failed to encode the visualized stereo frame")

        message = self._message_type()
        message.format = "jpeg"
        message.data = encoded.tobytes()
        message.header.frame_id = "stereo_visual_return"
        if frame.timestamp_ns is not None:
            message.header.stamp.sec, message.header.stamp.nanosec = divmod(
                int(frame.timestamp_ns), 1_000_000_000
            )
        elif frame.timestamp is not None:
            seconds = int(frame.timestamp)
            message.header.stamp.sec = seconds
            message.header.stamp.nanosec = int(
                round((frame.timestamp - seconds) * 1_000_000_000)
            )
        self._publisher.publish(message)
        return visual

    def close(self):
        self.node.destroy_publisher(self._publisher)
