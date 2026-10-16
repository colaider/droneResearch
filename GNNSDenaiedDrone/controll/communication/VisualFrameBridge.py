"""Render triangulated visual tracks and publish the processed stereo view."""

import cv2
import numpy as np

_RIGHT_MESH_COLOR = (110, 110, 110)
_NO_DEPTH_COLOR = (170, 170, 170)
_TURBO_START = 25           # skip TURBO's near-black end, it vanishes on dark image areas


def _turbo(level):
    """BGR TURBO colours for levels in [0, 1]."""
    index = _TURBO_START + (255 - _TURBO_START) * np.clip(np.asarray(level, float), 0, 1)
    return cv2.applyColorMap(index.astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)[:, 0]


def depth_colors(depths, depth_range):
    """BGR colours for depths in metres: blue = near end of depth_range, red = far."""
    near, far = depth_range
    return _turbo((np.asarray(depths, float) - near) / max(far - near, .01))


def frame_depth_range(stereo):
    """(near, far) metres from this frame's valid depths (5th-95th percentile, >= 0.1 m wide),
    or None when no point has a valid depth."""
    depths = np.asarray(stereo.depths, float)[np.asarray(stereo.valid, bool)]
    depths = depths[np.isfinite(depths)]
    if not len(depths):
        return None
    near, far = np.percentile(depths, [5, 95])
    middle, half = (near + far) / 2, max((far - near) / 2, .05)
    return middle - half, middle + half


def _outlined_text(image, text, origin, scale=.4):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_depth_legend(image, depth_range):
    """Depth colour bar in the bottom-left corner, labelled with the near/far depth in metres."""
    width = min(160, image.shape[1] - 16)
    x0, y0 = 8, image.shape[0] - 14
    image[y0:y0 + 8, x0:x0 + width] = _turbo(np.linspace(0, 1, width))
    near, far = depth_range
    far_label = f"{far:.2f} m"
    (far_width, _), _ = cv2.getTextSize(far_label, cv2.FONT_HERSHEY_SIMPLEX, .4, 1)
    _outlined_text(image, f"{near:.2f} m", (x0, y0 - 5))
    _outlined_text(image, far_label, (x0 + width - far_width, y0 - 5))


def render_visual_frame(frame, tracks, stereo, depth_range=None):
    """Render tracked features and stereo matches side by side.

    Right-image matches are coloured by their stereo depth over depth_range (near, far) metres,
    or over this frame's own depth spread when it is None; matches without a valid depth are
    grey rings.
    """
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
    right_points = right_points[:count]
    matched = np.isfinite(right_points).all(axis=1)
    depths = np.asarray(stereo.depths, float)[:count]
    has_depth = matched & np.asarray(stereo.valid, bool)[:count] & np.isfinite(depths)
    if depth_range is None:
        depth_range = frame_depth_range(stereo)

    for triangle in triangles:
        if (np.all((triangle >= 0) & (triangle < count))
                and matched[triangle].all()):
            polygon = np.rint(right_points[triangle]).astype(np.int32)
            cv2.polylines(right, [polygon], True, _RIGHT_MESH_COLOR, 1, cv2.LINE_AA)
    for point in right_points[matched & ~has_depth]:
        cv2.circle(right, tuple(np.rint(point).astype(int)), 2, _NO_DEPTH_COLOR, 1)
    if has_depth.any() and depth_range is not None:
        order = np.argsort(-depths[has_depth])          # far first, so near points end on top
        for point, dot_color in zip(right_points[has_depth][order],
                                    depth_colors(depths[has_depth][order], depth_range)):
            cv2.circle(right, tuple(np.rint(point).astype(int)), 3, tuple(map(int, dot_color)), -1)
    if depth_range is not None:
        _draw_depth_legend(right, depth_range)

    cv2.putText(left, "Left: tracked feature mesh", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(right, f"Right: depth {int(has_depth.sum())}/{int(matched.sum())} matches", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.hconcat((left, right))


class VisualFrameBridge:
    """Publish rendered stereo tracking and depth visualization as a JPEG topic."""

    def __init__(self, node, publish_ns="/stereoVisualReturn", depth_range_smoothing=0.8):
        """depth_range_smoothing: low-pass weight of the previous colour range (0 = per-frame
        range only), so the depth colours don't flicker from frame to frame."""
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage

        self.node = node
        self.depth_range_smoothing = depth_range_smoothing
        self._depth_range = None
        self._message_type = CompressedImage
        namespace = "/" + publish_ns.strip("/")
        self.topic = f"{namespace}/processed/compressed"
        self._publisher = node.create_publisher(
            CompressedImage, self.topic, qos_profile_sensor_data
        )

    def publish(self, frame, tracks, stereo):
        """Render the completed estimator outputs and publish one side-by-side JPEG."""
        current = frame_depth_range(stereo)
        if current is not None:
            if self._depth_range is not None:
                a = self.depth_range_smoothing
                current = tuple(a * old + (1 - a) * new
                                for old, new in zip(self._depth_range, current))
            self._depth_range = current
        visual = render_visual_frame(frame, tracks, stereo, self._depth_range)
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
