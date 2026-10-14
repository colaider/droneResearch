"""Image display composition helpers for visual-estimation outputs."""

import cv2
import numpy as np


def compose_camera_display(processed, tracking_gray):
    """Filtered color above the exact grayscale inputs used for feature tracking."""
    if len(processed) < 2 or len(tracking_gray) < 2:
        return None
    height, width = processed[0].shape[:2]
    scale = min(640 / width, 450 / height, 1.0)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))

    def tile(frame, label, grayscale=False):
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        image = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        image = cv2.copyMakeBorder(
            image, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24)
        )
        cv2.putText(
            image, label, (10, 19), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (235, 235, 235), 1, cv2.LINE_AA,
        )
        return image

    top = cv2.hconcat([
        tile(processed[0], "Left camera"),
        tile(processed[1], "Right camera"),
    ])
    bottom = cv2.hconcat([
        tile(tracking_gray[0], "Left tracking input", True),
        tile(tracking_gray[1], "Right tracking input", True),
    ])
    return cv2.vconcat([top, bottom])


def compose_layer_display(gray, points, links, layer, color):
    """Draw one pyramid layer's tracks; points use that layer's native pixels."""
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    points = np.asarray(points).reshape(-1, 2)
    pixels = np.rint(points).astype(int)
    for a, b in links:
        cv2.line(image, tuple(pixels[a]), tuple(pixels[b]), color, 1)
    for x, y in pixels:
        cv2.circle(image, (x, y), 2, color, -1)
    height, width = gray.shape
    image = cv2.copyMakeBorder(
        image, 28, 0, 0, max(0, 330 - width), cv2.BORDER_CONSTANT,
        value=(32, 32, 32),
    )
    label = f'Layer {layer} | {width}x{height} | {len(points)} points'
    cv2.putText(
        image, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .45,
        (235, 235, 235), 1, cv2.LINE_AA,
    )
    cv2.line(image, (0, 26), (image.shape[1] - 1, 26), color, 2)
    return image


def compose_depth_display(gray, points, total_points):
    """Draw sparse measured depth; label the per-frame depth colour range in metres."""
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    height, width = gray.shape
    scale = min(640 / width, 450 / height, 1.0)
    image = cv2.resize(image, (round(width * scale), round(height * scale)))
    label = f'Stereo depth: {len(points)}/{total_points} points'
    if len(points):
        low, high = np.percentile(points[:, 2], [2, 98])
        level = np.clip((points[:, 2] - low) / max(high - low, .01), 0, 1)
        colors = cv2.applyColorMap(
            (255 * level).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO
        )[:, 0]
        for (u, v, _), color in zip(points, colors):
            cv2.circle(
                image, (round(u * scale), round(v * scale)), 3,
                tuple(map(int, color)), -1,
            )
        label += f' | blue {low:.2f}m - red {high:.2f}m'
    image = cv2.copyMakeBorder(
        image, 30, 0, 0, max(0, 640 - image.shape[1]), cv2.BORDER_CONSTANT
    )
    cv2.putText(
        image, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .45,
        (240, 240, 240), 1, cv2.LINE_AA,
    )
    return image


def draw_tracks(annotated_frame, annotated_right, points_left, points_right, links,
                color=(255, 0, 0)):
    """Draw tracked feature links and points on the left and available right-image points."""
    for a, b in links:
        cv2.line(
            annotated_frame, tuple(points_left[a].astype(int)),
            tuple(points_left[b].astype(int)), color, 2,
        )
    for x, y in points_left.astype(int):
        cv2.circle(annotated_frame, (x, y), 2, color, -1)

    valid_right = np.isfinite(points_right).all(axis=1)
    for a, b in links:
        if valid_right[a] and valid_right[b]:
            cv2.line(
                annotated_right, tuple(points_right[a].astype(int)),
                tuple(points_right[b].astype(int)), color, 2,
            )
    for x, y in points_right[valid_right].astype(int):
        cv2.circle(annotated_right, (x, y), 2, color, -1)
    return annotated_frame, annotated_right
