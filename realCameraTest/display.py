"""Four-panel camera preview matching the swarm display, without Genesis."""
import cv2


def compose_camera_display(processed, tracking_gray):
        """Filtered color above the exact grayscale inputs used for feature tracking."""
        if len(processed) < 2 or len(tracking_gray) < 2:
            return None
        h, w = processed[0].shape[:2]
        scale = min(640 / w, 450 / h, 1.0)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))

        def tile(frame, label, grayscale=False):
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            image = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            image = cv2.copyMakeBorder(image, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
            cv2.putText(image, label, (10, 19), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (235, 235, 235), 1, cv2.LINE_AA)
            return image

        top = cv2.hconcat([tile(processed[0], "Left camera"), tile(processed[1], "Right camera")])
        bottom = cv2.hconcat([tile(tracking_gray[0], "Left tracking input", True), tile(tracking_gray[1], "Right tracking input", True)])
        return cv2.vconcat([top, bottom])
