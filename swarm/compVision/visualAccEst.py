import cv2 
import numpy as np 

class EnumFrame:
    def __init__(self):
        self.frames = []
        self.idx = 0


class VisulaAcEst:
    def __init__(self):
        self.current_frame = EnumFrame() 
        self.buffer = []

    def update_frame(self, frame, idx):
        self.current_frame.frames = frame
        self.current_frame.idx = idx
        if(len(self.buffer) >= 100):
            self.buffer.pop(0)

        self.buffer.append(self.current_frame)

    def processing(self, frame, idx):
        self.update_frame(frame,idx)
        # self.find_edges()
        if len(self.buffer) >= 2:
            annotated_frame, _, _, _ = self.lucas_kanade_flow()
            self.current_frame.frames[0] = annotated_frame

        return self.current_frame.frames




    def lucas_kanade_flow(self):
        gray1 = cv2.cvtColor(self.buffer[-2].frames[0], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.buffer[-1].frames[0], cv2.COLOR_BGR2GRAY)

        # ---------------------------------------------------------
        # 2. Find features in frame1
        # ---------------------------------------------------------
        old_points = cv2.goodFeaturesToTrack(
            gray1,
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7
        )

        # Make a copy so we don't modify the original frame2
        annotated_frame = self.buffer[-1].frames[0].copy()

        if old_points is None: return annotated_frame, None, None, None

        # ---------------------------------------------------------
        # 3. Lucas-Kanade optical flow
        # ---------------------------------------------------------
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            gray1,
            gray2,
            old_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS |
                cv2.TERM_CRITERIA_COUNT,
                30,
                0.01
            )
        )

        if new_points is None: return annotated_frame, None, None, None

        status = status.ravel()

        old_points = old_points.reshape(-1, 2)
        new_points = new_points.reshape(-1, 2)

        good_old = old_points[status == 1]
        good_new = new_points[status == 1]

        flow = good_new - good_old
        for old, new, motion in zip(good_old, good_new, flow):

            # Coordinates in frame1
            x1, y1 = old.astype(int)
            x2, y2 = new.astype(int)

            # Motion vector
            dx, dy = motion

            # Draw old position in GREEN
            cv2.circle(
                annotated_frame,
                (x2, y2),
                3,
                (0, 255, 0),
                -1
            )

            # Draw arrow showing motion
            cv2.arrowedLine(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                1,
                tipLength=0.3
            )

        return annotated_frame, good_old, good_new, flow
