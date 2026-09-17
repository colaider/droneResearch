import cv2 
import numpy as np 

class EnumFrame:
    def __init__(self):
        self.frames = []
        self.idx = 0


class VisulaAcEst:
    def __init__(self, res, fov):
        self.current_frame = EnumFrame() 
        self.buffer = []
        self.foc_l = (res[0]/2)/ np.tan(fov/2)


    def update_frame(self, frame, idx):
        self.current_frame.frames = frame
        self.current_frame.idx = idx
        if not hasattr(self, 'frame_size'):
            self.frame_size = np.shape(frame[0])
                    
        if(len(self.buffer) >= 100):
            self.buffer.pop(0)

        self.buffer.append(self.current_frame)


    def processing(self, frame, idx):
        self.update_frame(frame,idx)        
        flow = self.aply_flow()
        return self.current_frame.frames


    def aply_flow(self):
        processed = []
        flow = 0
        for i, f in enumerate(self.current_frame.frames):
            if len(self.buffer) >= 2:  
                frame1 = self.buffer[-2].frames[i]
                frame2 = self.buffer[-1].frames[i]
                fr, flow = self.lucas_kanade_flow(frame1, frame2)
                processed.append(fr)

        self.current_frame.frames = processed
        return flow


    def lucas_kanade_flow(self, frame1, frame2):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # gray2 = self.normalize_lighting(gray1, gray2).copy()

        old_points = cv2.goodFeaturesToTrack(
            gray1,
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7
        )

        annotated_frame = frame2.copy()

        if old_points is None: return annotated_frame, None
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            gray1,
            gray2,
            old_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        if new_points is None: return annotated_frame, None

        status = status.ravel()

        old_points = old_points.reshape(-1, 2)
        new_points = new_points.reshape(-1, 2)

        ys, xs, _ = self.frame_size
        diameter = (xs if xs < ys else ys) * 0.92
        
        good_old = old_points[status == 1]
        good_new = new_points[status == 1]

        mask_new = np.sqrt((good_new[:, 0] - xs / 2)**2 + (good_new[:, 1]-ys/2)**2) < diameter / 2
        mask_old = np.sqrt((good_old[:, 0] - xs / 2)**2 + (good_old[:, 1]-ys/2)**2) < diameter / 2
        central_good_new = good_new[mask_new]
        central_good_old = good_old[mask_old]

        central_flow = central_good_new - central_good_old

        for new in good_new:
            x2, y2 = new.astype(int)
            collor = (0, 100,200 ) if (x2-xs/2)**2 + (y2-ys/2)**2 < (diameter / 2)**2 else (0, 255, 0)
            cv2.circle(annotated_frame, (x2, y2), 3, collor , -1)

        return annotated_frame, central_flow

    @staticmethod
    def normalize_lighting(prev_gray, curr_gray):
        """Compensate for global gain and offset between frames."""
        # Estimate gain (contrast) and offset (brightness)
        mean_prev = prev_gray.mean()
        mean_curr = curr_gray.mean()
        std_prev = prev_gray.std()
        std_curr = curr_gray.std()
        
        gain = std_prev / (std_curr + 1e-6)
        offset = mean_prev - gain * mean_curr
        
        # Apply to current frame
        normalized = curr_gray.astype(np.float32) * gain + offset
        return np.clip(normalized, 0, 255).astype(np.uint8)