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
        new_frame_data = EnumFrame()
        new_frame_data.frames = self.add_noise([f.copy() for f in frame])  
        new_frame_data.idx = idx
        
        if not hasattr(self, 'frame_size'):
            self.frame_size = np.shape(frame[0])
        
        # Buffer holds the new object
        if len(self.buffer) >= 100:
            self.buffer.pop(0)
        self.buffer.append(new_frame_data)
        
        self.current_frame = new_frame_data


    def processing(self, frame, idx):
        self.update_frame(frame,idx)        
        flow = self.aply_flow()
        return self.current_frame.frames


    def aply_flow(self):
        processed = []
        flow = 0
       
        if len(self.buffer) < 2:
            return flow   # need at least 2 frames
        
        for i in range(min(len(self.buffer[-2].frames), len(self.buffer[-1].frames))):
            frame1 = self.buffer[-2].frames[i]
            frame2 = self.buffer[-1].frames[i]
            fr, flow = self.lucas_kanade_flow(frame1, frame2)
            processed.append(fr)
        
        self.current_frame.frames = processed
        print(flow)
        return flow


    def lucas_kanade_flow(self, frame1, frame2):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        old_points = cv2.goodFeaturesToTrack(
            gray1,
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7
        )

        annotated_frame = frame2.copy()

        if old_points is None: return annotated_frame, None
        new_points, status_fwd, error_fwd = cv2.calcOpticalFlowPyrLK(
            gray1, gray2, old_points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # Backward: frame2 → frame1
        back_points, status_bwd, error_bwd = cv2.calcOpticalFlowPyrLK(
            gray2, gray1, new_points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        if new_points is None: return annotated_frame, None

        diff = np.abs(old_points - back_points).reshape(-1, 2).max(axis=1)
        good_mask = (status_fwd.ravel() == 1) & (diff < 0.5)   

        old_points = old_points.reshape(-1, 2)
        new_points = new_points.reshape(-1, 2)

        ys, xs, _ = self.frame_size
        diameter = (xs if xs < ys else ys) * 0.92
    
        good_old = old_points[good_mask]
        good_new = new_points[good_mask]


        mask = np.sqrt((good_new[:, 0] - xs / 2)**2 + (good_new[:, 1]-ys/2)**2) < diameter / 2
        central_good_new = good_new[mask]
        central_good_old = good_old[mask]

        central_flow = central_good_new - central_good_old

        for new in good_new:
            x2, y2 = new.astype(int)
            collor = (0, 100,200 ) if (x2-xs/2)**2 + (y2-ys/2)**2 < (diameter / 2)**2 else (0, 255, 0)
            cv2.circle(annotated_frame, (x2, y2), 3, collor , -1)

        return annotated_frame, central_flow

    
    @staticmethod
    def add_noise(frames, sigma=50):
        out = []
        for frame in frames:
            noise = np.random.normal(0, sigma, frame.shape)
            noisy = frame.astype(np.float32) + noise
            out.append(np.clip(noisy, 0, 255).astype(np.uint8))
        return out
