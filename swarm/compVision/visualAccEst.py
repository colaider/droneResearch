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
        self.buffer.append(self.current_frame)

    def find_edges(self, sample_step=8, point_radius=2, edge_color=(0, 255, 0), point_color=(0, 0, 255)):
        # Grayscale for edge detection

        for i, frame_bgr in enumerate(self.current_frame.frames):
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            # Optional blur to reduce noise
            gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
            
            # Canny edge detection
            edges = cv2.Canny(gray, 100, 200)
            
            # Start with the original frame
            output = frame_bgr.copy()
            
            # Overlay edges in color
            output[edges > 0] = edge_color
            
            # Find edge pixel coordinates (y, x)
            edge_points = np.column_stack(np.where(edges > 0))
            
            # Subsample to avoid drawing thousands of points
            if len(edge_points) > 0:
                sampled = edge_points[::sample_step]
                for (y, x) in sampled:
                    cv2.circle(output, (int(x), int(y)), point_radius, point_color, -1)
        
            self.current_frame.frames[i] = output

    def processing(self, frame, idx):
        self.update_frame(frame,idx)
        self.find_edges()
        return self.current_frame.frames