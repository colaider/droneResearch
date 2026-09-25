import cv2 
import numpy as np 
from collections import deque


class EnumFrame:
    def __init__(self):
        self.frames = []
        self.idx = 0


class VisulaAcEst:
    def __init__(self, res, fov):
        self.current_frame = EnumFrame() 
        self.buffer = []
        self.foc_l = (res[0]/2)/ np.tan(np.deg2rad(fov)/2)
       

        self.drone_pos = np.zeros(3)
        self.dt = 0.01
        self.drone_vel = np.zeros(3)#
        self.camera_saperation = 0
        self.drone_ang_vel = np.zeros(3)
        self.imu_att = np.zeros(3)
        self.drone_vel = np.zeros(3)
        self.drone_pos = np.zeros(3)

        self.expected_vel_err = 0
        self.previous_cmd_vel = np.zeros(4)
        self.camera_velocity = []

        self.vkf = VelocityKalmanFilter()


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
        points = []
        flow_lr = []
        
        if len(self.buffer) < 2:
            return flow   # need at least 2 frames
        
        for i in range(min(len(self.buffer[-2].frames), len(self.buffer[-1].frames))):
            frame1 = self.buffer[-2].frames[i]
            frame2 = self.buffer[-1].frames[i]
            fr, flow, p = self.lucas_kanade_flow(frame1, frame2)
            
            processed.append(fr)
            flow_lr.append(flow)
            points.append(p)

        points = self.trinagulate_altitude(points, flow_lr)

        self.current_frame.frames = processed
        self.estimate_velocities(flow_lr, points)
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

        if old_points is None: return annotated_frame, None, None
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

        if new_points is None: return annotated_frame, None, None

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
      
        
        for new in good_new:
            x2, y2 = new.astype(int)
            collor = (0, 100,200 ) if (x2-xs/2)**2 + (y2-ys/2)**2 < (diameter / 2)**2 else (0, 255, 0)
            cv2.circle(annotated_frame, (x2, y2), 3, collor , -1)

        central_good_old = np.hstack((central_good_old, np.full((central_good_old.shape[0], 1), self.drone_pos[2])))
        central_good_new = np.hstack((central_good_new, np.full((central_good_new.shape[0], 1), self.drone_pos[2])))


        resud_mask = self.point_prediction_filtering(central_good_old,central_good_new)
        central_flow = central_good_new[:, :2] - central_good_old[:, :2]
        avg_pos = (central_good_old + central_good_new) / 2

        avg_pos = avg_pos[resud_mask]
        central_flow = central_flow[resud_mask]
        return annotated_frame, central_flow, avg_pos


    def trinagulate_altitude(self, points, flow) -> list:
        #complete
        out = []
        for p,f in zip(points, flow):
            out.append(p)
        return out


    def estimate_velocities(self, flow, points):
        v_out = []
        i = 0
        nph = np.array([self.previous_cmd_vel[0], 
                                    self.previous_cmd_vel[1], 
                                    self.previous_cmd_vel[3]
                                    ])
        if not hasattr(self, 'sfkl'): 
            self.sfkp = [nph, nph]
            self.mean = deque(maxlen=1)

        if len(points[0]) < 3:
            self.camera_velocity = self.drone_vel
            print('hui')
            return None
        
        for f, p in zip(flow, points):
            if len(p) == 0: return None
            v_duerot = p[:, 2, None]*(1/np.cos(self.imu_att[:2])**2)*self.drone_ang_vel[:2]

            kv = (1/(self.foc_l * self.dt)) * p[:, 2]
            v = f*kv[:,None]
            B = v - v_duerot
            c = np.array(self.frame_size[:2])/2

            r = (p[:, 2, None] / self.foc_l) * (p[:, :2] - c) + self.camera_saperation / 2
            A = np.column_stack([np.ones(len(p)), np.zeros(len(p)), -r[:, 1]])
            A2 = np.column_stack([np.zeros(len(p)), np.ones(len(p)), r[:, 0]])
            A = np.vstack([A, A2])
            B = np.concatenate([B[:, 0], B[:, 1]])
            x, *_ = np.linalg.lstsq(A, B, rcond=None)
            x *= -1
            
            v_l = v - x[2]*r
         
            self.vkf.predict(self.sfkp[i])
            self.vkf.update(x)
            self.sfkp[i] = self.vkf.get()        
            self.mean.append(self.sfkp[i])

            v_out.append(np.mean(self.mean, axis=0))   
            i += 1

        vel = (v_out[0] + v_out[1]) / 2
        np.insert(vel, 2, self.drone_vel[2])
        self.camera_velocity = vel
           


    def point_prediction_filtering(self, old, new, threshold=1.5):
        v = self.previous_cmd_vel[:3]    # (3,) m/s
        ang = self.drone_ang_vel # (3,) rad/s
        c = np.array(self.frame_size[:2]) / 2

        u = old[:, 0] - c[0]  
        v_pix = old[:, 1] - c[1]  
        z = old[:, 2]          
                
        du_trans = (-self.foc_l * v[0] + u * v[2]) / z * self.dt
        
        dv_trans = (-self.foc_l * v[1] + v_pix * v[2]) / z * self.dt
        du_yaw = -v_pix * ang[2] * self.dt
        dv_yaw = u * ang[2] * self.dt
        predicted_u = old[:, 0] + du_trans + du_yaw
        predicted_v = old[:, 1] + dv_trans + dv_yaw
        predicted = np.column_stack([predicted_u, predicted_v])
        
        # Residuals in pixels
        residuals = np.clip(np.linalg.norm(new[:, :2] - predicted, axis=1), 0, 150)
        self.expected_vel_err = np.mean(residuals)
        if residuals.sum() > 0: threshold = np.min(residuals) + 0.12*(np.max(residuals) - np.min(residuals))
        return residuals < threshold


                               
    @staticmethod
    def add_noise(frames, sigma=20):
        out = []
        for frame in frames:
            noise = np.random.normal(0, sigma, frame.shape)
            noisy = frame.astype(np.float32) + noise
            out.append(np.clip(noisy, 0, 255).astype(np.uint8))
        return out


    def set_dt(self, dt):
        self.dt = dt



class VelocityKalmanFilter:
    def __init__(self, process_var=0.6, measurement_var=5):
        self.state = np.zeros(3)
        self.P = np.eye(3) * 1.0
        self.Q = np.eye(3) * process_var
        self.R = np.eye(3) * measurement_var
    
    def predict(self, commanded_velocity):
        """Prediction = commanded velocity (with some uncertainty)."""
        self.state = commanded_velocity
        self.P = self.P + self.Q  # uncertainty grows
    
    def update(self, measured_velocity):
        """Correct with measurement."""
        K = self.P @ np.linalg.inv(self.P + self.R)
        self.state = self.state + K @ (measured_velocity - self.state)
        self.P = (np.eye(3) - K) @ self.P
    
    def get(self):
        return self.state.copy() 