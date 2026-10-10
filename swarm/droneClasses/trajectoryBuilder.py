from swarm.droneClasses.droneVisulaControler import DroneVisulaCTRL
import numpy as np


#z_0, z_d_0, z_dd_0, z3d_0, z_f, z_d_f, z_dd_f, z_3d_f
class Segment:
    def __init__(self, b_con: np.ndarray, T):
        self.z = b_con
        self.T = T
        self.M = self.build_M()
        self.c = self.solve()

    def build_M(self):
        T = self.T
        M = np.array([
            [1, 0, 0, 0, 0,       0,        0,         0        ],
            [0, 1, 0, 0, 0,       0,        0,         0        ],
            [0, 0, 2, 0, 0,       0,        0,         0        ],
            [0, 0, 0, 6, 0,       0,        0,         0        ],
            [1, T, T**2, T**3, T**4, T**5, T**6, T**7],
            [0, 1, 2*T,  3*T**2, 4*T**3, 5*T**4, 6*T**5, 7*T**6],
            [0, 0, 2,    6*T,    12*T**2, 20*T**3, 30*T**4, 42*T**5],
            [0, 0, 0,    6,      24*T,    60*T**2, 120*T**3, 210*T**4],
        ])
        return M


    def solve(self):
        return np.linalg.solve(self.M, self.z)


    def get_trajectory(self):
        return self.c



class MINCO:
    def __init__(self, w:np.ndarray):
        self.w = w

    def create_sub_trajectories(self):
        pass

    def interconect_trajectory(self):
        pass

    def discritize_trajectory(self):
        pass