from swarm.droneClasses.droneVisulaControler import DroneVisulaCTRL
import numpy as np
import math


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



import math
import numpy as np

class MINCO:
    def __init__(self, waypoints: np.ndarray, durations: np.ndarray):
        self.waypoints = np.asarray(waypoints, float)   # (M+1, D)
        self.durations = np.asarray(durations, float)   # (M,)
        self.M = len(self.durations)
        self.D = self.waypoints.shape[1]
        self.segments = [[None] * self.M for _ in range(self.D)]

    # ---------------- build pipeline ----------------

    def create_sub_trajectories(self):
        """For each axis: solve junction derivatives, then build segments."""
        for d in range(self.D):
            q_axis = self.waypoints[:, d]
            xi = self.inner_tridiagonal_solver(q_axis)       # (3*(M-1),)
            for i in range(self.M):
                b = self._assemble_boundary(i, q_axis, xi)   # 8-vector per segment
                self.segments[d][i] = Segment(b, self.durations[i])

    def _assemble_boundary(self, i: int, q_axis: np.ndarray, xi: np.ndarray) -> np.ndarray:
        """Build the 8 boundary values for segment i from waypoints + solved ξ."""
        b = np.zeros(8)
        b[0] = q_axis[i]       # start position (waypoint)
        b[4] = q_axis[i + 1]   # end   position (waypoint)

        # left junction (i): zero at start of trajectory, otherwise from ξ
        if i > 0:
            v, a, j = xi[3 * (i - 1): 3 * (i - 1) + 3]
            b[1], b[2], b[3] = v, a, j

        # right junction (i+1): zero at end, otherwise from ξ
        if i < self.M - 1:
            v, a, j = xi[3 * i: 3 * i + 3]
            b[5], b[6], b[7] = v, a, j
        return b

    # ---------------- cost ----------------

    def calculate_control_effort(self):
        J = 0.0
        for d in range(self.D):
            for i in range(self.M):
                seg = self.segments[d][i]
                Q = self._Q_snap(seg.T)
                J += float(seg.c @ Q @ seg.c)
        return J

    # ---------------- inner solve ----------------

    def inner_tridiagonal_solver(self, q_axis: np.ndarray) -> np.ndarray:
        """
        One axis. Solve for ξ* = (v_1, a_1, j_1, v_2, a_2, j_2, …, v_{M-1}, a_{M-1}, j_{M-1})
        that minimises Σ_i cᵢᵀ Q(Tᵢ) cᵢ, with rest-to-rest boundaries at the ends.

        Each segment cost is  bᵢᵀ Kᵢ bᵢ   where  Kᵢ = M(Tᵢ)⁻ᵀ Q(Tᵢ) M(Tᵢ)⁻¹.
        bᵢ = Lᵢ ξ + mᵢ  (Lᵢ picks which ξ entries land in slots 1-3 and 5-7, mᵢ
        holds the fixed waypoint positions in slots 0 and 4).
        
        Total cost:  J(ξ) = ξᵀ H ξ + 2 gᵀ ξ + const
        with  H = Σᵢ LᵢᵀKᵢLᵢ,   g = Σᵢ LᵢᵀKᵢmᵢ.
        Minimum:  ξ* = -H⁻¹ g.
        """
        M = self.M
        N = 3 * (M - 1)                  # number of free junction derivatives
        H = np.zeros((N, N))
        g = np.zeros(N)

        for i in range(M):
            T_i = self.durations[i]
            Mi = self._build_M(T_i)
            Qi = self._Q_snap(T_i)
            Mi_inv = np.linalg.inv(Mi)
            Ki = Mi_inv.T @ Qi @ Mi_inv  # 8×8

            # Selection matrix L_i (8 × N) and fixed waypoint vector m_i (8,)
            Li = np.zeros((8, N))
            mi = np.zeros(8)
            mi[0] = q_axis[i]
            mi[4] = q_axis[i + 1]

            if i > 0:                     # left junction derivatives from ξ
                base = 3 * (i - 1)
                Li[1, base + 0] = 1       # v_i
                Li[2, base + 1] = 1       # a_i
                Li[3, base + 2] = 1       # j_i

            if i < M - 1:                 # right junction derivatives from ξ
                base = 3 * i
                Li[5, base + 0] = 1       # v_{i+1}
                Li[6, base + 1] = 1       # a_{i+1}
                Li[7, base + 2] = 1       # j_{i+1}

            H += Li.T @ Ki @ Li
            g += Li.T @ Ki @ mi

        return np.linalg.solve(H, -g)

    # ---------------- shared matrix helpers ----------------

    @staticmethod
    def _build_M(T: float) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0, 0,       0,        0,         0        ],
            [0, 1, 0, 0, 0,       0,        0,         0        ],
            [0, 0, 2, 0, 0,       0,        0,         0        ],
            [0, 0, 0, 6, 0,       0,        0,         0        ],
            [1, T, T**2, T**3, T**4, T**5, T**6, T**7],
            [0, 1, 2*T,  3*T**2, 4*T**3, 5*T**4, 6*T**5, 7*T**6],
            [0, 0, 2,    6*T,    12*T**2, 20*T**3, 30*T**4, 42*T**5],
            [0, 0, 0,    6,      24*T,    60*T**2, 120*T**3, 210*T**4],
        ])

    @staticmethod
    def _Q_snap(T: float) -> np.ndarray:
        Q = np.zeros((8, 8))
        for i in range(4, 8):
            for j in range(4, 8):
                ci = math.factorial(i) // math.factorial(i - 4)
                cj = math.factorial(j) // math.factorial(j - 4)
                k = i + j - 7
                Q[i, j] = ci * cj * T**k / k
        return Q