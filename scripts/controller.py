import numpy as np
from scipy.optimize import minimize
import time
import casadi as ca


# --- CASADI PHYSICS (For MPC) ---
def get_dynamics_symbolic(dynamics, dt):
    """
    Creates the CasADi function using SX symbols.
    SX is required to avoid 'LinsolQr' errors with matrix inversion.
    """
    # 1. Use SX (Scalar Expression) symbols
    x = ca.SX.sym('x', 6)
    u = ca.SX.sym('u', 1)
    
    pos, theta1, theta2 = x[0], x[1], x[2]
    dpos, dtheta1, dtheta2 = x[3], x[4], x[5]
    
    # 2. Math definitions
    c1 = ca.cos(theta1)
    c2 = ca.cos(theta2)
    c12 = ca.cos(theta1 - theta2)
    s1 = ca.sin(theta1)
    s2 = ca.sin(theta2)
    s12 = ca.sin(theta1 - theta2)
    
    # 3. Mass Matrix (SX)
    M = ca.SX(3, 3)
    M[0,0] = dynamics.h1
    M[0,1] = dynamics.h2 * c1
    M[0,2] = dynamics.h3 * c2
    M[1,0] = dynamics.h2 * c1
    M[1,1] = dynamics.h4
    M[1,2] = dynamics.h5 * c12
    M[2,0] = dynamics.h3 * c2
    M[2,1] = dynamics.h5 * c12
    M[2,2] = dynamics.h6
    
    # 4. Coriolis Vector (SX)
    C_vec = ca.vertcat(
        dynamics.h2 * dtheta1**2 * s1 + dynamics.h3 * dtheta2**2 * s2 + u,
        dynamics.h7 * s1 - dynamics.h5 * dtheta2**2 * s12,
        dynamics.h5 * dtheta1**2 * s12 + dynamics.h8 * s2
    )
    
    # 5. Symbolic Solve 
    acc = ca.solve(M, C_vec)
    
    x_dot = ca.vertcat(dpos, dtheta1, dtheta2, acc[0], acc[1], acc[2])
    x_next = x + x_dot * dt
    
    return ca.Function('F', [x, u], [x_next], ['x', 'u'], ['x_next'])


def get_energy_symbolic(model, x_state):
    """
    Calculates Kinetic (T) and Potential (V) energy using CasADi operations.
    x_state = [pos, theta1, theta2, dpos, dtheta1, dtheta2]
    """
    pos, theta1, theta2 = x_state[0], x_state[1], x_state[2]
    dpos, dtheta1, dtheta2 = x_state[3], x_state[4], x_state[5]

    # --- Kinetic Energy (T) ---
    # Cart
    T_cart = 0.5 * model.m0 * dpos**2
    
    # Rod 1 (Tip position kinematics)
    # v1_x = dpos + l1 * dtheta1 * cos(theta1)
    # v1_y = l1 * dtheta1 * sin(theta1)
    v1x = dpos + model.l1 * dtheta1 * ca.cos(theta1)
    v1y = model.l1 * dtheta1 * ca.sin(theta1)
    v1_sq = v1x**2 + v1y**2
    T_rod1 = 0.5 * model.m1 * v1_sq + 0.5 * model.J1 * dtheta1**2
    
    # Rod 2
    # The kinematics chain is longer here
    # Elbow velocity (end of rod 1)
    v_elbow_x = dpos + model.L1 * dtheta1 * ca.cos(theta1)
    v_elbow_y = model.L1 * dtheta1 * ca.sin(theta1)
    
    # Center of mass of rod 2 (relative to elbow)
    v_rel_x = model.l2 * dtheta2 * ca.cos(theta2)
    v_rel_y = model.l2 * dtheta2 * ca.sin(theta2)
    
    # Total velocity of mass 2
    v2x = v_elbow_x + v_rel_x
    v2y = v_elbow_y + v_rel_y
    v2_sq = v2x**2 + v2y**2
    
    T_rod2 = 0.5 * model.m2 * v2_sq + 0.5 * model.J2 * dtheta2**2
    
    T_total = T_cart + T_rod1 + T_rod2

    # --- Potential Energy (V) ---
    # Reference is 0 at the pivot height. Up is positive Y.
    h1 = model.l1 * ca.cos(theta1)
    h2 = model.L1 * ca.cos(theta1) + model.l2 * ca.cos(theta2)
    
    V_rod1 = model.m1 * model.g * h1
    V_rod2 = model.m2 * model.g * h2
    
    V_total = V_rod1 + V_rod2
    
    return T_total, V_total


class MPCController:
    def __init__(self, model, dt, N, weight_V, weight_T, weight_pos, weight_u, max_force=15.0):
        """
        Args:
            weight_E: Cost weight for Energy difference (Swing-up)
            weight_pos: Cost weight for Cart Position (Stabilization)
            weight_u: Cost weight for Control Effort
        """
        self.dt = dt
        self.N = N
        
        opti = ca.Opti()
        
        # Variables
        self.X = opti.variable(6, N+1)
        self.U = opti.variable(1, N)
        self.P_x0 = opti.parameter(6)
        
        self.f_discrete = get_dynamics_symbolic(model, dt)
        
        # --- 1. Compute Target Energy (Upright Static) ---
        # State: [0, 0, 0, 0, 0, 0] is Upright & Static
        x_upright = ca.DM([0, 0, 0, 0, 0, 0])
        _, V_target = get_energy_symbolic(model, x_upright)
        
        # Initial State Constraint
        opti.subject_to(self.X[:, 0] == self.P_x0)
        
        cost = 0
        for k in range(N):
            # Dynamics
            x_next = self.f_discrete(self.X[:, k], self.U[:, k])
            opti.subject_to(self.X[:, k+1] == x_next)
            
            # Constraints
            opti.subject_to(opti.bounded(-max_force, self.U[:, k], max_force))
            opti.subject_to(opti.bounded(-2.0, self.X[0, k], 2.0)) # Cart limits
            
            # --- 2. Split Energy Cost Function ---
            T_k, V_k = get_energy_symbolic(model, self.X[:, k])
            
            # Term 1: Maximize Potential Energy 
            # We minimize the squared difference from the maximum possible potential (V_target)
            cost += weight_V * (V_k - V_target)**2
            
            # Term 2: Minimize Kinetic Energy
            # We penalize T_k directly (target is 0)
            cost += weight_T * T_k
            
            # Term 3: Cart Stabilization
            cost += weight_pos * self.X[0, k]**2
            
            # Term 4: Control Effort
            cost += weight_u * self.U[:, k]**2
            
        opti.minimize(cost)
        # Solver Options (Relaxed for energy shaping)
        p_opts = {'expand': True, 'print_time': False}
        s_opts = {
            'max_iter': 1000,       
            'print_level': 0,
            'sb': 'yes',
            'tol': 1e-3,            
            'acceptable_tol': 1e-1, 
            'acceptable_iter': 15
        }
        opti.solver('ipopt', p_opts, s_opts)

        self.opti = opti
        self.is_initialized = False 

    def solve(self, current_state):
        start_time = time.time() 

        self.opti.set_value(self.P_x0, current_state)
        
        # --- Warm Start Logic (Try/Except) ---
        if self.is_initialized:
            try:
                # Get previous optimal solution
                prev_x = self.opti.value(self.X)
                prev_u = self.opti.value(self.U)
                
                # Shift guess forward
                self.opti.set_initial(self.X[:, :-1], prev_x[:, 1:])
                self.opti.set_initial(self.X[:, -1], prev_x[:, -1])
                self.opti.set_initial(self.U[:, :-1], prev_u[1:])
                self.opti.set_initial(self.U[:, -1], prev_u[-1])
            except Exception:
                # Fallback if reading values fails
                self.is_initialized = False
                self.opti.set_initial(self.X, ca.repmat(current_state, 1, self.N+1))
                self.opti.set_initial(self.U, 0.0)
        else:
            # Cold Start
            self.opti.set_initial(self.X, ca.repmat(current_state, 1, self.N+1))
            self.opti.set_initial(self.U, 0.0)
        
        # --- Solve ---
        try:
            sol = self.opti.solve()
            self.is_initialized = True
            
            u_opt = sol.value(self.U[:, 0])
            elapsed_time = (time.time() - start_time) * 1000
            
            print(f"MPC: {elapsed_time:.2f} ms | U: {u_opt:.2f}")
            return u_opt
            
        except RuntimeError as e:
            print(f"MPC Failure: {e}")
            return 0.0

    def __call__(self, state):
        return self.solve(state)
