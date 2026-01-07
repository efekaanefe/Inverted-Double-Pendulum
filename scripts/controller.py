import numpy as np
from scipy.optimize import minimize
import time
import casadi as ca


# --- CASADI PHYSICS (For MPC) ---
def get_casadi_func(dynamics, dt):
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


class MPCController:
    def __init__(self, model, dt, N, Q_weights, R_weight, max_force = 10.0):
        self.dt = dt
        self.N = N
        
        opti = ca.Opti()
        
        # Variables
        self.X = opti.variable(6, N+1)
        self.U = opti.variable(1, N)
        self.P_x0 = opti.parameter(6)
        
        # Dynamics Constraint
        self.f_discrete = get_casadi_func(model, dt)
        
        # Cost weights
        Q = ca.diag(ca.vertcat(*Q_weights))
        R = R_weight
        x_target = ca.vertcat(0, np.pi, np.pi, 0, 0, 0) # stabilize at stable position
        # x_target = ca.vertcat(0, 0, 0, 0, 0, 0)
        
        # Initial State Constraint
        opti.subject_to(self.X[:, 0] == self.P_x0)
        
        cost = 0
        for k in range(N):
            # Dynamics
            x_next = self.f_discrete(self.X[:, k], self.U[:, k])
            opti.subject_to(self.X[:, k+1] == x_next)
            
            # Constraints
            opti.subject_to(opti.bounded(-max_force, self.U[:, k], max_force))
            
            # Cost
            e = self.X[:, k] - x_target
            cost += ca.mtimes([e.T, Q, e]) + ca.mtimes([self.U[:, k].T, R, self.U[:, k]])
            
        opti.minimize(cost)
        
        p_opts = {'expand': True, 'print_time': False}
        s_opts = {
            'max_iter': 500,        
            'print_level': 0,
            'sb': 'yes',
            'tol': 1e-4,            
            'acceptable_tol': 1e-1, 
            'acceptable_iter': 50
        }
        opti.solver('ipopt', p_opts, s_opts)

        self.opti = opti
        
        self.is_initialized = False 

    def solve(self, current_state):
        start_time = time.time() 

        self.opti.set_value(self.P_x0, current_state)
        
        # --- Warm Start Logic ---
        if self.is_initialized:
            # We can only call value() if the previous solve succeeded!
            try:
                prev_x = self.opti.value(self.X)
                prev_u = self.opti.value(self.U)
                
                # Shift guess
                self.opti.set_initial(self.X[:, :-1], prev_x[:, 1:])
                self.opti.set_initial(self.X[:, -1], prev_x[:, -1])
                self.opti.set_initial(self.U[:, :-1], prev_u[1:])
                self.opti.set_initial(self.U[:, -1], prev_u[-1])
            except Exception:
                # If reading values fails, reset initialization
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
            elapsed_time = time.time() - start_time

            print(f"Elapsed time for controller : {elapsed_time*1000:.2f} ms | control force : {sol.value(self.U[:,0])}")

            return sol.value(self.U[:, 0])
        except RuntimeError as e:
            print(f"MPC Failure: {e}")
            return 0.0

    def __call__(self, state):
            return self.solve(state)
