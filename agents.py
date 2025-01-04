import numpy as np

class PIDAgent:
    def __init__(self, P, I, D):
        self.p = P
        self.i = I
        self.d = D
        self.prev_error = 0 # error for all the observations, shape -> (4,)
        self.integral = 0

    def choose_action(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error)/dt
        self.prev_error = error
        action = self.p * error + self.i * self.integral + self.d * derivative
        # action = sigmoid(pid).astype(np.int16)

        max_limit = 255; min_limit = -max_limit
        action = max(min(action, max_limit), min_limit)

        return action

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class LQRAgent:
    def __init__(self, A, B, Q, R):
        """
        Initialize the LQR Agent.
        :param A: State matrix (n x n)
        :param B: Input matrix (n x m)
        :param Q: State cost matrix (n x n)
        :param R: Input cost matrix (m x m)
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.K = self.compute_gain()  # Precompute the LQR gain matrix
    
    def compute_gain(self):
        """
        Compute the LQR gain matrix K by solving the Riccati equation.
        """
        from scipy.linalg import solve_continuous_are
        
        # Solve the continuous-time Algebraic Riccati Equation (ARE)
        P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        
        # Compute the optimal gain K
        K = np.linalg.inv(self.R) @ self.B.T @ P
        return K

    def choose_action(self, state):
        """
        Compute the control action based on the current state.
        :param state: Current state vector (n,)
        :return: Control action vector (m,)
        """
        # Compute the control action
        action = -self.K @ state

        # Clamp the action within specified limits
        max_limit = 255; min_limit = -max_limit
        # action = np.clip(action, min_limit, max_limit)

        return action
