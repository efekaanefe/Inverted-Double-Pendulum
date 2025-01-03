import numpy as np

class PIDAgent:
    def __init__(self, P, I, D):
        self.p = P
        self.i = I
        self.d = D
        self.prev_error = 0 # error for all the observations, shape -> (4,)
        self.integral = 0

    def choose_action(self, error):
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        action = self.p * error + self.i * self.integral + self.d * derivative
        # action = sigmoid(pid).astype(np.int16)

        max_limit = 255 * 3; min_limit = -max_limit
        action = max(min(action, max_limit), min_limit)

        return action

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
