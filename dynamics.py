import numpy as np


class DoubleInvertedPendulum:
    def __init__(self):
        # Physical parameters
        self.m0 = 0.6   # kg, mass of the cart
        self.m1 = 0.2   # kg, mass of the first rod
        self.m2 = 0.2   # kg, mass of the second rod
        self.L1 = 0.5   # m, length of the first rod
        self.L2 = 0.5   # m, length of the second rod
        self.g = 9.81   # m/s^2, gravity
        
        # Derived parameters
        self.l1 = self.L1 / 2  # center of mass position
        self.l2 = self.L2 / 2
        self.J1 = (self.m1 * self.l1**2) / 3  # moment of inertia
        self.J2 = (self.m2 * self.l2**2) / 3
        
        # Pre-computed constants for efficiency
        self.h1 = self.m0 + self.m1 + self.m2
        self.h2 = self.m1 * self.l1 + self.m2 * self.L1
        self.h3 = self.m2 * self.l2
        self.h4 = self.m1 * self.l1**2 + self.m2 * self.L1**2 + self.J1
        self.h5 = self.m2 * self.l2 * self.L1
        self.h6 = self.m2 * self.l2**2 + self.J2
        self.h7 = (self.m1 * self.l1 + self.m2 * self.L1) * self.g
        self.h8 = self.m2 * self.l2 * self.g

        self.position_limit = 2.0 # to stop at boundaries
        
        # State history for visualization
        self.time_history = []
        self.state_history = []
        
    def mass_matrix(self, theta1, theta2):
        """Construct the mass matrix M"""
        c1 = np.cos(theta1)
        c2 = np.cos(theta2)
        c12 = np.cos(theta1 - theta2)
        
        M = np.array([
            [self.h1, self.h2 * c1, self.h3 * c2],
            [self.h2 * c1, self.h4, self.h5 * c12],
            [self.h3 * c2, self.h5 * c12, self.h6]
        ])
        return M
    
    def coriolis_gravity_vector(self, theta1, theta2, dtheta1, dtheta2, u):
        """Construct the Coriolis and gravity vector"""
        s1 = np.sin(theta1)
        s2 = np.sin(theta2)
        s12 = np.sin(theta1 - theta2)
        
        C = np.array([
            self.h2 * dtheta1**2 * s1 + self.h3 * dtheta2**2 * s2 + u,
            self.h7 * s1 - self.h5 * dtheta2**2 * s12,
            self.h5 * dtheta1**2 * s12 + self.h8 * s2
        ])
        return C
    
    def energy(self, state):
        """Calculate total energy of the system"""
        pos, theta1, theta2, dpos, dtheta1, dtheta2 = state
        
        # Kinetic energy
        E_kin_cart = 0.5 * self.m0 * dpos**2
        
        E_kin_p1 = 0.5 * self.m1 * (
            (dpos + self.l1 * dtheta1 * np.cos(theta1))**2 +
            (self.l1 * dtheta1 * np.sin(theta1))**2
        ) + 0.5 * self.J1 * dtheta1**2
        
        E_kin_p2 = 0.5 * self.m2 * (
            (dpos + self.L1 * dtheta1 * np.cos(theta1) + 
             self.l2 * dtheta2 * np.cos(theta2))**2 +
            (self.L1 * dtheta1 * np.sin(theta1) + 
             self.l2 * dtheta2 * np.sin(theta2))**2
        ) + 0.5 * self.J2 * dtheta2**2
        
        E_kin = E_kin_cart + E_kin_p1 + E_kin_p2
        
        # Potential energy
        E_pot = (self.m1 * self.g * self.l1 * np.cos(theta1) + 
                 self.m2 * self.g * (self.L1 * np.cos(theta1) + 
                                     self.l2 * np.cos(theta2)))
        
        return E_kin, E_pot
    
    def dynamics(self, t, state, u = 0):
        """System dynamics: state = [pos, theta1, theta2, dpos, dtheta1, dtheta2], u is the control force (scalar in x)"""
        pos, theta1, theta2, dpos, dtheta1, dtheta2 = state
        
        # TODO: kinetic energy constantly increases if I set u = some non-zero constant

        # Position limits (hard constraints)
        if abs(pos) >= self.position_limit:
            if pos > 0 and dpos > 0:
                dpos = 0  # Stop at boundary
            elif pos < 0 and dpos < 0:
                dpos = 0
        
        # Construct mass matrix and force vector
        M = self.mass_matrix(theta1, theta2)
        C = self.coriolis_gravity_vector(theta1, theta2, dtheta1, dtheta2, u)
        
        # Solve for accelerations: M * [ddpos, ddtheta1, ddtheta2]^T = C
        accelerations = np.linalg.solve(M, C)
        
        ddpos, ddtheta1, ddtheta2 = accelerations
        
        # Return state derivative
        return np.array([dpos, dtheta1, dtheta2, ddpos, ddtheta1, ddtheta2])
    
    def get_pendulum_positions(self, state):
        """Get x,y positions of pendulum joints and masses"""
        pos, theta1, theta2, _, _, _ = state
        
        # Cart position
        cart_pos = np.array([pos, 0])
        
        # First pendulum joint (base of first rod)
        joint1_pos = cart_pos
        
        # First pendulum mass (center of first rod)
        mass1_pos = joint1_pos + np.array([self.l1 * np.sin(theta1), 
                                          self.l1 * np.cos(theta1)])
        
        # Second pendulum joint (top of first rod)
        joint2_pos = joint1_pos + np.array([self.L1 * np.sin(theta1), 
                                           self.L1 * np.cos(theta1)])
        
        # Second pendulum mass (center of second rod)
        mass2_pos = joint2_pos + np.array([self.l2 * np.sin(theta2), 
                                          self.l2 * np.cos(theta2)])
        
        # End of second rod
        end_pos = joint2_pos + np.array([self.L2 * np.sin(theta2), 
                                        self.L2 * np.cos(theta2)])
        
        return cart_pos, joint1_pos, mass1_pos, joint2_pos, mass2_pos, end_pos
    
    



