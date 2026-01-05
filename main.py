from dynamics import DoubleInvertedPendulum
from renderer import create_animation

from scipy.integrate import solve_ivp
import numpy as np


def simulate(dynamics_model, initial_state, t_span, dt=0.01):
        """Simulate the system"""
        t_eval = np.arange(t_span[0], t_span[1], dt)
        
        def event_boundary(t, state):
            return abs(state[0]) - dynamics_model.position_limit
        event_boundary.terminal = False
        event_boundary.direction = 0
        
        sol = solve_ivp(dynamics_model.dynamics, t_span, initial_state, 
                       t_eval=t_eval, method='RK45', 
                       rtol=1e-8, atol=1e-10)
        
        return sol.t, sol.y.T


if __name__ == "__main__":
    # Create pendulum system
    pendulum = DoubleInvertedPendulum()
    
    # Initial state: [pos, theta1, theta2, dpos, dtheta1, dtheta2]
    # Start with pendulum hanging down and give it a small push
    initial_state = np.array([0.0, 0.1, 0.1, 0.0, 0.1, 0.1])
    
    print("Simulating double inverted pendulum...")
    print("Position limits: ±{:.1f} m".format(pendulum.position_limit))
    
    # Simulate
    times, states = simulate(pendulum, initial_state, [0, 20], dt=0.01)
    
    print(f"Simulation completed: {len(times)} time steps")
    
    # Create and show animation
    fig, anim = create_animation(pendulum, times, states)
    
    # Print final state
    final_state = states[-1]
    print(f"\nFinal state:")
    print(f"Position: {final_state[0]:.3f} m")
    print(f"Angle 1: {final_state[1]:.3f} rad ({np.degrees(final_state[1]):.1f}°)")
    print(f"Angle 2: {final_state[2]:.3f} rad ({np.degrees(final_state[2]):.1f}°)")
    
    # Check if upright
    upright_threshold = 0.2  # radians
    if (abs(final_state[1] - np.pi) < upright_threshold and 
        abs(final_state[2] - np.pi) < upright_threshold):
        print("✓ Pendulum successfully stabilized in upright position!")
    else:
        print("✗ Pendulum not fully stabilized")
