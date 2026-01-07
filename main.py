from scripts.dynamics import DoubleInvertedPendulum
from scripts.controller import MPCController
from scripts.renderer import create_animation
from scripts.utils import load_data, check_success_at_state

from scipy.integrate import solve_ivp
import numpy as np
import time


def simulate(dynamics_model, controller, initial_state, t_span, dt_control, save_results = True, filename="data/sim_results.npz"):
    """
    Simulates the system combining discrete control with continuous physics.
    
    Args:
        dynamics_model: The physics class (DoubleInvertedPendulum)
        controller: The MPC controller (must have .solve() or __call__)
        initial_state: Starting state array
        t_span: Tuple (t_start, t_end)
        dt_control: Time step for the controller (physics is adaptive)
        save_results (bool): If True, saves data to .npz file
        filename (str): Name of the file to save to
    """
    t_current = t_span[0]
    t_end = t_span[1]

    t_history = [t_current]
    state_history = [initial_state]
    control_history = [] 

    current_state = initial_state

    print(f"Simulating from {t_current}s to {t_end}s...")
    print(f"Controller dt: {dt_control}s | Physics: RK45 (Adaptive)")

    while t_current < t_end:
        # 1. CONTROLLER STEP (Discrete)
        # We hold this u constant for the duration of dt_control
        u = controller(current_state)
        
        control_history.append(u)

        # Determine the end of this control step
        t_next = min(t_current + dt_control, t_end)

        # 2. PHYSICS STEP (Continuous/Adaptive)
        # We solve the ODE from t_current to t_next
        def continuous_dynamics(t, y):
            return dynamics_model.dynamics(t, y, u)

        sol = solve_ivp(
            continuous_dynamics, 
            [t_current, t_next], 
            current_state, 
            method='RK45',   
            rtol=1e-8, 
            atol=1e-10
        )

        current_state = sol.y[:, -1]
        t_current = t_next

        t_history.append(t_current)
        state_history.append(current_state)

    print(f"Simulation completed: {len(t_history)} time steps")

    t_arr = np.array(t_history)
    x_arr = np.array(state_history)
    u_arr = np.array(control_history)

    if save_results:
        print(f"Saving simulation data to '{filename}'...")
        np.savez(filename, time=t_arr, states=x_arr, controls=u_arr)
        print("Save complete.")

    return t_arr, x_arr, u_arr


SIMULATE_NEW = True


if __name__ == "__main__":
    pendulum = DoubleInvertedPendulum()

    if SIMULATE_NEW:
        initial_state = np.array([0.0, np.deg2rad(0), np.deg2rad(0), -0.3, 0.1, 0.1]) # state: [pos, theta1, theta2, dpos, dtheta1, dtheta2]

        Q = [100, 100, 100, 1, 1, 1]
        R = 0.1
        N = 100
        dt_control = 0.01
        controller = MPCController(pendulum, dt_control, N, Q, R, max_force=50.0)

        times, states, controls = simulate(pendulum, controller, initial_state, [0, 10.02], dt_control=dt_control, save_results = True)
    else:
        times, states, controls = load_data()

    fig, anim = create_animation(pendulum, times, states, controls)
    check_success_at_state(states[-1])

