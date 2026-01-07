import numpy as np

def load_data(filename = "data/sim_results.npz"):
    data = np.load(filename)
    print("Loaded data")
    return data['time'], data['states'], data['controls']


def check_success_at_state(state):
    def wrap_angle(angle): 
        return (angle + np.pi) % (2 * np.pi) - np.pi
    theta1_wrapped = wrap_angle(state[1])
    theta2_wrapped = wrap_angle(state[2])
    print(f"\nFinal state:")
    print(f"Position: {state[0]:.3f} m")
    print(f"Angle 1: {state[1]:.3f} rad ({np.degrees(state[1]):.1f}°)")
    print(f"Angle 2: {state[2]:.3f} rad ({np.degrees(state[2]):.1f}°)")
    upright_threshold = 0.2  # radians
    is_th1_up = abs(theta1_wrapped) < upright_threshold
    is_th2_up = abs(theta2_wrapped) < upright_threshold
    if is_th1_up and is_th2_up:
        print("✓ Pendulum successfully stabilized in upright position!")
        return True
    else:
        print("✗ Pendulum not fully stabilized")
        print(f"  (Errors: Th1={theta1_wrapped:.2f}, Th2={theta2_wrapped:.2f})")
        return False
