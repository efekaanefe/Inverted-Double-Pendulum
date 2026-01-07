from scripts.dynamics import DoubleInvertedPendulum
from scripts.renderer import create_animation
from scripts.utils import load_data, check_success_at_state


if __name__=="__main__":
    filename = "data/stabilize_downwards.npz"

    pendulum = DoubleInvertedPendulum()
    times, states, controls = load_data(filename)
    fig, anim = create_animation(pendulum, times, states, controls)
    check_success_at_state(states[-1])

