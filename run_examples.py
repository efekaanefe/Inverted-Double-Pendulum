from scripts.dynamics import DoubleInvertedPendulum
from scripts.renderer import create_animation
from scripts.utils import load_data, check_success_at_state


if __name__=="__main__":
    filename = "stabilize_downwards"
    filename = "stabilize_upwards"
    filename = "stabilize_upwards_inverted"
    filename_load = f"data/{filename}.npz"
    filename_gif = f"gifs/{filename}.gif"

    pendulum = DoubleInvertedPendulum()
    times, states, controls = load_data(filename_load)
    fig, anim = create_animation(pendulum, times, states, controls, filename=filename_gif)
    check_success_at_state(states[-1])

