import numpy as np
from inverted_pendulum_env import InvertedPendulumEnv


if __name__ == "__main__":
    env = InvertedPendulumEnv(
        gravity=150, 
        dt=1/120.0, 
        force_mag=1000,                   
        base_size=(30, 30), 
        base_mass=5,
        link_size=(6, 250), 
        link_mass=2,
        max_angle=np.pi/3, 
        max_position=500)

    obs = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()  # Replace with a trained policy for better control
        obs, reward, done, info = env.step(action)
        print(action, obs)
        if done:
            break
    env.close()
