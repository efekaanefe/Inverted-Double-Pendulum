import numpy as np
from inverted_pendulum_env import InvertedPendulumEnv
from agents import PIDAgent

DEBUG = True

if __name__ == "__main__":
    env = InvertedPendulumEnv(
        gravity=150, 
        dt=1/30,
        force_mag=1000,                   
        base_size=(30, 30), 
        base_mass=5,
        link_size=(6, 250), 
        link_mass=2,
        groove_length = 600,
        max_angle=np.pi/3, 
        max_position=500,
        max_steps = 1000)

    obs_goal = np.array([env.groove_length/2, 0, np.pi, 0]) # x, xdot, theta, theta_dot

    obs = env.reset()

    P, I, D = 15, 5, 100 
    agent = PIDAgent(P, I, D)

    for _ in range(1000):
        obs_error = np.mean(obs_goal - obs)

        action = agent.choose_action(obs_error)  # Replace with a trained policy for better control
        obs, reward, done, info = env.step(action)
        
        if DEBUG:
            print(_, obs, np.round(reward,2), np.round(obs_error,2))

        if done:
            break


    env.close()
