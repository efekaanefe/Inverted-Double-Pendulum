import numpy as np
from inverted_pendulum_env import InvertedPendulumEnv
from agents import PIDAgent

DEBUG = True

if __name__ == "__main__":
    env = InvertedPendulumEnv(
        gravity=-98.1*2, 
        dt=1/75,
        force_mag=1500*3,                   
        base_size=(30, 30), 
        base_mass=5,
        link_size=(4, 250), 
        link_mass=1,
        groove_length = 600,
        initial_angle=270,
        max_steps = 5000)

    obs_goal = np.array([env.groove_length/2, 0, 90, 0]) # x, xdot, theta, theta_dot

    obs, _ = env.reset()

    P, I, D = 15, 5, 100 
    agent = PIDAgent(P, I, D)

    total_reward = 0

    for iter in range(10000):
        obs_error = np.mean(obs_goal - obs)

        action = agent.choose_action(obs_error)  # Replace with a trained policy for better control
        # action = env.action_space.sample()
        action = 1
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        
        if DEBUG:
            print(
                iter,
                # action, 
                # obs,
                np.round(reward,2),
                total_reward,
                # np.round(obs_error,2)
                )

        env.render()
        if done:
            obs = env.reset()

    env.close()
