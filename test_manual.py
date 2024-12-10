import numpy as np
from inverted_pendulum_env import InvertedPendulumEnv
from constants import *


DEBUG = True

if __name__ == "__main__":

    env = InvertedPendulumEnv(
            gravity=gravity, 
            dt=dt,
            base_size=base_size, 
            base_mass=base_mass,
            link_size=link_size, 
            link_mass=link_mass,
            groove_length = groove_length,
            initial_angle=initial_angle,
            max_steps = max_steps * 3,
            actuation_max=actuation_max, # force or speed
            margin = margin,
            render_mode = "human",
            input_mode = "human"
            ) 

    env.render_scale = 0.5

    obs_goal = np.array([env.groove_length/2, 0, 90, 0]) # x, xdot, theta, theta_dot

    obs, _ = env.reset()

    total_reward = 0

    for iter in range(10000):
        obs_error = np.mean(obs_goal - obs)

        action = env.actuation_max
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        
        if DEBUG:
            print(
                iter,
                #action, 
                obs[3],
                #np.round(reward,2),
                #total_reward,
                # np.round(obs_error,2)
                )

        env.render(manual_test = True)
        if done:
            obs = env.reset()

    env.close()
