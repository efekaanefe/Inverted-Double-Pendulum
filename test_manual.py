import numpy as np
from inverted_pendulum_env import InvertedPendulumEnv

DEBUG = True

if __name__ == "__main__":

    env = InvertedPendulumEnv(
        gravity=-9.81*10*10, 
        dt=1/75,
        base_size=(30, 30), 
        base_mass=5,
        link_size=(5, 400), 
        link_mass=0.5,
        groove_length = 1500,
        initial_angle=270,
        max_steps = 5000,
        actuation_max=10000,
        render_mode = "human",
        input_mode = "human") # force or speed

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
