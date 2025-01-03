import numpy as np
from inverted_pendulum_env import InvertedPendulumEnv
from constants import *
from agents import PIDAgent

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
            max_steps = max_steps * 6,
            actuation_max=actuation_max, # force or speed
            margin = margin,
            render_mode = "human",
            input_mode = "agent",
            control_type="stabilization"
            # control_type="swing-up"
            ) 

    thetapid = PIDAgent(P=100, I=0, D=100)

    obs_goal = np.array([env.groove_length/2, 0, 90, 0]) # x, xdot, theta, theta_dot

    obs, _ = env.reset()

    total_reward = 0

    env.step(50) # initial push to the right

    for iter in range(int(max_steps * 3)):
        try:
            obs_error = np.mean(obs_goal - obs)
        except:
            break
        
        error = obs_goal[2] - obs[2]

        action = thetapid.choose_action(error)
        # action = 0 if np.abs(action)< 50 else action

        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        
        if DEBUG:
            print(
                iter,
                action, 
                # obs[2],
                # error,
                #np.round(reward,2),
                #total_reward,
                # np.round(obs_error,2)
                )

        env.render(manual_test = True)
        if done:
            obs = env.reset()

    env.close()
