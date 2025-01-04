import numpy as np
from inverted_pendulum_env import InvertedPendulumEnv
from constants import *
from agents import PIDAgent

DEBUG = False
dt = 1/200

if __name__ == "__main__":
    env = InvertedPendulumEnv(
            gravity=-9.81*100, 
            dt=dt,
            base_size=base_size, 
            base_mass=0.3,
            link_size=link_size, 
            link_mass=0.1,
            groove_length = groove_length,
            max_steps = max_steps * 100,
            actuation_max=255, # force or speed
            margin = margin,
            render_mode = "human",
            input_mode = "agent",
            control_type="stabilization"
            # control_type="swing-up"
            ) 

    pid_theta = PIDAgent(P=100, I=0, D=100)
    pid_x = PIDAgent(P=0, I=0, D=0)

    obs_goal = np.array([env.groove_length/2, 0, 90, 0]) # x, xdot, theta, theta_dot

    obs, _ = env.reset()

    total_reward = 0
    
    env.base_body.apply_force_at_local_point((env.actuation_max*50, 0)) # initial push to the right

    for iter in range(int(max_steps * 100)):
        try:
            obs_error = np.mean(obs_goal - obs)
        except:
            break

        # disturbance
        if np.random.uniform(0, 1) < 0.05:
            force = np.random.uniform(-env.actuation_max, env.actuation_max)
            env.base_body.apply_force_at_local_point((force,0))

            print(f"Disturbance: {force}")
        
        error_theta = obs_goal[2] - obs[2]      # deg
        error_x = (obs_goal[0] - obs[0]) / 10  # cm

        action = pid_theta.choose_action(error_theta, dt) + pid_x.choose_action(error_x, dt)

        # action = 0

        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        
        if DEBUG:
            print(
                iter,
                action, 
                # obs[2],
                # obs[0],
                error_theta,
                error_x,actuation_max
                #np.round(reward,2),
                #total_reward,
                # np.round(obs_error,2)
                )

        env.render(manual_test = True)
        if done:
            obs = env.reset()

    env.close()
