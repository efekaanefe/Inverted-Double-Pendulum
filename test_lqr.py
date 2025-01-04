import numpy as np
from inverted_pendulum_env import InvertedPendulumEnv
from constants import *
from agents import LQRAgent
import pymunk

DEBUG = True
dt = 1/100

if __name__ == "__main__":
    env = InvertedPendulumEnv(
            gravity=gravity, 
            dt=dt,
            base_size=base_size, 
            base_mass=base_mass,
            link_size=link_size, 
            link_mass=link_mass,
            groove_length = groove_length,
            max_steps = max_steps * 100,
            actuation_max=255, # force or speed
            margin = margin,
            render_mode = "human",
            input_mode = "agent",
            control_type="stabilization"
            # control_type="swing-up"
            ) 

    def compute_system_matrices(env):
        # Extract environment variables
        m1 = env.base_mass  # Base (cart) mass
        m2 = env.link_mass  # Pendulum mass
        g = env.gravity  # Gravity
        l = env.link_size[1] / 2  # Link half-length
        base_size = env.base_size  # Base dimensions (w, h)
        link_size = env.link_size  # Link dimensions (w, h)

        # Calculate moments of inertia using pymunk
        I_link = pymunk.moment_for_box(m2, link_size)  # Moment of inertia for pendulum
        denom = I_link + m2 * (l**2)  # Effective denominator for linearization

        # State-space matrices
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, -(m2 * g * l) / denom, 0],
            [0, 0, 0, 1],
            [0, 0, g * (m1 + m2) / denom, 0]
        ])
        
        B = np.array([
            [0],
            [1 / denom],
            [0],
            [-l / denom]
        ])

        # Define Q and R matrices
        Q = np.diag([10, 1, 10000, 1])  # Penalize x, x_dot, theta, theta_dot
        R = np.diag([0.1])           # Penalize control effort

        return A, B, Q, R

    A, B, Q, R = compute_system_matrices(env)
    
    lqr_agent = LQRAgent(A, B, Q, R)

    obs_goal = np.array([env.groove_length/2, 0, 90, 0]) # x, xdot, theta, theta_dot
    obs, _ = env.reset()
    total_reward = 0
    
    env.base_body.apply_force_at_local_point((env.actuation_max*25, 0)) # initial push to the right

    for iter in range(int(max_steps * 100)):
        try:
            obs_error = np.mean(obs_goal - obs)
        except:
            break

        # disturbance
        if np.random.uniform(0, 1) < 0.00:
            force = np.random.uniform(-env.actuation_max, env.actuation_max)
            force = env.actuation_max*10
            env.base_body.apply_force_at_local_point((force,0))

            print(f"Disturbance: {force}")
        
        error = obs_goal - obs

        action = 0
        action += lqr_agent.choose_action(error) 

        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        
        if DEBUG:
            print(
                # iter,
                action, 
                # obs[2],
                # obs[0],
                error,
                #np.round(reward,2),
                #total_reward,
                # np.round(obs_error,2)
                )

        env.render(manual_test = True)
        if done:
            obs = env.reset()

    env.close()
