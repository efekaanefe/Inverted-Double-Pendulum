import numpy as np
from inverted_pendulum_env import InvertedPendulumEnv
from constants import *
from agents import LQRAgent
import pymunk

DEBUG = True
dt = 1/200

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
        g = env.gravity     # Gravity
        l = env.link_size[1]  # Link half-length
        # base_size = env.base_size  # Base dimensions (w, h)
        # link_size = env.link_size  # Link dimensions (w, h)

        g /= 100
        l /= 1000
        # link_size = (link_size[0]/scale, link_size[1]/scale) 
        
        # State-space matrices
        denom1 = (4*m1 + m2)
        denom2 = l * (4*m1 + m2)
        
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, -(3*m2*g)/denom1, 0],
            [0, 0, 0, 1],
            [0, 0, 3*g*(m1 + m2)/denom2, 0]
        ])

        B = np.array([
            [0],
            [4 / denom1],
            [0],
            [-3 / denom2]
        ])
        
        # Define Q and R matrices
        Q = np.diag([100, 1, 10, 1])/1 # Penalize x, x_dot, theta, theta_dot
        R = np.diag([0.01])           # Penalize control effort

        return A, B, Q, R

    A, B, Q, R = compute_system_matrices(env)
    
    lqr_agent = LQRAgent(A, B, Q, R)

    print(lqr_agent.A)
    print(lqr_agent.B)
    print(lqr_agent.Q)
    print(lqr_agent.R)
    print(lqr_agent.K)

    obs_goal = np.array([env.groove_length/2, 0, 90, 0]) # x, xdot, theta, theta_dot
    obs, _ = env.reset()
    total_reward = 0

    env.base_body.apply_force_at_local_point((env.actuation_max*25, 0)) # initial push to the right
    env.link_body.apply_force_at_local_point((env.actuation_max*25, 0)) # initial push to the right

    for iter in range(int(max_steps * 100)):
        try:
            obs_error = np.mean(obs_goal - obs)
        except:
            break

        # disturbance
        if np.random.uniform(0, 1) < 0.00:
            force = np.random.uniform(-env.actuation_max, env.actuation_max) * 5
            # mag = 20
            # force = env.actuation_max*np.random.choice([-mag, mag])
            env.link_body.apply_force_at_local_point((force,0))

            print(f"Disturbance: {force}")
        
        error = obs_goal - obs

        scale = 1
        error[0] /= scale 
        # error[1] /= scale 
        # error[2] = np.deg2rad(error[2])
        # error[3] = np.deg2rad(error[3])

        action = 0
        action += lqr_agent.choose_action(error) 
        
        # action /= 100

        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        
        if DEBUG:
            print(
                # iter,
                action, 
                obs,
                # error,
                #total_reward,
                )

        env.render(manual_test = True)
        if done:
            obs = env.reset()

    env.close()
