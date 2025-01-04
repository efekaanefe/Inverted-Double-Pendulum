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
        m1 = M = env.base_mass  # Cart mass
        m2 = m = env.link_mass  # Pendulum mass
        g = np.abs(env.gravity)
        w, l = env.link_size

        w /= 1000
        l /= 1000
        g /= 100

        I = pymunk.moment_for_box(m2, (w, l))
        b = 0

        l = l/2
        I = (I + m * l**2)
        print(I)

        # M = .5;
        # m = 0.2;
        b = 0.0;
        # I = 0.006;
        # g = 9.8;
        # l = 0.3;

        # State-space matrices

        # denom1 = (4*m1 + m2)
        # denom2 = l * (4*m1 + m2)
        # A = np.array([
        #     [0, 1, 0, 0],
        #     [0, 0, -(3*m2*g)/denom1, 0],
        #     [0, 0, 0, 1],
        #     [0, 0, 3*g*(m1 + m2)/denom2, 0]
        # ])
        # B = np.array([
        #     [0],
        #     [4 / denom1],
        #     [0],
        #     [-3 / denom2]
        # ])

        p = I * (M + m) + M * m * l**2

        A = np.array([
            [0, 1, 0, 0],
            [0, -(I + m * l**2) * b / p, (m**2 * g * l**2) / p, 0],
            [0, 0, 0, 1],
            [0, -(m * l * b) / p, m * g * l * (M + m) / p, 0]
        ])

        # Define the B matrix
        B = np.array([
            [0],
            [(I + m * l**2) / p],
            [0],
            [m * l / p]
        ])
            
        # Define Q and R matrices
        Q = np.diag([5000, 0, 100, 0]) # Penalize x, x_dot, theta, theta_dot
        R = np.diag([1])           # Penalize control effort

        return A, B, Q, R

    A, B, Q, R = compute_system_matrices(env)
    
    lqr_agent = LQRAgent(A, B, Q, R)

    # print(lqr_agent.A)
    # print(lqr_agent.B)
    # print(lqr_agent.Q)
    # print(lqr_agent.R)
    print(lqr_agent.K)

    obs_goal = np.array([env.groove_length/2, 0, 90, 0]) # x, xdot, theta, theta_dot
    obs, _ = env.reset()

    total_reward = 0

    # env.base_body.apply_force_at_local_point((env.actuation_max, 0)) # initial push to the right
    env.link_body.apply_force_at_local_point((env.actuation_max*0.5, 0)) # initial push to the right

    for iter in range(int(max_steps * 100)):
        # disturbance
        if np.random.uniform(0, 1) < 0.00:
            force = np.random.uniform(-env.actuation_max, env.actuation_max) * 5
            # mag = 20
            # force = env.actuation_max*np.random.choice([-mag, mag])
            env.link_body.apply_force_at_local_point((force,0))

            print(f"Disturbance: {force}")
        
        error = obs_goal - obs
        
        scale = 1000
        error[0] /= scale 
        error[1] /= scale 
        # error[2] = np.deg2rad(error[2]) 
        # error[3] = np.deg2rad(error[3]) 
        
        error[2] *= -1
        error[3] *= -1

        action = lqr_agent.choose_action(error)
        # action = 0

        # action /= 100

        obs, reward, done, _, info = env.step(action)
        total_reward += reward

        if DEBUG:
            print(
                # iter,
                action, 
                # obs,
                error,
                # lqr_agent.K,
                #total_reward,
                )

        env.render(manual_test = True)
        if done:
            obs = env.reset()

    env.close()
