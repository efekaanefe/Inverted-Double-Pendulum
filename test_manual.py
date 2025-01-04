import numpy as np
from inverted_pendulum_env import InvertedPendulumEnv
from constants import *


DEBUG = True
LOG = False
SAVE = False

if __name__ == "__main__":
    env = InvertedPendulumEnv(
            gravity=gravity, 
            dt=dt,
            base_size=base_size, 
            base_mass=base_mass,
            link_size=link_size, 
            link_mass=link_mass,
            groove_length = groove_length,
            max_steps = max_steps * 10,
            actuation_max=actuation_max, # force or speed
            margin = margin,
            render_mode = "human",
            input_mode = "human",
            # control_type="stabilization"
            control_type="swing-up"
            ) 

    env.render_scale = 0.5

    obs_goal = np.array([env.groove_length/2, 0, 90, 0]) # x, xdot, theta, theta_dot

    obs, _ = env.reset()

    total_reward = 0

    if LOG:
        import matplotlib.pyplot as plt
        obs_log = []
        force_log = []

    for iter in range(int(max_steps * 3)):
        try:
            obs_error = np.mean(obs_goal - obs)
        except:
            break

        action = env.actuation_max
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        

        if LOG:

            obs_log.append([
                obs[0]/100,        # m
                obs[1]/100,        # m/s
                obs[2],             # deg
                np.rad2deg(obs[3])]) # deg/s
            try:
              force_log.append(info["force"]/100) # N
            except:
                break

        if DEBUG:
            print(
                iter,
                #action, 
                obs[2],
                #np.round(reward,2),
                #total_reward,
                # np.round(obs_error,2)
                )

        env.render(manual_test = True)
        if done:
            obs = env.reset()

    env.close()

    if LOG:
        fig, ax = plt.subplots(2, 1, figsize=(8, 6)) 

        ax[0].plot(obs_log, label='obs_log')
        ax[0].set_title('Observation Log')
        ax[0].set_ylabel('Observation Value')
        ax[0].legend(["x [m]", "x_dot [m/s]", "theta [deg]", "theta_dot [deg/s]"])

        ax[1].plot(force_log, label='force_log [N]', color='orange')
        ax[1].set_title('Force Log')
        ax[1].set_xlabel('Time Steps')
        ax[1].set_ylabel('Force Value')
        ax[1].legend()

        plt.tight_layout()

        if SAVE:
            np.save('data\\obs_log.npy', obs_log)
            np.save('data\\force_log.npy', force_log)

            print("Arrays saved as .npy files.")

            # Save the figure
            fig.savefig("data\\logs_plot.png", dpi=300, bbox_inches='tight')  # Save with high resolution
            print("Figure is saved.")


        plt.show()
