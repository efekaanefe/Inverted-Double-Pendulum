from inverted_pendulum_env import InvertedPendulumEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback


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
        max_steps = 1500,
        render_mode="human")


    # Define and train the model
    model = PPO("MlpPolicy", env, verbose=1)
    model.load("logs//best_model")
    # Test the trained agent
    obs, _ = env.reset()
    for _ in range(5000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        env.render()
        
        print(action, obs)

        if done:
            obs, _ = env.reset()

    env.close()