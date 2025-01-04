from inverted_pendulum_env import InvertedPendulumEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback


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
        max_steps = max_steps,
        actuation_max=255, # force or speed
        margin = margin,
        render_mode = "agent",
        input_mode = "agent",
        control_type = "stabilization"
        ) 


    # # Check the environment for compatibility
    # check_env(env)

    # # Define and train the model
    model = PPO("MlpPolicy", env, verbose=1)
    
    eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                                log_path='./logs/', eval_freq=1000,
                                deterministic=True, render=False)

    model.learn(total_timesteps=100000, callback=eval_callback)

    # Save the model
    model.save("models\\rl\\ppo_inverted_pendulum")

    # model.load("logs//best_model")
    # # Test the trained agent
    # obs, _ = env.reset()
    # for _ in range(5000):
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, reward, done, _, _ = env.step(action)
    #     env.render()
        
    #     print(action)

    #     if done:
    #         obs, _ = env.reset()

    # env.close()
