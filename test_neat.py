import neat
import pickle
import numpy as np
from inverted_pendulum_env import InvertedPendulumEnv 
from constants import *

# TODO: make control type reached from constants
control_type = "stabilization"
reward = 4.44
GENOME_PATH = f"models\\{control_type}\\neat\\best_genome_{reward}.pkl"
CONFIG_PATH = "config-feedforward.txt"

def visualize_best_model():
    """
    Load the best genome and visualize its performance in the environment.
    """
    config = neat.Config(
        neat.DefaultGenome, 
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet, 
        neat.DefaultStagnation,
        CONFIG_PATH)

    with open(GENOME_PATH, "rb") as f:
        best_genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(best_genome, config)

    env = InvertedPendulumEnv(
            gravity=gravity, 
            dt=dt,
            base_size=base_size, 
            base_mass=base_mass,
            link_size=link_size, 
            link_mass=link_mass,
            groove_length = groove_length,
            max_steps = max_steps,
            actuation_max=actuation_max, # force or speed
            margin = margin,
            render_mode = "human",
            input_mode = "agent",
            control_type = control_type
            ) 

    obs,_  = env.reset()
    total_reward = 0
    done = False

    env.step(3000) # initial push to the right

    while not done:
        env.render()

        # disturbance
        if np.random.uniform(0, 1) < 0.0:
            force = np.random.uniform(-env.actuation_max, env.actuation_max)
            env.base_body.apply_force_at_local_point((force,0))

            print(f"Disturbance: {force}")

        action = net.activate(obs)[0] * env.actuation_max  

        obs, reward, done, _, info  = env.step(action)
        total_reward += reward
        
        print(
            action,
            net.activate(obs)[0] 
            # -env.actuation_max + action,
            # reward
        )

    print(f"Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    visualize_best_model()
