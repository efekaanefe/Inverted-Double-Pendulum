import neat
import pickle
import numpy as np
import gym
from inverted_pendulum_env import InvertedPendulumEnv 
from constants import *


GENOME_PATH = "models/neat/best_genome_667.pkl"
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
            max_steps = max_steps * 1,
            actuation_max=actuation_max, # force or speed
            margin = margin,
            render_mode = "human",
            input_mode = "agent"
            ) 

    obs,_  = env.reset()
    total_reward = 0
    done = False

    while not done:
        env.render()

        action = net.activate(obs)[0] * env.actuation_max  

        obs, reward, done, info, _ = env.step(action)
        total_reward += reward

    print(f"Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    visualize_best_model()
