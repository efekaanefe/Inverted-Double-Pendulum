import neat
import pickle
import numpy as np
import gym
from inverted_pendulum_env import InvertedPendulumEnv  

GENOME_PATH = "models/neat/best_genome.pkl"
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

    dt = 1 / 75
    env = InvertedPendulumEnv(
            gravity=-9.81*10*10, 
            dt=dt,
            base_size=(30, 30), 
            base_mass=5,
            link_size=(4, 300), 
            link_mass=1,
            groove_length = 1500,
            initial_angle=270,
            max_steps = 50 // dt,
            actuation_max=1000, # force or speed
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
