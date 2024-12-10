import neat
import pickle
import numpy as np
import gym
from inverted_pendulum_env import InvertedPendulumEnv 
from constants import *
from agents import PIDAgent


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

    env = InvertedPendulumEnv(
            gravity=gravity, 
            dt=dt,
            base_size=base_size, 
            base_mass=base_mass,
            link_size=link_size, 
            link_mass=link_mass,
            groove_length = groove_length,
            initial_angle=initial_angle,
            max_steps = max_steps * 3,
            actuation_max=actuation_max, # force or speed
            margin = margin,
            render_mode = "human",
            input_mode = "agent"
            ) 

    obs,_  = env.reset()
    total_reward = 0
    done = False

    P, I, D = 100, 5, 2 
    agent = PIDAgent(P, I, D)

    pid_started_once = False

    while not done:
        env.render()

        theta = obs[2]
        if 90 - env.margin <= theta <= 90 + env.margin or pid_started_once == True: # stabilizing
            observation_error = 90 - theta
            action = agent.choose_action(observation_error) 
            action = (action * 2 - 1) * env.actuation_max/5 # [0, 1] -> action [-env.actuation_max, env.actuation_max]

            

            print(f"PID: {action}, {observation_error}")

            pid_started_once = True
        else:
            action = net.activate(obs)[0] * env.actuation_max  
            #print(f"NEAT: {action}")


        obs, reward, done, info, _ = env.step(action)
        total_reward += reward

    print(f"Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    visualize_best_model()
