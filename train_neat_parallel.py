import neat
import numpy as np
from inverted_pendulum_env import InvertedPendulumEnv
from constants import *
import pickle
import os
from concurrent.futures import ThreadPoolExecutor

import visualize

control_type="stabilization"

save_dir = f"models\\{control_type}\\neat"
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

def evaluate_genome(genome_config):
    genome_id, genome, config = genome_config

    # Create the neural network for this genome
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    env = InvertedPendulumEnv(
        gravity=gravity,
        dt=dt,
        base_size=base_size,
        base_mass=base_mass,
        link_size=link_size,
        link_mass=link_mass,
        groove_length=groove_length,
        max_steps=max_steps,
        actuation_max=actuation_max,
        margin=margin,
        render_mode="agent",
        input_mode="agent",
        control_type=control_type
    )

    obs, _ = env.reset()
    total_reward = 0
    done = False

    env.base_body.apply_force_at_local_point((env.actuation_max,0))

    while not done:

        # disturbance
        if np.random.uniform(0, 1) < 0.0:
            force = np.random.uniform(-env.actuation_max, env.actuation_max)
            env.base_body.apply_force_at_local_point((force,0))

        action = net.activate(obs)[0] * env.actuation_max
        obs, reward, done, _, info = env.step(action)
        total_reward += reward

    genome.fitness = total_reward


def evaluate_genomes(genomes, config):
    # Prepare genome data for parallel processing
    genome_configs = [(genome_id, genome, config) for genome_id, genome in genomes]

    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor() as executor:
        executor.map(evaluate_genome, genome_configs)


def run_neat(config_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run NEAT
    winner = population.run(evaluate_genomes, n=100)  # Run for n generations
    print("\nBest genome:\n", winner)

    save_path = os.path.join(save_dir, f"best_genome_{winner.fitness}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(winner, f)


if __name__ == "__main__":
    run_neat("config-feedforward.txt")
