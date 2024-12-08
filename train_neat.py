import neat
import numpy as np
from inverted_pendulum_env import InvertedPendulumEnv
import pickle
import os

import visualize

save_dir = "models/neat"
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

def evaluate_genomes(genomes, config):
    for genome_id, genome in genomes:
        # Create the neural network for this genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        dt = 1 / 75

        env = InvertedPendulumEnv(
            gravity=-98.1*3, 
            dt=dt,
            base_size=(30, 30), 
            base_mass=5,
            link_size=(4, 300), 
            link_mass=1,
            groove_length = 600,
            initial_angle=270,
            max_steps = int(10 / dt),
            actuation_max=2000, # force or speed
            render_mode = None
            ) 
       
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = np.array(net.activate(obs))[0] * env.actuation_max
            #raw_outputs = net.activate(obs)
            #max_index = np.argmax(raw_outputs)         
            #action = (max_index / (len(raw_outputs) - 1)) * 2 * env.actuation_max - env.actuation_max
            #print(action)

            obs, reward, done, info, _ = env.step(action)  # Use action[0] for scalar action
            total_reward += reward

        genome.fitness = total_reward


def run_neat(config_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run NEAT
    winner = population.run(evaluate_genomes, n=1)  # Run for n generations
    print("\nBest genome:\n", winner)

    save_path = os.path.join(save_dir, "best_genome.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(winner, f)

if __name__ == "__main__":
    run_neat("config-feedforward.txt")
