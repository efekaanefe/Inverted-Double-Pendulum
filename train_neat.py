import neat
import numpy as np
from inverted_pendulum_env import InvertedPendulumEnv
import pickle
import os
from multiprocessing import Pool, cpu_count

import visualize

save_dir = "models/neat"
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

def evaluate_genomes(genomes, config):
    # TODO: add multiprocessing
    for genome_id, genome in genomes:
        # Create the neural network for this genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        dt = 1 / 75

        env = InvertedPendulumEnv(
            gravity=-9.81*10*5, 
            dt=dt,
            base_size=(30, 30), 
            base_mass=5,
            link_size=(4, 300), 
            link_mass=0.5,
            groove_length = 1500,
            initial_angle=270,
            max_steps = 15 // dt,
            actuation_max=15000, # force or speed
            margin = 5,
            render_mode = None,
            input_mode = "agent",
            ) 
       
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = net.activate(obs)[0]

            scaled_action = action * env.actuation_max

            #raw_outputs = net.activate(obs)
            #max_index = np.argmax(raw_outputs)         
            #action = (max_index / (len(raw_outputs) - 1)) * 2 * env.actuation_max - env.actuation_max

            obs, reward, done, _, info = env.step(scaled_action)
            total_reward += reward

        genome.fitness = total_reward

# def evaluate_single_genome(genome_data):
#     """
#     Evaluate a single genome in the environment.
#     """
#     genome_id, genome, config = genome_data

#     # Create the neural network for this genome
#     net = neat.nn.FeedForwardNetwork.create(genome, config)
#     dt = 1 / 75

#     env = InvertedPendulumEnv(
#         gravity=-98.1 * 3,
#         dt=dt,
#         base_size=(30, 30),
#         base_mass=5,
#         link_size=(4, 300),
#         link_mass=1,
#         groove_length=600,
#         initial_angle=270,
#         max_steps=int(10 / dt),
#         actuation_max=2000,
#         render_mode=None
#     )

#     obs, _ = env.reset()
#     total_reward = 0
#     done = False

#     while not done:
#         action = np.array(net.activate(obs))[0] * env.actuation_max
#         obs, reward, done, _, info = env.step(action)
#         total_reward += reward

#     # Assign fitness to the genome
#     genome.fitness = total_reward
#     return genome_id, genome

# def evaluate_genomes(genomes, config):
#     """
#     Evaluate all genomes using multiprocessing.
#     """
#     # Prepare data for parallel processing
#     genome_data = [(genome_id, genome, config) for genome_id, genome in genomes]

#     # Use multiprocessing Pool for parallel genome evaluation
#     with Pool(processes=cpu_count()) as pool:
#         results = pool.map(evaluate_single_genome, genome_data)

#     # Update genomes with fitness from results
#     for genome_id, genome in results:
#         print(genome_id)
#         genomes[genome_id-1] = genome


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
    winner = population.run(evaluate_genomes, n=500)  # Run for n generations
    print("\nBest genome:\n", winner)

    save_path = os.path.join(save_dir, "best_genome.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(winner, f)

if __name__ == "__main__":
    run_neat("config-feedforward.txt")
