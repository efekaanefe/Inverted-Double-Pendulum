import neat
import numpy as np
from inverted_pendulum_env import InvertedPendulumEnv

# import visualize
# visualize.draw_net(config, winner)
# visualize.plot_stats(stats)
# visualize.plot_species(stats)
#
def evaluate_genome(genome, config):
    # Create the neural network for this genome
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = InvertedPendulumEnv(
        gravity=-98.1*2,
        dt=1/75,
        force_mag=1500,
        base_size=(30, 30),
        base_mass=5,
        link_size=(6, 250),
        link_mass=1,
        groove_length=600,
        initial_angle=np.pi,
        max_steps=200
    )

    obs = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Use the neural network to decide the action
        action = net.activate(obs)
        obs, reward, done, info = env.step(action[0])  # action[0] is used for scalar actions
        total_reward += reward

    return total_reward


def run_neat(config_file):
    # Load NEAT configuration
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    # Create the population
    population = neat.Population(config)

    # Add reporters to show progress in the terminal
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run NEAT
    winner = population.run(evaluate_genome, n=50)  # Run for 50 generations
    print("\nBest genome:\n", winner)

if __name__ == "__main__":
    run_neat("config-feedforward")
