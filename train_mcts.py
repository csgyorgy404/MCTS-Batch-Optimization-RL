import copy
import shutil
import numpy as np

from utils import *
from env import Env
from dqn.model import Model
from mcts.mcts_agent import MCTS
from memory.stochastic import Buffer
from visualise import visualise_trian_results
from dqn.dqn_agent import DeepQNetworkAgent as Agent


def main():
    # Parse command-line arguments to obtain configuration file location and other parameters.
    args = get_args()

    # Read configuration settings from the specified config file.
    config = read(args.config)

    # Retrieve the seed value from the configuration for reproducibility.
    seed = config.enviroment.seed

    # Set random seeds for various libraries (Python random, NumPy, torch, etc.).
    set_seeds(seed)

    # Initialize the environment with the parameters specified in the configuration.
    env = Env(config.enviroment.name, config.enviroment.max_steps, config.enviroment.render_mode, seed)

    # Create the DQN model with the specified architecture.
    model = Model(
        in_features=env.in_features,
        hidden_features=config.model.hidden_features,
        out_features=env.out_features,
        hidden_activation=config.model.hidden_activation,
        out_activation=config.model.out_activation,
        lr=config.model.lr,
    )

    # Initialize the experience replay buffer with the environment and configuration settings.
    memory = Buffer(env, config.memory.size, config.memory.batch_size)

    # Initialize the MCTS agent with the provided branching factor, number of training episodes, exploration constant,
    # the replay buffer and the environment.
    mcts = MCTS(config.mcts.branching_factor, config.mcts.train_episodes, config.mcts.c, memory, env)

    # Initialize the DQN agent; a deep copy of the model is used for the target network.
    agent = Agent(
        model,
        copy.deepcopy(model),
        1,  # This parameter may serve as an initial hyperparameter (e.g., initial epsilon or learning rate multiplier).
        config.agent.discount_factor,
        config.agent.epsilon_decay,
        config.agent.target_update_frequency,
    )

    # Execute the MCTS search process.
    try:
        # Start the MCTS search from the root node.
        root = mcts.search(agent)
        # Select the best branch from the search result.
        best = mcts.best_branch(root)
    except KeyboardInterrupt:
        # If the process is interrupted, choose the best branch from the current MCTS root.
        best = mcts.best_branch(mcts.root)

    # Traverse the best branch to extract the core rewards.
    rewards = []
    start = best
    while len(start.childrens) > 0:
        rewards.append(start.core_reward)
        start = start.childrens[0]  # Move to the next node in the best branch

    # Save the collected MCTS rewards into a NumPy file for later analysis or visualization.
    np.save("rewards_mcts.npy", rewards)

    # Remove the directory containing node data to free up space.
    shutil.rmtree("nodes/")

    # Visualise the MCTS training results using matplotlib.
    visualise_trian_results(mcts_reward=rewards, smoothing_f=15)


if __name__ == "__main__":
    main()
