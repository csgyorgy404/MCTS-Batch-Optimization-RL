import copy

from utils import *
from env import Env
from dqn.model import Model
from memory.stochastic import Buffer
from visualise import visualise_trian_results
from dqn.dqn_agent import DeepQNetworkAgent as Agent


def main() -> None:
    # Parse command line arguments (e.g., configuration file path).
    args = get_args()

    # Read configuration settings from the given config file.
    config = read(args.config)

    # Retrieve the seed value from the configuration for reproducibility.
    seed = config.enviroment.seed

    # Set all random seeds (Python, NumPy, torch, etc.) using the utility function.
    set_seeds(seed)

    # Initialize the environment using configuration parameters.
    env = Env(config.enviroment.name, config.enviroment.max_steps, config.enviroment.render_mode, seed)

    # Create the DQN model with the specified architecture and hyperparameters.
    model = Model(
        in_features=env.in_features,
        hidden_features=config.model.hidden_features,
        out_features=env.out_features,
        hidden_activation=config.model.hidden_activation,
        out_activation=config.model.out_activation,
        lr=config.model.lr,
    )

    # Initialize the experience replay buffer.
    memory = Buffer(env, config.memory.size, config.memory.batch_size)

    # Initialize the DQN agent.
    # A deep copy of the model is used for the target network in DQN.
    agent = Agent(
        model,
        copy.deepcopy(model),
        1,  # This parameter can represent initial epsilon, learning rate multiplier, or other hyperparameter.
        config.agent.discount_factor,
        config.agent.epsilon_decay,
        config.agent.target_update_frequency,
    )

    # Fill the experience replay buffer by having the agent interact with the environment.
    memory.fill(agent)

    # Train the DQN agent using the experiences stored in the memory buffer.
    # The function returns the rewards collected during training.
    rewards = agent.train(env, memory, 0, config.mcts.train_episodes)

    # Save the collected training rewards to a file for later analysis/visualisation.
    np.save("rewards_dqn.npy", rewards)

    # Visualise the training results using matplotlib.
    visualise_trian_results(dqn_reward=rewards, smoothing_f=15)


if __name__ == "__main__":
    main()
