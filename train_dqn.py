import copy

from utils import *
from env import Env
from dqn.model import Model
from memory.stochastic import Buffer
from dqn.dqn_agent import DeepQNetworkAgent as Agent


def main():
    args = get_args()

    config = read(args.config)

    seed = config.enviroment.seed

    set_seeds(seed)

    env = Env(config.enviroment.name, config.enviroment.max_steps, config.enviroment.render_mode, seed)

    model = Model(
        in_features=env.in_features,
        hidden_features=config.model.hidden_features,
        out_features=env.out_features,
        hidden_activation=config.model.hidden_activation,
        out_activation=config.model.out_activation,
        lr=config.model.lr,
    )

    memory = Buffer(env, config.memory.size, config.memory.batch_size)

    agent = Agent(model, copy.deepcopy(model), 1, config.agent.discount_factor, config.agent.epsilon_decay, config.agent.target_update_frequency)

    memory.fill(agent)

    agent.train_with_interaction(env, memory, 0, config.mcts.train_episodes, True)

    print(agent.validate(env))

if __name__ == "__main__":
    main()

#taxi no one hot
#check cliffwalking termination state
#figure out something for frozenlake reward
