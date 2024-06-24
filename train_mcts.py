import sys
sys.setrecursionlimit(1000000)

import copy
import shutil
import numpy as np

from utils import *
from env import Env
from dqn.model import Model
from visualise import print_tree
from mcts.mcts_agent import MCTS
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

    mcts = MCTS(config.mcts.branching_factor, config.mcts.train_episodes, config.mcts.c, memory, env)

    agent = Agent(model, copy.deepcopy(model), 1,  config.agent.discount_factor, config.agent.epsilon_decay, config.agent.target_update_frequency)


    try:
        root = mcts.search(agent, False)
        best = mcts.best_branch(root)
    except KeyboardInterrupt:
        best = mcts.best_branch(mcts.root)

    rewards = []
    start = best

    while len(start.childrens) > 0:
        rewards.append(start.core_reward)
        start = start.childrens[0]

    np.save('rewards_mcts.npy', rewards)
        
    print_tree(best, 'best')
    print_tree(mcts.root, 'all')

    shutil.rmtree('nodes/')


if __name__ == "__main__":
    main()
