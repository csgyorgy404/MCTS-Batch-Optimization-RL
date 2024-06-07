import sys
sys.setrecursionlimit(1000000)

import os
import ast
import copy
import torch
import random
import shutil
import numpy as np
from types import SimpleNamespace
from configparser import ConfigParser

from env import create_env
from dqn.model import Model
from visualise import print_tree
from mcts.mcts_agent import MCTS
from memory.stochastic import Buffer
from dqn.dqn_agent import DeepQNetworkAgent as Agent


def read(path):
    file_conf = ConfigParser()

    file_conf.read(path, encoding="utf8")

    conf_dict = {}
    for section_name in file_conf.sections():
        d = {}
        for key, val in file_conf.items(section_name):
            d[key] = ast.literal_eval(val)

        item = SimpleNamespace(**d)
        conf_dict[section_name] = item
    conf = SimpleNamespace(**conf_dict)

    return conf


def set_seeds(seed):
    """
    Setting packages seed
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    config = read('config.ini')

    seed = config.enviroment.seed

    set_seeds(seed)

    env, in_features, out_features = create_env(config.enviroment.name, config.enviroment.render_mode, seed)

    model = Model(
        in_features=in_features,
        hidden_features=config.model.hidden_features,
        out_features=out_features,
        hidden_activation=config.model.hidden_activation,
        out_activation=config.model.out_activation,
        lr=config.model.lr,
    )

    device = next(model.parameters()).device

    print("Model is on device:", device)

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
