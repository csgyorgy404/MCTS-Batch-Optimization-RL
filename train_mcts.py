import ast
import sys
import os
import torch
import random
import numpy as np
import networkx as nx
from treelib import Tree
import matplotlib.pyplot as plt
from types import SimpleNamespace
from configparser import ConfigParser

from env import create_env
from dqn.model import Model
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


def print_tree(node, title):
    tree = Tree()

    def add_nodes_to_tree(node, tree, parent_id=None):
        node_tag = f"e: {node.epoch}, n: {node.n}, q: {node.q}, X: {node.core_reward}"  # Include n and q values in the node tag
        current_id = tree.create_node(tag=node_tag, data=node, parent=parent_id)
        for child in node.childrens:
            add_nodes_to_tree(child, tree, parent_id=current_id)

    add_nodes_to_tree(node, tree)

    with open(f"{title}_structure.txt", "w") as f:
        sys.stdout = f
        print(tree.show(stdout=False))
        sys.stdout = sys.__stdout__


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

    memory = Buffer(env, config.memory.size, config.memory.batch_size)

    mcts = MCTS(config.mcts.branching_factor,config.mcts.search_steps, config.mcts.train_episodes, config.mcts.validation_episodes, config.mcts.c, memory, env)

    agent = Agent(model, model, config.agent.epsilon_decay, config.agent.discount_factor, config.agent.epsilon_decay, config.agent.target_update_frequency)

    best = mcts.search(agent)

    print_tree(best, 'best')
    print_tree(mcts.root, 'all')


if __name__ == "__main__":
    main()
