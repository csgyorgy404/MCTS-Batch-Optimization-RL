import ast
import sys
import networkx as nx
from treelib import Tree
import matplotlib.pyplot as plt
from types import SimpleNamespace
from configparser import ConfigParser

from env import create_env
from dqn.model import Model
from mcts.agent import MCTS
from memory.stochastic import Buffer
from dqn.agent import DeepQNetworkAgent as Agent


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


def main():
    config = read('config.ini')

    env, in_features, out_features = create_env(config.enviroment.name, config.enviroment.render_mode)

    model = Model(
        in_features=in_features,
        hidden_features=config.model.hidden_features,
        out_features=out_features,
        hidden_activation=config.model.hidden_activation,
        out_activation=config.model.out_activation
    )

    memory = Buffer(env, config.memory.size, config.memory.batch_size)

    agent = Agent(model, model, 0.997, config.agent.discount_factor, config.agent.epsilon_decay, config.agent.target_update_frequency)

    memory.fill(agent)

    agent.train(env, memory, 0, config.mcts.train_episodes, True)

    agent.validate(env)

  


if __name__ == "__main__":
    main()
