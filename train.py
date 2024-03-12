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

def add_nodes(node, G):
    node_label = f"epoch: {node.epoch}\nn={node.n}\nq={node.q}"
    G.add_node(node.version, label=node_label)
    for child in node.childrens:
        G.add_edge(node.version, child.version)
        add_nodes(child, G)

def plot_mcts_tree(root_node, save_path=None):
    G = nx.DiGraph()
    add_nodes(root_node, G)
    pos = hierarchy_pos(G, root_node)
    
    labels = nx.get_node_attributes(G, 'label')  # Get labels for nodes
    nx.draw(G, pos, with_labels=False, node_size=700, node_color="skyblue", font_size=10, font_weight="bold")
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black', verticalalignment='center')  # Draw labels separately
    
    if save_path:
        plt.savefig(save_path)  # Save the figure if save_path is provided
    else:
        plt.show()


def hierarchy_pos(G, root, width=1., vert_gap=0.5, vert_loc=0, xcenter=0.5):
    pos = {root.version: (xcenter, vert_loc)}
    children = [child for child in root.childrens] # Convert to list
    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos[child.version] = (nextx, vert_loc - vert_gap)
            pos.update(hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc - vert_gap, xcenter=nextx))
    return pos

def add_nodes_to_tree(node, tree, parent_id=None):
    node_tag = f"e: {node.epoch}, n: {node.n}, q: {node.q}"  # Include n and q values in the node tag
    current_id = tree.create_node(tag=node_tag, data=node, parent=parent_id)
    for child in node.childrens:
        add_nodes_to_tree(child, tree, parent_id=current_id)

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

    mcts = MCTS(config.mcts.branching_factor,config.mcts.search_steps, config.mcts.train_episodes, config.mcts.validation_episodes, config.mcts.c, memory, env)

    agent = Agent(model, model, 1, config.agent.discount_factor, config.agent.epsilon_decay, config.agent.target_update_frequency)

    mcts.search(agent)

    plot_mcts_tree(mcts.root, save_path="mcts_tree.png")

    tree = Tree()

    add_nodes_to_tree(mcts.root, tree)

    with open("tree_structure.txt", "w") as f:
        sys.stdout = f
        print(tree.show(stdout=False))
        sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
