import sys
import networkx as nx
from treelib import Tree
import matplotlib.pyplot as plt


def print_tree(node, title):
    tree = Tree()

    def add_nodes_to_tree(node, tree, parent_id=None):
        node_tag = f"e: {node.epoch}, n: {node.n}, q: {node.q}, core: {node.core_reward}"  # Include n and q values in the node tag
        current_id = tree.create_node(tag=node_tag, data=node, parent=parent_id)
        for child in node.childrens:
            add_nodes_to_tree(child, tree, parent_id=current_id)

    add_nodes_to_tree(node, tree)

    with open(f"{title}_structure.txt", "w") as f:
        sys.stdout = f
        print(tree.show(stdout=False))
        sys.stdout = sys.__stdout__