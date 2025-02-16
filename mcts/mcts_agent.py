import torch
from copy import deepcopy
from typing import Optional, Any

from mcts.node import Node
from dqn.dqn_agent import DeepQNetworkAgent as Agent


class MCTS:
    """Monte Carlo Tree Search (MCTS) agent using a DQN for model evaluation.

    This class implements the MCTS algorithm integrated with a Deep Q-Network.
    It uses the DQN model to guide tree expansion and value estimation.

    Args:
        branching_factor (int): The number of children each node can have.
        train_episodes (int): The total number of training episodes.
        c_value (float): The exploration constant used in the UCT formula.
        memory (Any): The experience replay buffer.
        env (Any): The environment instance.
    """

    def __init__(self, branching_factor: int, train_episodes: int, c_value: float, memory: Any, env: Any) -> None:
        self.branching_factor: int = branching_factor
        self.train_episodes: int = train_episodes
        self.c_param: float = c_value
        self.memory: Any = memory
        self.env: Any = env

    def _init_search(self, agent: Agent) -> None:
        """Initializes the search process by setting up base models and the root node.

        Loads the DQN model and target model from the provided agent, fills the memory,
        performs an initial training (fit) on the memory, and creates the root node
        with the corresponding core reward.

        Args:
            agent (Agent): The agent used to initialize the search.
        """
        self._base_dqn_model = agent.model
        self._base_dqn_target_model = agent.target_model

        self.discount_factor = agent.discount_factor
        self.epsilon_decay = agent.epsilon_decay
        self.target_update_frequency = agent.target_update_frequency

        self.memory.fill(agent)

        agent.fit(self.memory)
        core_reward: float = agent.validate(self.env)

        self.root = Node(0, 1, core_reward=core_reward, parent=None)

        torch.save(agent.model.state_dict(), self.root.main_path)
        torch.save(agent.model.state_dict(), self.root.target_path)

    def search(self, agent: Agent) -> Node:
        """Performs the MCTS search process.

        Iteratively expands the tree using the tree policy and rollout functions,
        backpropagating the received rewards.

        Args:
            agent (Agent): The agent used for both search expansion and evaluation.

        Returns:
            Node: The root node of the search tree after completing the search.
        """
        self._init_search(agent)
        self.do_search: bool = True

        while True:
            v: Node = self.tree_policy()

            if not self.do_search:
                break
            reward: float = self.rollout(v)
            v.backpropagate(reward)

        return self.root

    def best_branch(self, node: Node) -> Node:
        """Selects the best branch from the provided node by recursively following the best child.

        Args:
            node (Node): The node from which to determine the best branch.

        Returns:
            Node: A deep copy of the node representing the best branch.
        """
        best: Node = deepcopy(node)

        def _best_branch(node: Node) -> None:
            """Recursively selects the best child branch.

            Args:
                node (Node): The current node in the branch.
            """
            if len(node.childrens) == 0:
                return

            best_child: Optional[Node] = node.best_child(self.c_param)
            if best_child is None:
                return

            # Retain only the best child for the branch.
            node.childrens = [best_child]
            _best_branch(best_child)

        _best_branch(best)
        return best

    def tree_policy(self) -> Node:
        """Determines the next node for expansion using the tree policy.

        Iteratively navigates the tree from the root until it finds a node that is not fully expanded
        or is terminal.

        Returns:
            Node: The selected node for expansion.
        """
        current_node: Node = self.root

        while True:
            terminal: bool = self.is_terminal_node(current_node)
            if terminal:
                self.do_search = False
                break
            else:
                if not self.is_fully_expanded(current_node):
                    return self.expand(current_node)
                else:
                    next_node: Optional[Node] = current_node.best_child(self.c_param)
                    if next_node is None:
                        break
                    current_node = next_node

        return current_node

    def is_terminal_node(self, node: Node) -> bool:
        """Checks if a given node is terminal.

        A node is considered terminal if its epoch equals train_episodes - 1.

        Args:
            node (Node): The node to check.

        Returns:
            bool: True if the node is terminal, False otherwise.
        """
        return node.epoch == self.train_episodes - 1

    def is_fully_expanded(self, node: Node) -> bool:
        """Checks whether a node is fully expanded.

        A node is fully expanded if the number of its children equals the branching factor.

        Args:
            node (Node): The node to check.

        Returns:
            bool: True if the node is fully expanded, False otherwise.
        """
        return len(node.childrens) == self.branching_factor

    def expand(self, node: Node) -> Node:
        """Expands a node by creating and appending a new child node.

        The expansion involves loading the saved model states from the node, updating epsilon,
        fitting the agent on the current memory, and validating to obtain a core reward.
        The newly created child node is then added to the parent's children.

        Args:
            node (Node): The node to be expanded.

        Returns:
            Node: The newly created child node.
        """
        self._base_dqn_model.load_state_dict(torch.load(node.main_path))
        self._base_dqn_target_model.load_state_dict(torch.load(node.target_path))

        epsilon: float = node.epsilon * self.epsilon_decay

        agent: Agent = Agent(
            self._base_dqn_model,
            self._base_dqn_target_model,
            epsilon,
            self.discount_factor,
            self.epsilon_decay,
            self.target_update_frequency,
        )

        agent.fit(self.memory)
        # Use validation to assess how much the new batch contributes to model performance.
        core_reward: float = agent.validate(self.env)

        child_node: Node = Node(node.epoch + 1, epsilon, core_reward=core_reward, parent=node)
        node.childrens.append(child_node)

        torch.save(agent.model.state_dict(), child_node.main_path)

        if child_node.epoch % self.target_update_frequency == 0:
            torch.save(agent.model.state_dict(), child_node.target_path)
        else:
            torch.save(agent.target_model.state_dict(), child_node.target_path)

        return child_node

    def rollout(self, node: Node) -> float:
        """Simulates a rollout from the given node.

        Loads the saved model states, creates an agent from the node's parameters,
        and conducts a training and validation process to obtain a reward.

        Args:
            node (Node): The node from which the rollout starts.

        Returns:
            float: The reward obtained from the rollout.
        """
        self._base_dqn_model.load_state_dict(torch.load(node.main_path))
        self._base_dqn_target_model.load_state_dict(torch.load(node.target_path))

        agent: Agent = Agent(
            self._base_dqn_model,
            self._base_dqn_target_model,
            node.epsilon,
            self.discount_factor,
            self.epsilon_decay,
            self.target_update_frequency,
        )

        agent.train(self.env, self.memory, node.epoch, self.train_episodes)

        val_reward: float = agent.validate(self.env)
        return val_reward
