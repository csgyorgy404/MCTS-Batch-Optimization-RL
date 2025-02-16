import os
import numpy as np
from uuid import uuid4
from typing import Optional, Union, List


class Node:
    """A node in a Monte Carlo Tree Search.

    Each node represents a state in the search tree and stores statistical
    information for the search algorithm.

    Args:
        epoch (Union[str, int]): The epoch or identifier for the node's creation.
        epsilon (float): Epsilon value for exploring actions.
        core_reward (Optional[float]): The core reward for this node. Defaults to None.
        parent (Optional[Node]): The parent node. Defaults to None.
    """

    def __init__(
        self, epoch: Union[str, int], epsilon: float, core_reward: Optional[float] = None, parent: Optional["Node"] = None
    ) -> None:
        self._visits: int = 0
        self._value: float = 0.0
        self._core_reward: Optional[float] = core_reward
        self.parent: Optional["Node"] = parent
        self.childrens: List["Node"] = []
        self.epoch: Union[str, int] = epoch
        self.epsilon: float = epsilon
        self.version: str = str(uuid4())

        self._init_paths(str(epoch), self.version)

    @property
    def core_reward(self) -> Optional[float]:
        """Gets the core reward.

        Returns:
            Optional[float]: The core reward.
        """
        return self._core_reward

    @core_reward.setter
    def core_reward(self, value: float) -> None:
        """Sets the core reward.

        Args:
            value (float): The new core reward.
        """
        self._core_reward = value

    @property
    def n(self) -> int:
        """Number of visits.

        Returns:
            int: The number of times this node has been visited.
        """
        return self._visits

    @property
    def q(self) -> float:
        """Cumulative value.

        Returns:
            float: The total value accumulated at this node.
        """
        return self._value

    def _init_paths(self, epoch: str, version: str) -> None:
        """Initializes file paths for saving node data.

        Creates the necessary directories and constructs file paths for the main and target files.

        Args:
            epoch (str): The epoch identifier.
            version (str): The unique version identifier for the node.
        """
        root_folder: str = "nodes"

        main_path: str = os.path.join(root_folder, epoch, version)
        target_path: str = os.path.join(root_folder, epoch, version)

        os.makedirs(main_path, exist_ok=True)
        os.makedirs(target_path, exist_ok=True)

        self.main_path: str = os.path.join(main_path, "main.pth")
        self.target_path: str = os.path.join(target_path, "target.pth")

    def backpropagate(self, score: float) -> None:
        """Backpropagates the given score up the tree.

        Increments the visit count and adds the score to the cumulative value.
        Recursively updates parent nodes.

        Args:
            score (float): The score to propagate.
        """
        self._visits += 1
        self._value += score
        if self.parent:
            self.parent.backpropagate(score)

    def best_child(self, c_value: float = 1.4) -> Optional["Node"]:
        """Selects the best child node based on the UCT (Upper Confidence Bound for Trees) formula.

        The score for each child is computed as the sum of an exploitation term and an
        exploration term. If multiple children have the same score, one is selected randomly.

        Args:
            c_value (float, optional): The exploration constant. Defaults to 1.4.

        Returns:
            Optional[Node]: The selected best child node, or None if no valid child is found.
        """
        exploit: List[float] = []
        explore: List[float] = []

        for child in self.childrens:
            try:
                exploit.append((child.q + child.core_reward) / child.n)
                explore.append(c_value * np.sqrt(2 * np.log(self.n) / child.n))
            except Exception:
                exploit.append(0)
                explore.append(np.inf)

        choices_weights = [i + j for i, j in zip(exploit, explore)]
        max_value = np.max(choices_weights)
        # If multiple children have the same weight, select one randomly.
        indexes = np.where(choices_weights == max_value)[0]

        try:
            return self.childrens[np.random.choice(indexes)]
        except Exception:
            return None
