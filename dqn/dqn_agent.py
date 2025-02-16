import torch
import random
import numpy as np
from torch import nn
from tqdm import tqdm
from typing import Any
import torch.nn.functional as F


class DeepQNetworkAgent:
    """Deep Q-Network (DQN) Agent.

    This agent selects actions using an epsilon-greedy policy and uses deep Q-networks
    for action-value estimation. It includes methods for training, inference, and target network updates.

    Args:
        model (nn.Module): The primary Q-network.
        target_model (nn.Module): The target Q-network.
        epsilon (float): The initial epsilon for the epsilon-greedy policy.
        discount_factor (float): The discount factor for future rewards.
        epsilon_decay (float): The decay factor for epsilon after each training iteration.
        target_update_frequency (int): The number of episodes after which to update the target network.
    """

    def __init__(
        self,
        model: nn.Module,
        target_model: nn.Module,
        epsilon: float,
        discount_factor: float,
        epsilon_decay: float,
        target_update_frequency: int,
    ) -> None:
        super().__init__()
        self.model: nn.Module = model
        self.target_model: nn.Module = target_model
        self.target_model.eval()

        self.epsilon: float = epsilon
        self.discount_factor: float = discount_factor
        self.epsilon_decay: float = epsilon_decay
        self.target_update_frequency: int = target_update_frequency

    def train_predict(self, state: np.ndarray) -> int:
        """Predicts an action during training using an epsilon-greedy policy.

        Args:
            state (np.ndarray): The current state as a numpy array.

        Returns:
            int: The selected action.
        """
        if self.epsilon > random.random():
            # Random action for exploration.
            p: int = np.random.randint(0, self.model.out_features, 1)[0]
        else:
            # Action predicted by the network.
            p = self.inference_predict(state)

        return p

    def inference_predict(self, state: np.ndarray) -> int:
        """Predicts an action during inference using the Q-network.

        Args:
            state (np.ndarray): The current state as a numpy array.

        Returns:
            int: The action with the highest predicted Q-value.
        """
        self.model.eval()

        with torch.no_grad():
            # Reshape the state to match the network's input shape.
            state_tensor: torch.Tensor = torch.reshape(torch.tensor(state, dtype=torch.float32), shape=(1, self.model.in_features))
            actions: torch.Tensor = self.model(state_tensor)
            p: int = int(torch.argmax(actions))

        self.model.train()

        return p

    def decay_epsilon(self) -> None:
        """Decays the epsilon value to reduce randomness in action selection over time."""
        self.epsilon *= self.epsilon_decay

    def fit(self, memory: Any) -> None:
        """Fits the Q-network using a batch of experiences sampled from memory.

        Args:
            memory (Any): The experience replay buffer.
        """
        states, actions, rewards, next_states, terminals = memory.sample()

        # Get predicted Q-values for current states.
        predicted_q: torch.Tensor = self.model(states)
        predicted_q = torch.gather(predicted_q, 1, actions.view(-1, 1)).squeeze()

        # Compute target Q-values using the target network.
        with torch.no_grad():
            target_q: torch.Tensor = self.target_model(torch.squeeze(next_states))
            target_q = target_q.max(dim=-1)[0]
        target_value: torch.Tensor = rewards + self.discount_factor * target_q * (1 - terminals)

        loss: torch.Tensor = F.mse_loss(predicted_q, target_value)

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

    def train(self, env: Any, memory: Any, start: int, end: int) -> list:
        """Trains the agent over multiple episodes.

        For each episode, the agent interacts with the environment to collect reward,
        fits the network on sampled experiences from memory, decays epsilon,
        and updates the target network at specified intervals.

        Args:
            env (Any): The environment in which the agent operates.
            memory (Any): The experience replay buffer.
            start (int): The starting episode index.
            end (int): The ending episode index.

        Returns:
            list: A list of cumulative rewards obtained per episode.
        """
        episode_rewards: list = []
        for episode in tqdm(range(start, end)):
            reward: float = self.validate(env, train_mode=True)

            self.fit(memory)
            self.decay_epsilon()

            episode_rewards.append(reward)

            if (episode + 1) % self.target_update_frequency == 0:
                self.update_target_network()

        return episode_rewards

    def validate(self, env: Any, train_mode: bool = False) -> float:
        """Validates the agent's performance in the environment.

        The agent interacts with the environment until termination,
        summing up the received rewards.

        Args:
            env (Any): The environment to validate the agent on.
            train_mode (bool, optional): Whether to use training mode (which uses epsilon-greedy prediction).
                                         Defaults to False.

        Returns:
            float: The total reward accumulated during validation.
        """
        state: np.ndarray = env.reset()
        total_reward: float = 0
        while True:
            if train_mode:
                action: int = self.train_predict(state)
            else:
                action = self.inference_predict(state)
            state, reward, done = env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward

    def update_target_network(self) -> None:
        """Updates the target network to match the primary Q-network."""
        self.target_model.load_state_dict(self.model.state_dict())
