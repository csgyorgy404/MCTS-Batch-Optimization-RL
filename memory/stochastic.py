import torch
import pandas as pd
from tqdm import tqdm
from typing import Any, Tuple


class Buffer:
    """Experience replay buffer for storing transitions.

    This buffer stores transitions and provides a method to sample batches
    for training.

    Args:
        env (Any): The environment instance used to generate experiences.
        memory_size (int): The maximum number of samples to store in the buffer.
        batch_size (int): The number of samples per batch when sampling.
    """

    def __init__(self, env: Any, memory_size: int, batch_size: int) -> None:
        self.env: Any = env
        self.memory_size: int = memory_size
        self.batch_size: int = batch_size
        self.data: pd.DataFrame = pd.DataFrame(columns=["state", "action", "next_state", "reward", "done"])

    def _add(self, state: Any, action: Any, reward: Any, next_state: Any, done: bool) -> None:
        """Adds a new transition to the buffer.

        Args:
            state (Any): The current state.
            action (Any): The action taken.
            reward (Any): The reward received.
            next_state (Any): The next state after taking the action.
            done (bool): Flag indicating whether the episode terminated.
        """
        new_row = {
            "state": state,
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "done": done,
        }
        self.data = pd.concat([self.data, pd.DataFrame([new_row])], ignore_index=True)

    def add(self, state: Any, action: Any, reward: Any, next_state: Any, done: bool) -> None:
        """Adds a new transition while maintaining a fixed memory size.

        This method removes the oldest entry if the memory is full.

        Args:
            state (Any): The current state.
            action (Any): The action taken.
            reward (Any): The reward received.
            next_state (Any): The next state after performing the action.
            done (bool): Flag indicating whether the episode terminated.
        """
        self.data = self.data.iloc[1:]
        self._add(state, action, reward, next_state, done)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples a batch of transitions from the buffer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                A tuple containing tensors for states, actions, next_states, rewards, and done flags.
        """
        batch = self.data.sample(self.batch_size)
        batch_np = batch.to_numpy()

        state = torch.tensor(batch_np[:, 0].tolist())
        action = torch.tensor(batch_np[:, 1].tolist())
        next_state = torch.tensor(batch_np[:, 2].tolist())
        reward = torch.tensor(batch_np[:, 3].tolist())
        done = torch.LongTensor(batch_np[:, 4].tolist())

        return state, action, reward, next_state, done

    def fill(self, agent: Any) -> None:
        """Fills the buffer by interacting with the environment using the agent.

        Args:
            agent (Any): The agent used to generate actions from states.
        """
        num_of_samples = 0

        with tqdm(total=self.memory_size, desc="Filling memory") as pbar:
            while num_of_samples < self.memory_size:
                state = self.env.reset()

                while True:
                    action = agent.train_predict(state)
                    next_state, reward, done = self.env.step(action)

                    self._add(state=state, action=action, reward=reward, next_state=next_state, done=done)

                    num_of_samples += 1
                    pbar.update(1)

                    state = next_state

                    if done or num_of_samples == self.memory_size:
                        break

        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Filled memory with {num_of_samples} samples")
