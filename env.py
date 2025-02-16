import numpy as np
import gymnasium as gym
from numpy import cos, sin
from typing import Optional, Union, Tuple, Any, Dict


class Env:
    """Environment wrapper for gymnasium environments.

    This class provides a unified API for different gymnasium environments,
    handling custom behavior such as one-hot encoding of states and special
    termination conditions.

    Args:
        env_name (str): Name of the gymnasium environment.
        max_steps (Optional[int]): Maximum number of steps for the environment.
        render_mode (Optional[str]): Rendering mode to be used by the environment.
        seed (int): Seed for the environment randomness.
    """

    def __init__(self, env_name: str, max_steps: Optional[int] = None, render_mode: Optional[str] = None, seed: int = 42) -> None:
        self.name: str = env_name
        self.seed: int = seed
        self.max_steps: Optional[int] = max_steps
        self._start_steps: int = 0
        try:
            if env_name == "FrozenLake-v1":
                self.gym_env = gym.make(env_name, render_mode=render_mode, is_slippery=False)
            else:
                self.gym_env = gym.make(env_name, render_mode=render_mode)
        except gym.error.NameNotFound:
            raise ValueError(f"{env_name} environment is not implemented or" f" not found in the official gym repository")

    @property
    def in_features(self) -> int:
        """Calculates the number of input features from the environment's observation space.

        Returns:
            int: The flattened number of input features.
        """
        try:
            if len(self.gym_env.observation_space.shape) > 1:
                return self.gym_env.observation_space.shape[0] * self.gym_env.observation_space.shape[1]
            return self.gym_env.observation_space.shape[0]
        except Exception:
            return self.gym_env.observation_space.n

    @property
    def out_features(self) -> int:
        """Retrieves the number of output features from the environment's action space.

        Returns:
            int: The number of output actions.
        """
        try:
            return self.gym_env.action_space.shape[0]
        except Exception:
            return self.gym_env.action_space.n

    def _one_hot(self, state: Union[int, np.ndarray]) -> np.ndarray:
        """Converts an integer state into a one-hot encoded numpy array.

        Args:
            state (Union[int, np.ndarray]): The state to be encoded.

        Returns:
            np.ndarray: The one-hot encoded state if input is an integer,
                otherwise returns the state unchanged.
        """
        if isinstance(state, int):
            one_hot = np.zeros(self.in_features)
            one_hot[state] = 1
            return one_hot.astype(np.float32)
        else:
            return state

    def reset(self) -> np.ndarray:
        """Resets the environment and returns the initial state.

        Returns:
            np.ndarray: The initial state after resetting the environment.
        """
        state, _ = self.gym_env.reset(seed=self.seed)
        self._start_steps = 0
        return self._one_hot(state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Performs a single step in the environment using the given action.

        Args:
            action (int): The action to be executed in the environment.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: A tuple containing:
                - next_state (np.ndarray): The state after taking the action.
                - reward (float): The reward received after taking the action.
                - done (bool): Flag indicating whether the episode terminated.
        """
        prev_action = None

        next_state, reward, terminated, truncated, info = self.gym_env.step(action)

        if self.name == "CliffWalking-v0":
            if prev_action == 0 and action == 2:
                terminated = True
            elif prev_action == 1 and action == 3:
                terminated = True
            elif prev_action == 2 and action == 0:
                terminated = True
            elif prev_action == 3 and action == 1:
                terminated = True
        elif self.name == "Acrobot-v1":
            state = self.gym_env.state

            height = -cos(state[0]) - cos(state[1] + state[0])

            reward += height * 0.01

        elif self.name == "MountainCar-v0":
            position, velocity = next_state

            # Reward for reaching the goal
            if position >= 0.5:
                reward = 100.0

            # Penalty for taking too long (can help encourage quicker solutions)
            time_penalty = -1.0

            # Small positive reward for moving right (toward the goal)
            direction_reward = 1.0 if velocity > 0 else -1.0

            reward = time_penalty + direction_reward

        self._start_steps += 1

        if self.max_steps is not None and self._start_steps >= self.max_steps:
            truncated = True

        done = terminated or truncated

        return self._one_hot(next_state), reward, done

    def render(self) -> None:
        """Renders the environment."""
        self.gym_env.render()

    def close(self) -> None:
        """Closes the environment."""
        self.gym_env.close()
