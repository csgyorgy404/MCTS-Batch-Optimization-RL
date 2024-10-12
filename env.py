import numpy as np
import gymnasium as gym
from numpy import cos, sin

class Env:
    def __init__(self, env_name, max_steps=None, render_mode=None, seed=42):
        self.name = env_name
        self.seed = seed
        self.max_steps = max_steps
        self._start_steps = 0
        try:
            if env_name == 'FrozenLake-v1':
                self.gym_env = gym.make(env_name,render_mode=render_mode, is_slippery=False)
            else:
                self.gym_env = gym.make(env_name, render_mode=render_mode)
        except gym.error.NameNotFound:
            raise ValueError(f"{env_name} environment is not implemented or"
                                f" not found in the official gym repository")
        
    @property
    def in_features(self):
        try:
            if len(self.gym_env.observation_space.shape) > 1:
                return self.gym_env.observation_space.shape[0] * self.gym_env.observation_space.shape[1]
            return self.gym_env.observation_space.shape[0]
        except:
            return self.gym_env.observation_space.n
    
    @property
    def out_features(self):
        try:
            return self.gym_env.action_space.shape[0]
        except:
            return self.gym_env.action_space.n
        
    def _one_hot(self, state):
        if isinstance(state, int):
            one_hot = np.zeros(self.in_features)
            one_hot[state] = 1

            return one_hot.astype(np.float32)
            # return np.float32(state)
        else:
            return state

    def reset(self):
        state, _ = self.gym_env.reset(seed=self.seed)

        self._start_steps = 0
        
        return self._one_hot(state)
        # return state

    def step(self, action):
        prev_action = None

        next_state, reward, terminated, truncated, _ = self.gym_env.step(action)

        if self.name == 'CliffWalking-v0':
            if prev_action == 0 and action == 2:
                terminated = True
            elif prev_action == 1 and action == 3:
                terminated = True
            elif prev_action == 2 and action == 0:
                terminated = True
            elif prev_action == 3 and action == 1:
                terminated = True
        elif self.name == 'Acrobot-v1':
            state = self.gym_env.state

            height = -cos(state[0]) - cos(state[1] + state[0])
        
            reward += height * 0.01


        self._start_steps += 1

        if self.max_steps is not None and self._start_steps >= self.max_steps:
            truncated = True

        done = terminated or truncated

        return self._one_hot(next_state), reward, done
        # return next_state, reward, done

    def render(self):
        self.gym_env.render()

    def close(self):
        self.gym_env.close()

    # def __del__(self):
    #     self.gym_env.close()
