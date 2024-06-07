import torch
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Buffer:
    def __init__(self, env, memory_size, batch_size):
        self.env = env
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = pd.DataFrame(columns=['state', 'action', 'next_state', 'reward', 'done'])


    def _add(self, state, action, reward, next_state,  done):
        new_row = {'state': state, 'action': action, 'next_state': next_state, 'reward': reward, 'done': done}

        self.data = pd.concat([self.data, pd.DataFrame([new_row])], ignore_index=True)

    def add(self, state, action, reward, next_state, done):
        self.data = self.data.iloc[1:]

        self._add(state, action, reward, next_state, done)

    # def sample(self):
    #     batch = self.memory.sample(self.batch_size)

    #     # state = torch.tensor(np.stack(batch['state'].values), dtype=torch.float32)
    #     # action = torch.tensor(np.stack(batch['action'].values), dtype=torch.int64)
    #     # next_state = torch.tensor(np.stack(batch['next_state'].values), dtype=torch.float32)
    #     # reward = torch.tensor(np.stack(batch['reward'].values), dtype=torch.float32)
    #     # done = torch.tensor(np.stack(batch['done'].values), dtype=torch.int64)

    #     state = torch.tensor(batch['state'].tolist()).to(device)
    #     action = torch.tensor(batch['action'].tolist()).to(device)
    #     next_state = torch.tensor(batch['next_state'].tolist()).to(device)
    #     reward = torch.tensor(batch['reward'].tolist()).to(device)
    #     done =  torch.LongTensor(batch['done'].tolist()).to(device)

    #     return state, action, reward, next_state, done
    def sample(self):
        batch = self.data.sample(self.batch_size)

        # Convert batch DataFrame to numpy array
        batch_np = batch.to_numpy()

        # Extract columns from the numpy array
        state = torch.tensor(batch_np[:, 0].tolist()).to(device)
        action = torch.tensor(batch_np[:, 1].tolist()).to(device)
        next_state = torch.tensor(batch_np[:, 2].tolist()).to(device)
        reward = torch.tensor(batch_np[:, 3].tolist()).to(device)
        done = torch.LongTensor(batch_np[:, 4].tolist()).to(device)

        return state, action, reward, next_state, done


    def fill(self, agent):
        num_of_samples = 0

        while num_of_samples < self.memory_size: #1a
            state, _ = self.env.reset()

            while True:
                action = agent.predict(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self._add(state=state, action=action, reward=reward, next_state=next_state,  done=done)

                num_of_samples += 1

                state = next_state

                if done or num_of_samples == self.memory_size:
                    break

        print(f'Filled memory with {num_of_samples} samples')