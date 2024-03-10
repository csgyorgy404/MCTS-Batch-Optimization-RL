import torch
import numpy as np
import pandas as pd


class Buffer:
    def __init__(self, env, memory_size, batch_size):
        self.env = env
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = pd.DataFrame(columns=['state', 'action', 'next_state', 'reward', 'done'])


    def add(self, state, action, reward, next_state,  done):
        new_row = {'state': state, 'action': action, 'next_state': next_state, 'reward': reward, 'done': done}

        self.memory = pd.concat([self.memory, pd.DataFrame([new_row])], ignore_index=True)

    def sample(self):
        batch = self.memory.sample(self.batch_size)

        # state = torch.tensor(np.stack(batch['state'].values), dtype=torch.float32)
        # action = torch.tensor(np.stack(batch['action'].values), dtype=torch.int64)
        # next_state = torch.tensor(np.stack(batch['next_state'].values), dtype=torch.float32)
        # reward = torch.tensor(np.stack(batch['reward'].values), dtype=torch.float32)
        # done = torch.tensor(np.stack(batch['done'].values), dtype=torch.int64)

        state = torch.tensor(batch['state'].tolist())
        action = torch.tensor(batch['action'].tolist())
        next_state = torch.tensor(batch['next_state'].tolist())
        reward = torch.tensor(batch['reward'].tolist())
        done =  torch.LongTensor(batch['done'].tolist())

        return state, action, reward, next_state, done

    def fill(self, agent):
        num_of_samples = 0

        while num_of_samples < self.memory_size:
            state, _ = self.env.reset()

            while True:
                action = agent.predict(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.add(state=state, action=action, reward=reward, next_state=next_state,  done=done)

                num_of_samples += 1

                state = next_state

                print(f'Filled memory with {num_of_samples} samples')

                if done:
                    break

        print(f'Filled memory with {num_of_samples} samples')