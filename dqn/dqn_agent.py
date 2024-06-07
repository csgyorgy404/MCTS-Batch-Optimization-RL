import torch
import random
import numpy as np
from tqdm import tqdm
from time import time
import torch.nn.functional as F


class DeepQNetworkAgent():
    def __init__(self,model, target_model, epsilon, discount_factor, epsilon_decay, target_update_frequency):
        super().__init__()
        self.model = model
        self.target_model = target_model
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.epsilon_decay = epsilon_decay
        self.target_update_frequency = target_update_frequency
        

    def predict(self, state: np.ndarray) -> int:
        if self.epsilon > random.random():
            p =  np.random.randint(0, self.model.out_features, 1)[0]
          
        else:
            p = self.inference(state)

        return p

    def inference(self, state: np.ndarray) -> int:
        self.model.eval()
        with torch.no_grad():
            actions = self.model(torch.reshape(torch.tensor(state, dtype=torch.float32),
                                                                shape=(1, self.model.in_features)))
                                                            #  shape=(1, self.model.in_features))).detach().cpu().numpy()
            p = int(torch.argmax(actions.cpu()))
        self.model.train()

        return p

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def fit(self, memory):

        states, actions, rewards, next_states, terminals = memory.sample() #1c batch size?

        predicted_q = self.model(states)
        predicted_q = torch.gather(predicted_q, 1, actions.view((-1, 1))).squeeze()

      
        with torch.no_grad():
            target_q = self.target_model(torch.squeeze(next_states))
            target_q = target_q.max(dim=-1)[0]
        target_value = rewards + self.discount_factor * target_q * (1 - terminals)

        loss = F.mse_loss(predicted_q, target_value)

        self.model.optimizer.zero_grad()
        loss.backward()

        self.model.optimizer.step()

        return loss

  
        
    def train(self, env, memory, start, end, verbose=False):
        print(f"Training from episode {start} to episode {end}")

        for episode in tqdm(range(start, end)): #1e
            _ =  self.fit(memory)

            if episode % self.target_update_frequency == 0:
                self.update_target_network()


          
    # def train(self, env, memory, start, end, verbose=False):
    #     print(f"Training from episode {start} to episode {end}")
    #     episode_rewards = []
    #     episode_losses = []
    #     episode_timesteps = []
    #     end_training = False

    #     for episode in tqdm(range(start, end)): #1e
    #         state, _ = env.reset()
    #         rewards = 0
    #         losses = 0
    #         timestep = 0

    #         start = time()
    #         while True:
    #             timestep += 1
    #             action = self.predict(state)
    #             next_state, reward, terminated, truncated, _ = env.step(action)
    #             next_state = np.array(next_state)
    #             done = terminated or truncated

    #             memory.add(state, action, reward, next_state, done)

    #             rewards += reward

    #             state = next_state

    #             # done = True

    #             if done:
    #                 loss =  self.fit(memory)
    #                 self.decay_epsilon()

    #                 # print(f'Episode {episode}: {self.epsilon}')

    #                 episode_rewards.append(rewards)
    #                 episode_losses.append(losses/timestep)
    #                 episode_timesteps.append(timestep)
    #                 # print(f"Episode {episode+1}/{end}, rewards: {rewards}")

    #                 if (episode + 1) % self.target_update_frequency == 0:
    #                     self.update_target_network()

    #                 #early stopping
    #                 if np.array(episode_timesteps[-100:]).mean() >= 475:
    #                     end_training = True

    #                 # print(f"Time epoch: {time() - start}")

    #                 break
            
    #         if end_training:
    #             break

    #     if verbose:
    #         import matplotlib.pyplot as plt
            
    #         plt.figure(figsize=(12, 6))

    #         plt.subplot(1, 2, 1)
    #         plt.plot(episode_rewards)
    #         plt.title("Rewards")
    #         plt.xlabel("Episode")
    #         plt.ylabel("Reward")

    #         # Plotting losses
    #         plt.subplot(1, 2, 2)
    #         plt.plot(episode_losses)
    #         plt.title("Losses")
    #         plt.xlabel("Episode")
    #         plt.ylabel("Loss")

    #         plt.tight_layout()
    #         # plt.show()
    #         plt.savefig(f'logs/rewards_losses_{time()}.png')

    #         np.save('rewards.npy', episode_rewards)


    def validate(self,env):
        rewards = 0
        
        state, _ = env.reset()

        while True:
            action = self.inference(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state)
            done = terminated or truncated

            state = next_state

            rewards += reward

            if done:
                break

        return rewards

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
