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
        self.target_model.eval()
        
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.epsilon_decay = epsilon_decay
        self.target_update_frequency = target_update_frequency
        

    def train_predict(self, state: np.ndarray) -> int:
        if self.epsilon > random.random():
            p =  np.random.randint(0, self.model.out_features, 1)[0]
          
        else:
            p = self.inference_predict(state)

        return p

    def inference_predict(self, state: np.ndarray):
        self.model.eval()

        with torch.no_grad():
            actions = self.model(torch.reshape(torch.tensor(state, dtype=torch.float32), shape=(1, self.model.in_features)))
            p = int(torch.argmax(actions))

        self.model.train()

        return p

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def fit(self, memory):

        states, actions, rewards, next_states, terminals = memory.sample()

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
  
        
    def train_no_interaction(self, env, memory, start, end, verbose=False):
        print(f"Training from episode {start} to episode {end}")

        for episode in tqdm(range(start, end)):
            self.fit(memory)

            if episode % self.target_update_frequency == 0:
                self.update_target_network()


          
    def train_with_interaction(self, env, memory, start, end, verbose=False):
        print(f"Training from episode {start} to episode {end}")
        episode_rewards = []
        end_training = False

        for episode in tqdm(range(start, end)):
            state = env.reset()
            rewards = 0

            steps = 0
            while True:
                action = self.train_predict(state)
                next_state, reward, done = env.step(action)

                # memory.add(state, action, reward, next_state, done)

                rewards += reward

                state = next_state

                if done:
                    self.fit(memory)
                    self.decay_epsilon()

                    # print('Validation reward:', self.validate(env))

                    episode_rewards.append(rewards)

                    if (episode + 1) % self.target_update_frequency == 0:
                        self.update_target_network()

                    break

                steps += 1
            
            if end_training:
                break

        if verbose:
            print(f"Episode {episode+1}/{end}, rewards: {rewards}, epsilon: {self.epsilon}")

            torch.save(self.model , 'model.pth')
            np.save('rewards.npy', episode_rewards)


    def validate(self,env):
        rewards = 0
        
        state = env.reset()

        while True:
            action = self.inference_predict(state)
            next_state, reward, done = env.step(action)

            env.render()

            state = next_state

            if reward == -100:
                pass

            rewards += reward

            if done:
                break

        return rewards

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
