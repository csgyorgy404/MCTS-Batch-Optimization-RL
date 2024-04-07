import torch
import random
import numpy as np
from tqdm import tqdm
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
        if self.epsilon < random.random():
            self.model.eval()
            with torch.no_grad():
                actions = self.model(torch.reshape(torch.tensor(state, dtype=torch.float32),
                                                                 shape=(1, self.model.in_features)))
                                                                #  shape=(1, self.model.in_features))).detach().cpu().numpy()
                p = int(np.argmax(actions))
            self.model.train()
        else:
            p =  np.random.randint(0, self.model.out_features, 1)[0]

        return p

    def inference(self, state: np.ndarray) -> int:
        with torch.no_grad():
            actions = self.model(state).detach().numpy()
        return np.argmax(actions)[0]

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def fit(self, memory):
        states, actions, rewards, next_states, terminals = memory.sample()

        predicted_q = self.model(states)
        predicted_q = torch.gather(predicted_q, 1, actions.view((-1, 1))).squeeze()

        self.target_model.eval()
        with torch.no_grad():
            target_q = self.target_model(torch.squeeze(next_states))
            target_q = target_q.max(dim=-1)[0]
        target_value = rewards + self.discount_factor * target_q * (1 - terminals)
        self.target_model.train()

        loss = F.mse_loss(predicted_q, target_value)

        self.model.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.model.optimizer.step()

        # For passing through the update information of the memory
        # with torch.no_grad():
        #     argmax_labels = predicted_q.max(dim=-1)[0].detach().cpu().numpy()
        #     td_error = target_value - predicted_q
        # self.update_memory(td_errors=td_error, labels=argmax_labels)

        # return loss

    # def update_memory(self, **kwargs):
    #     self.memory.update_training_sample_weights(**kwargs)
        
    def train(self, env, memory, start, end, verbose=False):
        print(f"Training from episode {start} to episode {end}")
        episode_rewards = []
        for episode in tqdm(range(start, end)):
            state, _ = env.reset()
            rewards = 0
            while True:
                action = self.predict(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = np.array(next_state)
                done = terminated or truncated

                rewards += reward

                state = next_state
     
                self.fit(memory)

                if done:
                    self.decay_epsilon()

                    if (episode + 1) % self.target_update_frequency == 0:
                        self.update_target_network()

                    break
            if verbose:
                episode_rewards.append(rewards)
                print(f"Episode {episode+1}/{end}, rewards: {rewards}")

        if verbose:
            import matplotlib.pyplot as plt
            
            plt.plot(episode_rewards)
            plt.title("Rewards")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.show()


    def validate(self,env):
        # rewards = []
        rewards = 0
        
        state, _ = env.reset()

        while True:
            action = self.predict(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state)
            done = terminated or truncated

            state = next_state

            rewards += reward

            if done:
                break

        print(f"Validation rewards: {rewards}")

        return rewards

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
