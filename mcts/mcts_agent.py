import sys
sys.path.append('/')

import torch
from copy import deepcopy
import matplotlib.pyplot as plt

from mcts.node import Node
from dqn.dqn_agent import DeepQNetworkAgent as Agent


class MCTS():
    def __init__(self, branching_factor, train_episodes,window, c_value, memory, env) -> None:
        self.branching_factor = branching_factor
        self.train_episodes = train_episodes
        self.c_param = c_value
        self.memory = memory
        self.env = env
        self.window = window

    def _init_search(self, agent):
        self._base_dqn_model = agent.model
        self._base_dqn_target_model = agent.target_model

        self.discount_factor = agent.discount_factor
        self.epsilon_decay = agent.epsilon_decay
        self.target_update_frequency = agent.target_update_frequency

        self.memory.fill(agent)

        agent.fit(self.memory)

        core_reward = agent.validate(self.env)

        self.root = Node(0, 1, core_reward=core_reward, parent=None)

        torch.save(agent.model.state_dict(), self.root.main_path)
        torch.save(agent.model.state_dict(), self.root.target_path)


    def search(self, agent, visualise=False):
        self._init_search(agent)
        self.do_search = True

        while True:
            v = self.tree_policy()

            if not self.do_search:
                break
            reward = self.rollout(v)

            if visualise:
                self.best_branch_visualisation(self.root)

            v.backpropagate(reward)
        
        return self.root
    

    def best_branch_visualisation(self, node):
        
        best = self.best_branch(node)

        start = False

        if len(best.childrens) > 0:
            self.best_branch_visualisation(best.childrens[0])
        else:
            start = True

        rewards = []
        start = best

        while len(start.childrens) > 0:
            rewards.append(start.core_reward)
            start = start.childrens[0]

        plt.plot(rewards)
        plt.ylim(0, 200)
        plt.xlim(0,self.train_episodes)
        plt.pause(0.0001)
        plt.cla()
        plt.show(block=False)
  

    def best_branch(self, node):

        best = deepcopy(node)

        def _best_branch(node):
            if len(node.childrens) == 0:
                return
            
            best_child = node.best_child(self.c_param)
            if best_child is None:
                return
            
            node.childrens = [best_child]

            return _best_branch(best_child)
        
        _best_branch(best)

        return best
    
    def tree_policy(self):
        current_node = self.root

        while True:
            terminal = self.is_terminal_node(current_node)
            if terminal:
                self.do_search = False
                break
            else:
                if not self.is_fully_expanded(current_node):
                    return self.expand(current_node)
                else:
                    current_node = current_node.best_child(self.c_param)

        return current_node
    

    def is_terminal_node(self, node: Node):
        return node.epoch == self.train_episodes-1

    def is_fully_expanded(self, node: Node):
        return True if len(node.childrens) == self.branching_factor else False

    def expand(self, node):
        self._base_dqn_model.load_state_dict(torch.load(node.main_path))
        self._base_dqn_target_model.load_state_dict(torch.load(node.target_path))

        epsilon = node.epsilon*self.epsilon_decay

        agent = Agent(self._base_dqn_model, self._base_dqn_target_model, epsilon, self.discount_factor, self.epsilon_decay, self.target_update_frequency)
        
        # for _ in range(5):
        #     agent.fit(self.memory)

        agent.fit(self.memory)

        # core_reward = agent.validate(self.env)
        core_reward = agent.validate_random(self.env)

        child_node = Node(node.epoch + 1, epsilon, core_reward=core_reward, parent=node)
        node.childrens.append(child_node)

        torch.save(agent.model.state_dict(), child_node.main_path)

        if (child_node.epoch) % self.target_update_frequency == 0:
            torch.save(agent.model.state_dict(), child_node.target_path)
        else:
            torch.save(agent.target_model.state_dict(), child_node.target_path)

        return child_node
    
    
    def rollout(self, node):
        self._base_dqn_model.load_state_dict(torch.load(node.main_path))
        self._base_dqn_target_model.load_state_dict(torch.load(node.target_path))

        agent = Agent(self._base_dqn_model, self._base_dqn_target_model, node.epsilon, self.discount_factor, self.epsilon_decay, self.target_update_frequency)

        if node.epoch+self.window < self.train_episodes:
            agent.train_no_interaction(self.env, self.memory, node.epoch, node.epoch+self.window)
        else:
            # agent.train_no_interaction(self.env, self.memory, node.epoch, self.train_episodes-(node.epoch+self.window))
            agent.train_no_interaction(self.env, self.memory, node.epoch,self.window)
        # agent.train_no_interaction(self.env, self.memory, node.epoch, self.train_episodes)

        # val_reward = agent.validate(self.env)
        val_reward = agent.validate_random(self.env)

        print(f'e: {node.epoch}, v: {node.version}, core_r: {node.core_reward}, val_r: {val_reward}')

        return val_reward