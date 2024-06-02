import sys
sys.path.append('/')

import torch
from copy import deepcopy

from mcts.node import Node
from dqn.dqn_agent import DeepQNetworkAgent as Agent


class MCTS():
    def __init__(self, branching_factor, train_episodes, c_value, memory, env) -> None:
        self.branching_factor = branching_factor
        self.train_episodes = train_episodes
        self.c_param = c_value
        self.memory = memory
        self.env = env

    def _init_search(self, agent):
        self._base_dqn_model = agent.model
        self._base_dqn_target_model = agent.target_model

        self.discount_factor = agent.discount_factor
        self.epsilon_decay = agent.epsilon_decay
        self.target_update_frequency = agent.target_update_frequency

        self.memory.fill(agent)

        agent.fit(self.memory)

        core_reward = agent.validate(self.env)

        self.root = Node(0, 1, core_reward=core_reward, parent=None) #TODO is it good to give core reward to root node?

        torch.save(agent.model.state_dict(), self.root.main_path)
        torch.save(agent.model.state_dict(), self.root.target_path)


    def search(self, agent):
        self._init_search(agent)
        self.do_search = True

        # for i in range(self.search_steps):
        while True:
            # print(f"Step {i+1}/{self.search_steps}")
            v = self.tree_policy()
            if not self.do_search:
                break
            reward = self.rollout(v)
            v.backpropagate(reward) #1f
        
        return self.root

    def best_branch(self, node):

        best = deepcopy(node)

        def _best_branch(node):
            if len(node.childrens) == 0:
                return
            
            best_child = node.best_child(self.c_param)
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
        return node.epoch == self.train_episodes-1 #1h

    def is_fully_expanded(self, node: Node): #1b
        return True if len(node.childrens) == self.branching_factor else False

    def expand(self, node):
        self._base_dqn_model.load_state_dict(torch.load(node.main_path))
        self._base_dqn_target_model.load_state_dict(torch.load(node.target_path))

        epsilon = node.epsilon*self.epsilon_decay

        agent = Agent(self._base_dqn_model, self._base_dqn_target_model, epsilon, self.discount_factor, self.epsilon_decay, self.target_update_frequency)

        agent.fit(self.memory) #1d

        core_reward = agent.validate(self.env)

        # child_node = Node(node.epoch + 1, epsilon, core_reward=None, parent=node)
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

        #2a comment out the line below
        agent.train(self.env, self.memory, node.epoch, self.train_episodes) #1e

        val_reward = agent.validate(self.env) #1f

        # if node.core_reward is None: #1f
        #     node.core_reward = val_reward

        print(f'e: {node.epoch}, v: {node.version}, core_r: {node.core_reward}, val_r: {val_reward}')

        return val_reward