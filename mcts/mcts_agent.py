import sys
sys.path.append('/')

import torch
from copy import deepcopy

from mcts.node import Node
from dqn.dqn_agent import DeepQNetworkAgent as Agent


class MCTS():
    def __init__(self, branching_factor, search_steps, train_episodes,val_episodes, c_value, memory, env) -> None:
        self.branching_factor = branching_factor
        self.search_steps = search_steps
        self.train_episodes = train_episodes
        self.val_episodes = val_episodes
        self.c_param = c_value
        self.memory = memory
        self.env = env

    def _init_search(self, agent):
        self.root = Node(0, 1, parent=None)

        torch.save(agent.model, self.root.main_path)
        torch.save(agent.model, self.root.target_path)

        self.memory.fill(agent)

        self.discount_factor = agent.discount_factor
        self.epsilon_decay = agent.epsilon_decay
        self.target_update_frequency = agent.target_update_frequency


    def search(self, agent):
        self._init_search(agent)

        for i in range(self.search_steps):
            print(f"Step {i+1}/{self.search_steps}")
            v = self.tree_policy()
            reward = self.rollout(v)
            v.backpropagate(reward)
        
        root = deepcopy(self.root)

        def best_branch(node):
            if len(node.childrens) == 0:
                return
            
            best_child = node.best_child(self.c_param)
            node.childrens = [best_child]

            return best_branch(best_child)
        
        best_branch(root)
            
        return root
    
    def tree_policy(self):
        current_node = self.root

        while not self.is_terminal_node(current_node):
            if not self.is_fully_expanded(current_node):
                return self.expand(current_node)
            else:
                current_node = current_node.best_child(self.c_param)

        return current_node

    def is_terminal_node(self, node: Node):
        return node.epoch == self.train_episodes

    def is_fully_expanded(self, node: Node):
        return True if len(node.childrens) == self.branching_factor else False

    def expand(self, node):
        model = torch.load(node.main_path)
        target_model = torch.load(node.target_path)

        epsilon = node.epsilon*self.epsilon_decay

        agent = Agent(model, target_model, epsilon, self.discount_factor, self.epsilon_decay, self.target_update_frequency)

        agent.fit(self.memory)

        core_reward = agent.validate(self.env)

        print(f"Core reward: {core_reward}")

        child_node = Node(node.epoch + 1, epsilon, core_reward=core_reward, parent=node)
        node.childrens.append(child_node)

        torch.save(agent.model, child_node.main_path)

        if (child_node.epoch) % self.target_update_frequency == 0:
            torch.save(agent.model, child_node.target_path)
        else:
            torch.save(agent.target_model, child_node.target_path)

        return child_node
    
    # def expand(self, node):
    #     model = torch.load(node.main_path)
    #     target_model = torch.load(node.target_path)

    #     child_node = Node(node.epoch + 1, node.epsilon*self.epsilon_decay, parent=node)
    #     node.childrens.append(child_node)

    #     agent = Agent(model, target_model, child_node.epsilon, self.discount_factor, self.epsilon_decay, self.target_update_frequency)

    #     agent.fit(self.memory)

    #     torch.save(agent.model, child_node.main_path)

    #     if (child_node.epoch) % self.target_update_frequency == 0:
    #         torch.save(agent.model, child_node.target_path)
    #     else:
    #         torch.save(agent.target_model, child_node.target_path)

    #     return child_node
    
    
    def rollout(self, node):
        model = torch.load(node.main_path)
        target_model = torch.load(node.target_path)

        agent = Agent(model, target_model, node.epsilon, self.discount_factor, self.epsilon_decay, self.target_update_frequency)

        # agent.train(self.env, self.memory, node.epoch, self.train_episodes)

        val_reward = agent.validate(self.env)

        print(node.epoch, node.version, val_reward)

        return val_reward