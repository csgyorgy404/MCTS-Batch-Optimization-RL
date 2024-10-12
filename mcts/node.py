import os
import datetime
import numpy as np
from uuid import uuid4

class Node:
    def __init__(self, epoch, epsilon,  core_reward=None , parent=None):
        self._visits = 0
        self._value = 0.0
        self._core_reward = core_reward
        self.parent = parent
        self.childrens = []
        self.epoch = epoch
        self.epsilon = epsilon
        self.version = str(uuid4())
        
        self._init_paths(str(epoch), self.version)

    @property
    def core_reward(self):
        return self._core_reward #1g
    
    @core_reward.setter
    def core_reward(self, value):
        self._core_reward = value

    @property
    def n(self):
        return self._visits
    
    @property
    def q(self):
        return self._value #1g
    

    def _init_paths(self, epoch, version):
        root_folder = 'nodes'

        main_path = os.path.join(root_folder, epoch, version)
        target_path = os.path.join(root_folder, epoch, version)

        os.makedirs(main_path, exist_ok=True)
        os.makedirs(target_path, exist_ok=True)

        self.main_path = os.path.join(main_path, 'main.pth')
        self.target_path = os.path.join(target_path, 'target.pth')

    def backpropagate(self, score):
        self._visits += 1
        self._value += score
        if self.parent:
            self.parent.backpropagate(score)

    def best_child(self, c_value=1.4):
        exploit = []
        explore = []

        for child in self.childrens:
            try:
                exploit.append((child.q + child.core_reward) / child.n)
                explore.append(c_value * np.sqrt(2 * np.log(self.n) / child.n))
            except:
                exploit.append(0)
                explore.append(np.inf)
            

        choices_weights = [i + j for i, j in zip(exploit, explore)]

        max_value = np.max(choices_weights)
        '''
        If multiple actions have the same weight, we select one of them randomly
        '''
        indexes = np.where(choices_weights == max_value)[0]

        try:
            return self.childrens[np.random.choice(indexes)]
        except:
            return None
        # return self.childrens[np.random.choice(indexes)]