import numpy as np
from collections import defaultdict
from copy import deepcopy
from scipy.stats import bernoulli
from math import log
import seaborn as sns
colors = sns.color_palette('colorblind')
import random
import matplotlib
import matplotlib.pyplot as plt


class Environment:
    """Generic environment.
    Reward only depends on the next state.
    """
    def __init__(self):
        self.state = self.init_state()
        
    def init_state(self):
        state = None
        return state
    
    def reinit_state(self):
        self.state = self.init_state() 
        
    @staticmethod
    def get_actions(state):
        actions = [None]
        return actions
    
    @staticmethod
    def get_states():
        raise ValueError("Not available. The state space might be too large.")
        
    @staticmethod
    def get_transition(state, action):
        probs = [1]
        states = [deepcopy(state)]
        return probs, states
        
    @staticmethod
    def get_reward(state):
        return None
    
    @staticmethod
    def is_terminal(state):
        return True
    
    @staticmethod
    def encode(state):
        return state
    
    @staticmethod
    def decode(state):
        return state
    
    @staticmethod
    def get_model(state, action):
        probs, states = Environment.get_transition(state, action)
        rewards = [Environment.get_reward(state) for state in states]
        return probs, states, rewards
    
    def step(self, action):
        """Apply action, get reward and modify state.
        Returns reward, stop (``True`` if terminal state).
        """
        reward = None
        stop = True
        if action is not None and action in self.get_actions(self.state):
            probs, states, rewards = self.get_model(self.state, action)
            i = np.random.choice(len(probs), p=probs)
            state = states[i]
            self.state = state
            reward = rewards[i]
            stop = self.is_terminal(state)
        return reward, stop
    
