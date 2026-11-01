import numpy as np
from copy import deepcopy
from scipy.stats import bernoulli



class Agent:
    """Agent. Default policy is purely random."""
    def __init__(self, environment, policy=None, n_steps=100):
        if policy is not None:
            self.policy = policy
        else:
            self.policy = self.random_policy
        self.environment = environment
        self.n_steps = n_steps
        
    def random_policy(self, state):
        actions = self.environment.get_actions(state)
        if len(actions):
            probs = np.ones(len(actions)) / len(actions)
        else:
            probs = [1]
            actions = None
        return probs, actions
    
    def get_action(self, state):
        action = None
        probs, actions = self.policy(state)
        if len(actions):
            i = np.random.choice(len(actions), p=probs)
            action = actions[i]
        return action
    
    def get_episode(self, n_steps=None):
        """Get the states and rewards for an episode."""
        self.environment.reinit_state()
        if n_steps is None:
            n_steps = self.n_steps
        state = deepcopy(self.environment.state)
        states = [state] # add the initial state
        rewards = [0]
        for t in range(n_steps):
            action = self.get_action(state)
            reward, stop = self.environment.step(action)
            state = deepcopy(self.environment.state)
            states.append(state)
            rewards.append(reward)
            if stop:
                break
        return stop, states, rewards