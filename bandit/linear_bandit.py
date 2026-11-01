import numpy as np
from .base import Environment

class LinearBandit(Environment):
    """
    (Contextual) Linear Bandit instance
    
    :param theta: unknown d-dimensional parameter of linear rewards
    :param K: number of arms (feature vectors) per round
    :param n_contexts: total number of possible contexts (different feature vectors per arm per round)
    :param var: standard deviation of Gaussian noise
    """
    
    def __init__(self, theta, K, n_contexts, var=1.):
        
        self.d = np.size(theta)
        self.theta = theta
        self.K = K
        self.var = var
        self.n_contexts = n_contexts

        self.features = np.random.multivariate_normal(np.zeros(self.d), np.eye(self.d), size=(self.n_contexts, K))
        norms = np.linalg.norm(self.features, axis=2)
        self.features = self.features / norms[:, :, np.newaxis]
        current_context_id = np.random.randint(self.n_contexts) 
        self.current_action_set = self.features[current_context_id, :, :]
        
    def get_action_set(self):
        """Returns the set of available arms (feature vectors) for the current round"""
        current_context_id = np.random.randint(self.n_contexts)
        self.current_action_set = self.features[current_context_id, :, :]
        return self.current_action_set
    
    def get_reward(self, action):
        """
        Sample a random reward

        :param action: the arm (d-dimensional feature vector) chosen by the learner)
        """
        mean = np.dot(action, self.theta)
        return np.random.normal(mean, scale=self.var)
    
    def get_means(self):
        """Returns the mean rewards of all the available arms"""
        return np.dot(self.current_action_set, self.theta)

    def name(self):
        return "LinearBandit"