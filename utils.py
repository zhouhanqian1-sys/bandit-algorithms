
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
from torch.utils.data import Dataset






def play_bandit(environment, agent, N, T):
    """
    Play N independent runs of length T for the specified agent.

    :param environment: a linear bandit instance
    :param agent: a contextual bandit algorithm
    :param N: number of independent simulations
    :param T: decision horizon
    :return: the agent's name, and the collected data in an ndarray
    """

    data = np.zeros((N, T))

    for n in range(N):
        agent.reset()
        for t in range(T):
            action_set = environment.get_action_set()
            action_index, action = agent.get_action(action_set)
            # Note that, differently from the previous part, now get_action needs to receive the action_set
            if environment.name() == 'UCIBandit':
                reward = environment.get_reward(action_index)
            else:
                reward = environment.get_reward(action)
            # Obtain the context (feature vector) corresponding to the chosen action
            context = action_set[action_index]
            # Pass the chosen arm, its context and the observed reward to the agent
            agent.receive_reward(action, context, reward)

            # Compute instant (pseudo) regret (environment-agnostic)
            means = environment.get_means()
            best_reward = np.max(means)
            # Identify the expected reward of the chosen arm regardless of how the agent encodes it

            chosen_mean = means[action_index]   
            data[n,t]= best_reward - chosen_mean  # pseudo regret

    return agent.name(), data


def run_experiment(environment, agents, N, T):
    """
    Play N trajectories for all agents over a horizon T. Store data in a dictionary.

    :param environment: a linear bandit instance
    :param agent: a list of contextual bandit algorithms to compare
    :param N: number of independent simulations
    :param T: decision horizon
    :return: the pseudo regret for each agent in a dictionary indexed by the agent's name
    """

    all_data = {}

    for agent in agents:
        agent_id, regrets = play_bandit(environment, agent, N, T)

        all_data[agent_id] = regrets

    return all_data

def plot_result(regrets, logscale=False, lb=None,q=10, mode = 'regret', cumulative=True):
    """
    regrets must be a dict {'agent_id':regret_table}
    """
    
    reg_plot = plt.figure()
    #compute useful stats
#     regret_stats = {}
    for i, agent_id in enumerate(regrets.keys()):
        data = regrets[agent_id]
        N, T = data.shape
        if cumulative:
            cumdata = np.cumsum(data, axis=1) # cumulative regret
        else:
            cumdata = data
        
        mean_reg = np.mean(cumdata, axis=0)
        q_reg = np.percentile(cumdata, q, axis=0)
        Q_reg = np.percentile(cumdata, 100-q, axis=0)
        
#         regret_stats[agent_id] = np.array(mean_reg, q_reg, Q_reg)
        
        plt.plot(np.arange(T), mean_reg, color=colors[i], label=agent_id)
        plt.fill_between(np.arange(T), q_reg, Q_reg, color=colors[i], alpha=0.2)
        
    if logscale:
        plt.xscale('log')
        plt.xlim(left=100)

    if lb is not None: 
        plt.plot(np.arange(T), lb, color='black', marker='*', markevery=int(T/10))
        
    plt.xlabel('time steps')
    header = "Cumulative " if cumulative else ""
    plt.ylabel(header + mode.capitalize())
    plt.legend()
    #reg_plot.show()
    return reg_plot #notebook version





class Collector(Dataset):
    '''
    Collect the context vectors that have appeared 
    '''

    def __init__(self):
        super(Collector, self).__init__()
        self.context = []
        self.rewards = []
        self.chosen_arms = []

    def __getitem__(self, key):
        return self.context[key], self.rewards[key]

    def __len__(self):
        return len(self.rewards)

    def collect_data(self, arm, context, reward):
        self.context.append(context.cpu())
        self.chosen_arms.append(arm)
        self.rewards.append(reward)

    def fetch_batch(self, batch_size=None):
        if batch_size is None or batch_size > len(self.rewards):
            return self.context, self.rewards
        else:
            offset = np.random.randint(0, len(self.rewards) - batch_size)
            return self.context[offset:offset + batch_size], self.rewards[offset: offset + batch_size]

    def clear(self):
        self.context = []
        self.rewards = []
        self.chosen_arms = []




def sample_data(dataloader):
    while True:
        for batch in dataloader:
            yield batch




def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("state value:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()

    print("policy:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("State value:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # Print with width 6 for better alignment
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()

    print("Policy:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # Goal state
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()