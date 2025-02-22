import numpy as np
import torch
import hockey.hockey_env as h_env

from omegaconf.listconfig import ListConfig

class OpponentPooler:
    def __init__(self, weak_prob, strong_prob, self_prob, max_episodes, self_opponent, custom_prob, custom_opponents, **kwargs):
        '''
            Args:
                weak_prob: the probability of sampling a weak opponent
        '''
        self.steps = 0
        self.weak_prob = weak_prob
        self.strong_prob = strong_prob
        self.self_prob = self_prob
        self.max_episodes = max_episodes
        self.custom_prob = custom_prob
        self.custom_opponents = custom_opponents
        self.step_size = 0

        if isinstance(weak_prob, (list, ListConfig)):
            # if list, we have a schedule of probabilities
            number_of_probabilities = len(weak_prob)
            self.step_size = max_episodes // number_of_probabilities
            self.probabilities = [[weak_prob[i], strong_prob[i], self_prob[i], custom_prob[i]] for i in range(len(weak_prob))]
        else:
            self.probabilities = [weak_prob, strong_prob, self_prob, custom_prob]

        self.strong_opponent = h_env.BasicOpponent(weak=False)
        self.weak_opponent = h_env.BasicOpponent(weak=True)
        self.self_opponent = self_opponent
        self.self_opponent.eval()

        for opponent in custom_opponents:
            opponent.eval()

    def sample_opponent(self):
        self.steps += 1
        choice = np.random.choice(['weak', 'strong', 'self', 'custom'], 
                                p=self.get_current_probabilities())
        if choice == 'weak':
            return self.weak_opponent
        elif choice == 'strong':
            return self.strong_opponent
        elif choice == 'self':
            return self.self_opponent
        elif choice == 'custom':
            return np.random.choice(self.custom_opponents)
    
    def update_self_opponent(self, self_opponent):
        self.self_opponent = self_opponent
        self.self_opponent.eval()

    def get_current_probabilities(self):
        if self.step_size > 0:
            return  self.probabilities[min(self.steps // self.step_size, len(self.probabilities) - 1)]
        return self.probabilities
