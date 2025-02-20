import numpy as np
import hockey.hockey_env as h_env

from omegaconf.listconfig import ListConfig

class OpponentPooler:
    def __init__(self, weak_prob, strong_prob, self_prob, max_episodes, self_opponent, **kwargs):
        '''
            Args:
                weak_prob: the probability of sampling a weak opponent
        '''
        self.steps = 0
        self.weak_prob = weak_prob
        self.strong_prob = strong_prob
        self.self_prob = self_prob
        self.max_episodes = max_episodes
        self.step_size = 0

        if isinstance(weak_prob, (list, ListConfig)):
            # if list, we have a schedule of probabilities
            number_of_probabilities = len(weak_prob)
            self.step_size = max_episodes // number_of_probabilities
            self.probabilities = [[weak_prob[i], strong_prob[i], self_prob[i]] for i in range(len(weak_prob))]
        else:
            self.probabilities = [weak_prob, strong_prob, self_prob]

        self.strong_opponent = h_env.BasicOpponent(weak=False)
        self.weak_opponent = h_env.BasicOpponent(weak=True)
        self.self_opponent = self_opponent
        self.self_opponent.eval()

    def sample_opponent(self):
        self.steps += 1
        return np.random.choice([self.weak_opponent, self.strong_opponent, self.self_opponent], 
                                p=self.get_current_probabilities())

    def get_opponent(self, opponent_type):
        return next((opponent for opponent in self.opponents if opponent[0] == opponent_type), None)
    
    def update_self_opponent(self, self_opponent):
        self.self_opponent = self_opponent
        self.self_opponent.eval()

    def get_current_probabilities(self):
        if self.step_size > 0:
            return  self.probabilities[min(self.steps // self.step_size, len(self.probabilities) - 1)]
        return self.probabilities
