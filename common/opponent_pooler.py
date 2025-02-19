import numpy as np
import hockey.hockey_env as h_env

class OpponentPooler:
    def __init__(self, weak_prob, strong_prob, self_prob, self_opponent):
        '''
            Args:
                weak_prob: the probability of sampling a weak opponent
        '''
        self.steps = 0
        self.weak_prob = weak_prob
        self.strong_prob = strong_prob
        self.self_prob = self_prob

        self.strong_opponent = h_env.BasicOpponent(weak=False)
        self.weak_opponent = h_env.BasicOpponent(weak=True)
        self.self_opponent = self_opponent
        self.self_opponent.eval()

        self.probabilities = [self.weak_prob, self.strong_prob, self.self_prob]
        
    def sample_opponent(self):
        return np.random.choice([self.weak_opponent, self.strong_opponent, self.self_opponent], 
                                p=self.probabilities)

    def get_opponent(self, opponent_type):
        return next((opponent for opponent in self.opponents if opponent[0] == opponent_type), None)
    
    def update_self_opponent(self, self_opponent):
        self.self_opponent = self_opponent
        self.self_opponent.eval()

