from pink import PinkActionNoise

class PinkNoise:
    def __init__(self, action_dim, seq_len, max_episodes, sigma=0.3, n_annealing_steps=10, complete_annealing_by=0.7):
        self.initial_sigma = sigma
        self.seq_len = seq_len
        self.action_dim = action_dim
        self.noise = PinkActionNoise(self.initial_sigma, seq_len, action_dim)
        self.max_episodes = int(max_episodes * complete_annealing_by)                     # slowly anneal the noise to zero, at 80% of the max_episodes
        self.step_size = self.max_episodes // n_annealing_steps
        self.steps = 0

    def reset(self):
        self.steps += 1
        self.noise.reset()

        if self.steps <= self.max_episodes and self.steps % self.step_size == 0:
            new_sigma = self.initial_sigma * (1 - self.steps / self.max_episodes)
            new_sigma = max(0, new_sigma)                           # minimum noise level
            self.noise = PinkActionNoise(new_sigma, self.seq_len, self.action_dim)

    def __call__(self):
        return self.noise()