import random
import torch
import numpy as np
from collections import defaultdict

class ReplayBuffer:
    def __init__(self, max_size=1_000_000):
        self.max_size = max_size
        self.buffer = []
        self.current_idx = 0

    def add(self, obs, next_state, action, reward, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append((obs, next_state, action, reward, done))
        else:
            self.buffer[self.current_idx] = (obs, next_state, action, reward, done)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch_size):
        batch_obs = []
        batch_next_state = []
        batch_action = []
        batch_reward = []
        batch_done = []

        if len(self.buffer) < batch_size:
            return []
        
        indices = np.random.choice(self.size(), size=batch_size, replace=False)
        for index in indices:
            batch_obs.append(self.buffer[index][0])
            batch_next_state.append(self.buffer[index][1])
            batch_action.append(self.buffer[index][2])
            batch_reward.append(self.buffer[index][3])
            batch_done.append(self.buffer[index][4])

        # make numpy arrays
        batch_obs = np.array(batch_obs)
        batch_next_state = np.array(batch_next_state)
        batch_action = np.array(batch_action)
        batch_reward = np.array(batch_reward)
        batch_done = np.array(batch_done)

        return batch_obs, batch_next_state, batch_action, batch_reward, batch_done

    def size(self):
        return len(self.buffer)


class ReplayBufferTorch:
    def __init__ (self, max_size=1_000_000, device='cpu'):
        self.max_size = max_size
        self.current_idx = 0
        self.buffers = defaultdict(lambda: None)
        self.keys = ['obs', 'next_state', 'action', 'reward', 'done']
        self.device = device

    def add(self, obs, next_state, action, reward, done):
        for key, value in zip(self.keys, [obs, next_state, action, reward, done]):
            if isinstance(value, float):
                if self.buffers[key] is None:
                    self.buffers[key] = torch.zeros((self.max_size, 1)).to(self.device)
                self.buffers[key][self.current_idx] = value
            elif isinstance(value, bool):
                if self.buffers[key] is None:
                    self.buffers[key] = torch.zeros((self.max_size, 1), dtype=torch.uint8).to(self.device)
                self.buffers[key][self.current_idx] = value
            else:
                if self.buffers[key] is None:
                    self.buffers[key] = torch.zeros((self.max_size,) + value.shape).to(self.device)
                self.buffers[key][self.current_idx] = torch.from_numpy(value).to(self.device)

        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch_size):
        if self.size() < batch_size:
            return []
        
        indices = np.random.choice(self.size(), size=batch_size, replace=False)
        
        return [self.buffers[key][indices] for key in self.keys]

    def size(self):
        return self.current_idx
