import random
import torch
import numpy as np
from collections import defaultdict
from torchrl.data import PrioritizedReplayBuffer as PRB, ListStorage

class ReplayBuffer:
    def __init__(self, max_size=1_000_000):
        self.max_size = max_size
        self.buffer = []
        self.current_idx = 0

    def add(self, data):
        obs, next_state, action, reward, done = data
        if len(self.buffer) < self.max_size:
            self.buffer.append((obs, next_state, action, reward, done))
        else:
            self.buffer[self.current_idx] = (obs, next_state, action, reward, done)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch_size, **kwargs):
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

        return (batch_obs, batch_next_state, batch_action, batch_reward, batch_done), np.ones(batch_size)

    def size(self):
        return len(self.buffer)


class ReplayBufferTorch:
    def __init__ (self, max_size=1_000_000, device='cpu'):
        self.max_size = max_size
        self.current_idx = 0
        self.buffers = defaultdict(lambda: None)
        self.keys = ['obs', 'next_state', 'action', 'reward', 'done']
        self.device = device
        self.fully_filled = False

        if 'cuda' in device and not torch.cuda.is_available():
            self.device = 'cpu'
        elif 'mps' in device and not torch.backends.mps.is_available():
            self.device = 'cpu'

    def add(self, data):
        obs, next_state, action, reward, done = data
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

        if self.current_idx == 0:
            self.fully_filled = True

    def sample(self, batch_size, **kwargs):
        if self.size() < batch_size:
            return []
        
        indices = np.random.choice(self.size(), size=batch_size, replace=False)
        
        return [self.buffers[key][indices] for key in self.keys], torch.ones(batch_size).to(self.device)

    def size(self):
        if self.fully_filled:
            return self.max_size
        return self.current_idx

# A wrapper class, to handle numpy arrays, and beta annealing for prioritized replay buffer
class PrioritizedReplayBuffer:
    def __init__(self, max_episodes, max_size=1_000_000, alpha=0.6, beta=0.4, device='cpu'):
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.buffer = PRB(alpha=alpha, beta=beta, storage=ListStorage(max_size))

        if 'cuda' in device and not torch.cuda.is_available():
            self.device = 'cpu'
        elif 'mps' in device and not torch.backends.mps.is_available():
            self.device = 'cpu'

        assert beta <= 1, "Beta should be less than or equal to 1"

        self.beta_anneal = (1 - beta) / max_episodes
        self.num_elements = 0

    def add(self, data):
        data_tensors = []
        for value in data:
            if isinstance(value, float):
                data_tensors.append(torch.tensor([value]).float().to(self.device))
            elif isinstance(value, bool):
                data_tensors.append(torch.tensor([value]).to(torch.uint8).to(self.device))
            else:
                data_tensors.append(torch.from_numpy(value).float().to(self.device))

        self.num_elements += 1
        self.buffer.add(data_tensors)
    
    def sample(self, batch_size, episode):
        self.buffer.sampler.beta = min(1, self.beta + self.beta_anneal * episode)
        return self.buffer.sample(batch_size, return_info=True)
    
    def update_priorities(self, indices, td_errors):
        self.buffer.update_priority(indices, td_errors)
    
    def size(self):
        return min(self.num_elements, self.max_size)
    
