import torch
import torch.nn as nn
import torch.nn.functional as F

def make_mlp(input_dim, hidden_dims, output_dim, activation=nn.ReLU):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dims[0]),
        activation(),
        *[layer for i in range(len(hidden_dims)-1) for layer in (nn.Linear(hidden_dims[i], hidden_dims[i+1]), activation())],
        nn.Linear(hidden_dims[-1], output_dim)
    )

def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

class Actor(nn.Module):
    def __init__(self, obs_dim, hidden_dims, action_dim, action_space):
        super(Actor, self).__init__()
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim
        self.feedforward = make_mlp(obs_dim, hidden_dims[:-1], hidden_dims[-1])
        self.mu_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)

        self.action_high = torch.tensor(action_space.high)
        self.action_low = torch.tensor(action_space.low)
        self.scale = (self.action_high - self.action_low) / 2
        self.offset = (self.action_high + self.action_low) / 2

        init_weights(self)

    def forward(self, state):
        x = self.feedforward(state)
        x = F.relu(x)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=10)
        return mu, log_std
    
    def sample(self, state, deterministic=False):
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)

        if not deterministic:
            x = dist.rsample()
            normalized_action = torch.tanh(x)
            unscale_action = normalized_action * self.scale + self.offset

            # Compute log probability   
            log_prob = dist.log_prob(x)
            log_prob -= torch.log(self.scale * (1 - normalized_action.pow(2)) + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:
            unscale_action = torch.tanh(mu) * self.scale + self.offset
            log_prob = torch.tensor(0.0)

        return unscale_action, log_prob
        

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dims, action_dim):
        super(Critic, self).__init__()
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim

        self.c1 = make_mlp(obs_dim + action_dim, hidden_dims, 1)
        self.c2 = make_mlp(obs_dim + action_dim, hidden_dims, 1)

        init_weights(self)

    def forward(self, x, a):
        return self.c1(torch.cat([x, a], dim=-1)), self.c2(torch.cat([x, a], dim=-1))

    def ema_update(self, other, tau):
        for param, other_param in zip(self.parameters(), other.parameters()):
            param.data.copy_(param.data * tau + other_param.data * (1 - tau))

    def hard_copy(self, other):
        for param, other_param in zip(self.parameters(), other.parameters()):
            param.data.copy_(other_param.data)



