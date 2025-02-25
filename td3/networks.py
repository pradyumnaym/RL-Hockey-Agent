import torch
import copy
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, L):
        super(PositionalEmbedding, self).__init__()
        self.L = L
        # Create frequency bands: 2^0, 2^1, ..., 2^(L-1)
        self.register_buffer("freq_bands", 2 ** torch.linspace(0, L - 1, steps=L))

    def forward(self, x):
        # x: (B, n)
        # Expand dimensions for broadcasting: (B, n, 1)
        x_expanded = x.unsqueeze(-1)
        # Multiply with frequency bands (scaled by pi)
        x_scaled = x_expanded * self.freq_bands * 3.141592653589793
        # Compute sin and cos embeddings: both are (B, n, L)
        emb_sin = torch.sin(x_scaled)
        emb_cos = torch.cos(x_scaled)
        # Concatenate along the last dimension to get (B, n, 2L)
        embedded = torch.cat([emb_sin, emb_cos], dim=-1)
        # Flatten last two dims to get (B, n * 2L)
        return embedded.view(x.shape[0], -1)

def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'selu':
        return nn.SELU()
    elif activation is None:
        return nn.Identity()
    else:
        raise ValueError("Invalid activation function")

class EMA(nn.Module):
    def __init__(self, model, tau=0.005):
        super(EMA, self).__init__()
        self.tau = tau
        # Create a deep copy of the model and set it to evaluation mode
        self.ema_model = copy.deepcopy(model)
        self._set_eval(self.ema_model)

    def _set_eval(self, model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(1 - self.tau).add_(model_p.data * self.tau)

    def forward(self, *args):
        return self.ema_model(*args)

class Actor(nn.Module):
    def __init__(self, obs_dim, action_space, hidden_dims, activation='relu', output_activation='tanh', embedding_dim=None):
        super(Actor, self).__init__()
        
        self.hidden_dims = hidden_dims
        self.action_space = action_space
        self.action_dim = action_space.shape[0]

        if embedding_dim is not None:
            self.embedding = PositionalEmbedding(embedding_dim)
            obs_dim = obs_dim * 2 * self.embedding.L

        # Build the network layers
        layers = [nn.Linear(obs_dim, hidden_dims[0]), get_activation(activation)]
        for dim_in, dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(get_activation(activation))

        layers.append(nn.Linear(hidden_dims[-1], self.action_dim))
        self.net = nn.Sequential(*layers)
        self.output_activation = get_activation(output_activation)

        # Set up action scaling based on the action_space bounds
        self.register_buffer("action_scale", torch.tensor((action_space.high - action_space.low) / 2., dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((action_space.high + action_space.low) / 2., dtype=torch.float32))

    def forward(self, obs):
        if hasattr(self, 'embedding'):
            obs = self.embedding(obs)
        x = self.net(obs)
        x = self.output_activation(x)
        return x * self.action_scale + self.action_bias
    
    def scale_action(self, action):
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0
    
    def unscale_action(self, scaled_action):
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))
        

class Critic(nn.Module):
    def __init__(self, obs_dim, action_space, hidden_dims, activation='relu', embedding_dim=None):
        super(Critic, self).__init__()
        action_dim = action_space.shape[0]
        input_dim = obs_dim + action_dim

        if embedding_dim is not None:
            self.embedding = PositionalEmbedding(embedding_dim)
            input_dim = input_dim * 2 * self.embedding.L

        # Q1 network
        layers = [nn.Linear(input_dim, hidden_dims[0]), get_activation(activation)]
        for dim_in, dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(get_activation(activation))
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.q1 = nn.Sequential(*layers)

        # Q2 network
        layers2 = [nn.Linear(input_dim, hidden_dims[0]), get_activation(activation)]
        for dim_in, dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers2.append(nn.Linear(dim_in, dim_out))
            layers2.append(get_activation(activation))
        layers2.append(nn.Linear(hidden_dims[-1], 1))
        self.q2 = nn.Sequential(*layers2)

    def forward(self, obs, action):
        xu = torch.cat([obs, action], dim=-1)
        if hasattr(self, 'embedding'):
            xu = self.embedding(xu)
        return self.q1(xu), self.q2(xu)