import torch
import copy
import torch.nn as nn

def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
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
    def __init__(self, obs_dim, action_space, hidden_dims, activation='relu', output_activation='tanh'):
        super(Actor, self).__init__()
        
        self.hidden_dims = hidden_dims
        self.action_dim = action_space.shape[0]

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
        x = self.net(obs)
        x = self.output_activation(x)
        return x * self.action_scale + self.action_bias
        

class Critic(nn.Module):
    def __init__(self, obs_dim, action_space, hidden_dims, activation='relu'):
        super(Critic, self).__init__()
        action_dim = action_space.shape[0]
        input_dim = obs_dim + action_dim

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
        return self.q1(xu), self.q2(xu)