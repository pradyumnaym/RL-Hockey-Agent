import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .models import Actor, Critic

class SAC(nn.Module):
    def __init__(self, config, obs_dim, action_space):
        super(SAC, self).__init__()
        self.config = config
        action_dim = action_space.shape[0]
        actor_hidden_dims = config['actor_hidden_dims']
        critic_hidden_dims = config['critic_hidden_dims']

        self.actor = Actor(obs_dim, actor_hidden_dims, action_dim, action_space)
        self.critic = Critic(obs_dim, critic_hidden_dims, action_dim)
        self.critic_target = Critic(obs_dim, critic_hidden_dims, action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['actor_lr'],  eps=0.000001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['critic_lr'], eps=0.000001)

        self.actor_scheduler = optim.lr_scheduler.MultiStepLR(self.actor_optimizer, milestones=config['actor_lr_milestones'], gamma=config['actor_lr_gamma'])
        self.critic_scheduler = optim.lr_scheduler.MultiStepLR(self.critic_optimizer, milestones=config['critic_lr_milestones'], gamma=config['critic_lr_gamma'])

        # hard copy critic target
        self.critic_target.hard_copy(self.critic)

        self.entropy_tuning = config['entropy_tuning']
        self.alpha = config['alpha']
        if self.entropy_tuning:
            self.register_buffer("log_alpha", torch.tensor(np.log(self.alpha), requires_grad=True))
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config['alpha_lr'])
            self.alpha_scheduler = optim.lr_scheduler.MultiStepLR(self.alpha_optimizer, milestones=config['alpha_lr_milestones'], gamma=config['alpha_lr_gamma'])
            self.register_buffer("target_entropy", -torch.prod(torch.Tensor(action_dim)).requires_grad_(False))
        
        self.is_training = True

    def train(self):
        self.is_training = True

    def eval(self):
        self.is_training = False

    def act(self, obs):
        device = next(self.actor.parameters()).device
        state = torch.FloatTensor(obs).to(device).unsqueeze(0)
        action, _ = self.actor.sample(state, deterministic=not self.is_training)
        return action.detach().cpu().numpy()[0]


    def schedulers_step(self):
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def update(self, iteration, data, weights):
        self.actor.train()
        self.critic.train()
        self.critic_target.train()

        state, next_state, action, reward, done = data

        device = next(self.actor.parameters()).device
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).to(device).unsqueeze(1)
            done = torch.FloatTensor(done).to(device).unsqueeze(1)

        # get td target for critic
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            
            target_q = reward + (1 - done) * self.config['discount_factor'] * target_q

        # compute critic loss
        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # get policy loss
        sampled_action, log_prob = self.actor.sample(state)
        q1_rollout, q2_rollout = self.critic(state, sampled_action)

        min_q = torch.min(q1_rollout, q2_rollout)

        policy_loss = (self.alpha * log_prob - min_q).mean(axis=0)

        # update actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        if iteration % self.config['critic_target_update_freq'] == 0:
            self.critic_target.ema_update(self.critic, self.config['critic_target_tau'])


        losses_dict = {
            'critic_loss': critic_loss.item(),
            'actor_loss': policy_loss.item(),
            'buffer_reward': reward.mean().item(),
        }

        # update alpha
        if self.entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha_scheduler.step()
            self.alpha = self.log_alpha.exp()

            losses_dict['alpha_loss'] = alpha_loss.item()
        
        return losses_dict, None