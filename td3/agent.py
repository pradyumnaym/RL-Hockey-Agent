import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import Actor, Critic, EMA

class TD3(nn.Module):
    def __init__(self, config, obs_dim, action_space):
        super(TD3, self).__init__()
        
        self.actor = Actor(obs_dim, action_space, **config.actor)
        self.actor_target = EMA(self.actor, config.tau)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.actor_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.actor_optimizer, milestones=config.actor_lr_milestones, gamma=config.actor_lr_gamma)
        
        self.critic = Critic(obs_dim, action_space, **config.critic)
        self.critic_target = EMA(self.critic, config.tau)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.critic_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.critic_optimizer, milestones=config.critic_lr_milestones, gamma=config.critic_lr_gamma)

        self.config = config
        self.obs_dim = obs_dim
        self.action_space = action_space

    def schedulers_step(self):
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(next(self.actor.parameters()).device)
        return self.actor(obs).detach().cpu().numpy()[0]
    
    def update(self, iteration, data):
        state, next_state, action, reward, done = data

        device = next(self.actor.parameters()).device
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32).to(device)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
            action = torch.tensor(action, dtype=torch.float32).to(device)
            reward = torch.tensor(reward, dtype=torch.float32).to(device).unsqueeze(1)
            done = torch.tensor(done, dtype=torch.float32).to(device).unsqueeze(1)

        # Compute the TD target 
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            noise = torch.normal(0, self.config.policy_noise, next_action.size()).to(device)
            next_action += torch.clamp(noise, -self.config.noise_clip, self.config.noise_clip)
            next_action = torch.clamp(next_action, self.action_space.low[0], self.action_space.high[0])

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.config.discount_factor * target_Q

        # Compute the critic loss
        Q1, Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)


        # Update the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state))[0].mean()

        # Get policy loss
        if iteration % self.config.policy_delay == 0:
            # use the first Q network to get the policy loss

            # Update the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the target networks
            self.actor_target.update(self.actor)
            self.critic_target.update(self.critic)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'buffer_reward': reward.mean().item()
        }