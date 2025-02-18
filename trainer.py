import os
import torch
import time
import hydra

import numpy as np
from collections import defaultdict
from tqdm import tqdm

from common.replay_buffer import ReplayBuffer
from common.logger import Logger
from td3.agent import TD3
from sac.agent import SACAgent

class Trainer:
    def __init__(self, cfg, env, logger, replay_buffer, action_noise=None):
        self.config = cfg
        self.env = env
        self.agent = globals()[cfg.agent_name](cfg.agent, env.observation_space.shape[0], env.action_space)
        self.logger = logger
        self.replay_buffer = replay_buffer
        self.action_noise = action_noise

    def evaluate(self, train_episode):
        self.agent.eval()
        mean_reward = 0
        win_rate = 0
        for episode in range(self.config.eval_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            for step in range(self.config.max_steps_in_episode):
                action = self.agent.act(obs)
                obs, reward, done, truncated, _info = self.env.step(action)

                episode_reward += reward

                if done or truncated:
                    # check for winner
                    if 'winner' in _info:
                        if _info['winner'] == 1:
                            win_rate += 1
                        break

            mean_reward += episode_reward
            
        mean_reward /= self.config.eval_episodes
        win_rate /= self.config.eval_episodes
        return {'train_episode': train_episode, 'eval_reward': mean_reward, 'eval_win_rate': win_rate}
    
    def train(self):

        self.agent.to(self.config.device)
        best_win_rate = 0
        iteration = 0

        pbar = tqdm(range(1, self.config.max_episodes+1), position=0, leave=True)
        for episode in pbar:

            # Collect the rollouts, if noise if available, add it to the actions in the normalised space
            obs, _ = self.env.reset()
            self.agent.train()

            if self.action_noise is not None:
                self.action_noise.reset()

            # Collect the rollouts
            for _ in range(self.config.max_steps_in_episode):
                iteration += 1
                action = self.agent.act(obs)

                # Add noise to the action
                if self.action_noise is not None:
                    noised_action = self.agent.actor.scale_action(action) + self.action_noise()
                    noised_action = np.clip(noised_action, -1, 1)
                    action = self.agent.actor.unscale_action(noised_action)

                next_state, reward, done, truncated, _info = self.env.step(action)
                self.replay_buffer.add(obs, next_state, action, reward, done)

                if done or truncated:
                    break

                obs = next_state

            # If in initial period 
            if self.replay_buffer.size() < self.config.batch_size or episode < self.config.start_training_after:
                continue

            losses_dict = defaultdict(list)
            self.agent.train()
            for _ in range(self.config.grad_steps):
                data = self.replay_buffer.sample(self.config.batch_size)
                loss_dict = self.agent.update(iteration, data)

                for key, value in loss_dict.items():
                    losses_dict[key].append(value)

            mean_losses_dict = {key: sum(value) / len(value) for key, value in losses_dict.items()}
            rounded_losses = {k: round(v, 2) for k, v in mean_losses_dict.items()}
            pbar.set_description(f"Losses: {rounded_losses}")

            self.agent.schedulers_step()

            self.logger.log({
                'train_episode': episode,
                **mean_losses_dict
            }, write_to_file = episode % self.config.log_freq == 0)

            if episode % self.config.eval_freq == 0:
                metrics = self.evaluate(episode)
                self.logger.log_metrics(metrics)

                if metrics['eval_win_rate'] > best_win_rate:
                    best_win_rate = metrics['eval_win_rate']
                    torch.save(self.agent, os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, f'best_model_{episode}.pth'))

                torch.save(self.agent, os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'model.pth'))
