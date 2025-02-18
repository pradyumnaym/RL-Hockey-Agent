import os
import torch
import time

from collections import defaultdict

from .agent import TD3
from common.replay_buffer import ReplayBuffer
from common.logger import Logger

class Trainer:
    def __init__(self, cfg, env, logger, replay_buffer):
        self.config = cfg
        self.env = env
        self.agent = TD3(cfg.agent, env.observation_space.shape[0], env.action_space)
        self.logger = logger
        self.replay_buffer = replay_buffer

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
    
        iteration = 0
        start_time = time.time()

        for episode in range(1, self.config.max_episodes+1):
            obs, _ = self.env.reset()
            self.agent.train()
            print("==== Episode: ", episode, "=====")
            for step in range(self.config.max_steps_in_episode):
                iteration += 1
                action = self.agent.act(obs)
                next_state, reward, done, truncated, _info = self.env.step(action)

                self.replay_buffer.add(obs, next_state, action, reward, done)

                if done or truncated:
                    break

                obs = next_state

            if self.replay_buffer.size() < self.config.batch_size:
                continue

            losses_dict = defaultdict(list)

            self.agent.train()
            for _ in range(self.config.grad_steps):
                data = self.replay_buffer.sample(self.config.batch_size)
                loss_dict = self.agent.update(iteration, data)

                for key, value in loss_dict.items():
                    losses_dict[key].append(value)

            mean_losses_dict = {key: sum(value) / len(value) for key, value in losses_dict.items()}

            print(f"Episode: {episode}, Mean Losses: {mean_losses_dict}, Time: {time.time() - start_time}")

            self.agent.schedulers_step()

            self.logger.log({
                'train_episode': episode,
                **mean_losses_dict
            })

            if episode % self.config.eval_freq == 0:
                metrics = self.evaluate(episode)
                self.logger.log_metrics(metrics)

                torch.save(self.agent, os.path.join(self.config.out_folder, 'model.pth'))
