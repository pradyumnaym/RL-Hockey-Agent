import os
import torch
import time
import hydra   
import copy
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from common.replay_buffer import ReplayBuffer
from common.logger import Logger
from td3.agent import TD3
from sac.agent import SAC
from common.opponent_pooler import OpponentPooler


class Trainer:
    def __init__(self, cfg, env, logger, replay_buffer, action_noise=None):
        self.config = cfg
        self.env = env
        self.agent = globals()[cfg.agent_name](cfg.agent, env.observation_space.shape[0], env.action_space)
        resume_from = cfg.agent.resume_from
        if resume_from is not None and os.path.exists(resume_from):
            state_dict = torch.load(resume_from, weights_only=False).state_dict()
            print("Resuming from: ", resume_from)
            self.agent.load_state_dict(state_dict, strict=False)

        self.logger = logger
        self.replay_buffer = replay_buffer
        self.opponent_pooler = OpponentPooler(self.config.opponent_pooler.weak_prob, self.config.opponent_pooler.strong_prob, 
                                              self.config.opponent_pooler.self_prob, copy.deepcopy(self.agent))

        if 'cuda' in self.config.device and torch.cuda.is_available():
            self.device = self.config.device
        elif 'mps' in self.config.device and torch.backends.mps.is_available():
            self.device = self.config.device
        else:
            self.device = 'cpu'

        self.agent.to(self.device)
        print(f"Using device: {self.device}")
        
        self.action_noise = action_noise

        self.best_win_rate = 0
        self.last_model_path = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'model_last.pth')
        self.best_model_path = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'model_best.pth')

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
        iteration = 0

        pbar = tqdm(range(1, self.config.max_episodes+1), position=0, leave=True)
        for episode in pbar:
            start_time = time.time()

            # randomly select an opponent from the opponent pool (by predefined probability)
            opponent = self.opponent_pooler.sample_opponent()
            self.env.set_opponent(opponent)

            obs, _ = self.env.reset()
            self.agent.train()

            if self.action_noise is not None:
                self.action_noise.reset()

            for step in range(self.config.max_steps_in_episode):
                iteration += 1
                action = self.agent.act(obs)

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

            eval_results = {'train_episode': episode, 'eval_reward': 0}
            if episode % self.config.eval_freq == 0:
                # save model
                torch.save(self.agent, self.last_model_path)
                
                eval_count = 0
                max_prob = 0
                main_opponent_name = "" # save best by eval_win_rate of the main opponent. The main opponent is the one with the highest pool probability
                for opponent, name in zip([self.opponent_pooler.weak_opponent, 
                                          self.opponent_pooler.strong_opponent, 
                                          self.opponent_pooler.self_opponent], 
                                         ['weak', 'strong', 'self']):
                    prob = self.config.opponent_pooler[f'{name}_prob']
                    if prob > max_prob:
                        max_prob = prob
                        main_opponent_name = name

                    if prob > 0:
                        eval_count += 1
                        partial_eval_results = self.evaluate(opponent)
                        eval_results['eval_reward'] += partial_eval_results['eval_reward']
                        eval_results[f'eval_win_rate_{name}'] = partial_eval_results['eval_win_rate']

                eval_results['eval_reward'] /= eval_count
                self.logger.log_metrics(eval_results)

                # Check if the current model is the best
                if main_opponent_name:
                    main_opponent_win_rate = eval_results.get(f'eval_win_rate_{main_opponent_name}', 0)
                    if main_opponent_win_rate > self.best_win_rate:
                        self.best_win_rate = main_opponent_win_rate
                        torch.save(self.agent, self.best_model_path)

            if episode % self.config.opponent_pooler.update_self_opponent_freq == 0:
                # only update self opponent if the current win rate > 55%
                if eval_results.get('eval_win_rate_self', 0) > 0.55:
                    print("Updating self opponent")
                    load_path = self.best_model_path if os.path.exists(self.best_model_path) else self.last_model_path
                    tmp_opponent = torch.load(load_path, weights_only=False)
                    self.opponent_pooler.update_self_opponent(tmp_opponent)
