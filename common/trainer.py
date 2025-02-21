import os
import torch
import time
import hydra   
import copy
import numpy as np


from collections import defaultdict
from tqdm import tqdm

from common.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
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
        tmp_self_opponent = globals()[cfg.agent_name](cfg.agent, env.observation_space.shape[0], env.action_space)
        tmp_self_opponent.load_state_dict(self.agent.state_dict())
        # custom opponents: previously trained Pytorch models (load from files)
        custom_opponents = []
        for path in cfg.opponent_pooler.custom_weight_paths:
            tmp_custom_opponent = torch.load(path, weights_only=False, map_location='cpu')
            tmp_custom_opponent.eval()
            custom_opponents.append(tmp_custom_opponent)  
        self.opponent_pooler = OpponentPooler(**self.config.opponent_pooler, self_opponent=tmp_self_opponent, custom_opponents=custom_opponents)

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

    def evaluate(self, opponent):
        self.agent.eval()
        mean_reward = 0
        win_rate = 0
        for episode in range(self.config['eval_episodes']):
            self.env.set_opponent(opponent)
            obs, _ = self.env.reset()
            episode_reward = 0
            for step in range(self.config['max_steps_in_episode']):
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
            
        mean_reward /= self.config['eval_episodes']
        win_rate /= self.config['eval_episodes']

        return {'eval_reward': mean_reward, f'eval_win_rate': win_rate}
    
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
                self.replay_buffer.add((obs, next_state, action, reward, done))
                if done or truncated:
                    break

                obs = next_state

            # If in initial period 
            if self.replay_buffer.size() < self.config.batch_size or episode < self.config.start_training_after:
                continue

            losses_dict = defaultdict(list)

            self.agent.train()
            for _ in range(self.config.grad_steps):
                data, weights = self.replay_buffer.sample(self.config.batch_size, episode=episode)

                if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
                    weights, indices = weights['_weight'], weights['index']
                    weights = weights.to(self.device)

                loss_dict, td_errors = self.agent.update(iteration, data, weights)

                if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
                    self.replay_buffer.update_priorities(indices, td_errors)

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
                for idx, (opponents, name) in enumerate(zip([self.opponent_pooler.weak_opponent, 
                                          self.opponent_pooler.strong_opponent, 
                                          self.opponent_pooler.self_opponent,
                                          self.opponent_pooler.custom_opponents], 
                                         ['weak', 'strong', 'self', 'custom'])):
                    prob = self.opponent_pooler.get_current_probabilities()[idx]
                    if prob > max_prob:
                        max_prob = prob
                        main_opponent_name = name

                    if prob > 0:
                        eval_count += 1
                        if name != 'custom': # single opponent (weak, strong, self)
                            partial_eval_results = self.evaluate(opponents)
                        else: # custom opponents (multiple Pytorch opponents)
                            partial_eval_results = []
                            for opponent in opponents:
                                eval_results_per_custom_opponent = self.evaluate(opponent)
                                partial_eval_results.append(eval_results_per_custom_opponent)
                            # mean of the results over all custom opponents
                            partial_eval_results = {k: sum(d[k] for d in partial_eval_results) / len(partial_eval_results) for k in partial_eval_results[0]}

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
                # only update self opponent if the current win rate > 80%
                if eval_results.get('eval_win_rate_self', 0) > 0.8:
                    print("Updating self opponent")
                    # load_path = self.best_model_path if os.path.exists(self.best_model_path) else self.last_model_path
                    
                    tmp_self_opponent = globals()[self.config.agent_name](self.config.agent, self.env.observation_space.shape[0], self.env.action_space)
                    # copy weights from the current model
                    tmp_self_opponent.load_state_dict(self.agent.state_dict())
                    self.opponent_pooler.update_self_opponent(tmp_self_opponent)