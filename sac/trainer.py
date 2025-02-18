import os
import torch
import time
import copy

from .agent import SACAgent 
from common.replay_buffer import ReplayBuffer
from common.logger import Logger
from ..common.opponent_pooler import OpponentPooler
class Trainer:
    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.agent = SACAgent(config, env.observation_space.shape[0], env.action_space.shape[0], env.action_space)
        self.resume_from = config.get('resume_from', None)
        if self.resume_from is not None:
            state_dict = torch.load(self.resume_from, weights_only=False).state_dict()

            # dirty fix, TODO: remove later
            if 'log_alpha' not in state_dict:
                import numpy as np
                state_dict['log_alpha'] = torch.tensor(np.log(self.agent.alpha), requires_grad=True).to(self.agent.device)
            self.agent.load_state_dict(state_dict, strict=False)

        self.logger = Logger(config)
        self.replay_buffer = ReplayBuffer(config['replay_buffer_size'])
        self.opponent_pooler = OpponentPooler(config['weak_prob'], config['strong_prob'], 
                                              config['self_prob'], copy.deepcopy(self.agent))
        
        

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
        out_folder = self.config['out_folder']
        os.makedirs(out_folder, exist_ok=True)
        
        iteration = 0
        start_time = time.time()
        for episode in range(1, self.config['max_episodes']+1):
            # sample opponent for each episode
            opponent = self.opponent_pooler.sample_opponent()
            self.env.set_opponent(opponent)
            obs, _ = self.env.reset()
            self.agent.train()
            print("==== Episode: ", episode, "=====")
            for step in range(self.config['max_steps_in_episode']):
                iteration += 1
                # obs = self.env.obs_agent_one()
                action = self.agent.act(obs)
                next_state, reward, done, truncated, _info = self.env.step(action)

                self.replay_buffer.add(obs, next_state, action, reward, done)

                if done:
                    break
                
                obs = next_state

            if self.replay_buffer.size() < self.config['batch_size']:
                continue
            
            mean_critic_loss = 0
            mean_actor_loss = 0
            mean_buffer_reward = 0
            mean_alpha_loss = 0
            
            for _ in range(self.config['grad_steps']):
                self.agent.train()
                data = self.replay_buffer.sample(self.config['batch_size'])

                critic_loss, actor_loss, buffer_reward, alpha_loss = self.agent.update(iteration, data)

                mean_critic_loss += critic_loss
                mean_actor_loss += actor_loss
                mean_buffer_reward += buffer_reward
                mean_alpha_loss += alpha_loss

            mean_critic_loss /= self.config['grad_steps']
            mean_actor_loss /= self.config['grad_steps']
            mean_buffer_reward /= self.config['grad_steps']
            mean_alpha_loss /= self.config['grad_steps']

            end_time = time.time()
            print(f"Time taken to train {episode} episodes: {end_time - start_time:.2f} seconds")

            self.agent.schedulers_step()
            
            self.logger.log({'train_episode': episode, 'critic_loss': mean_critic_loss, 'actor_loss': mean_actor_loss, 
                             'buffer_reward': mean_buffer_reward, 'alpha_loss': mean_alpha_loss})

            eval_results = {'train_episode': episode, 'eval_reward': 0}
            if episode % self.config['eval_freq'] == 0:
                # save model
                torch.save(self.agent, f'{out_folder}/sac_agent.pth')
                
                eval_count = 0
                for opponent, name in zip([self.opponent_pooler.weak_opponent, 
                                          self.opponent_pooler.strong_opponent, 
                                          self.opponent_pooler.self_opponent], 
                                         ['weak', 'strong', 'self']):
                    if self.config[f'{name}_prob'.format(name)] > 0:
                        eval_count += 1
                        partial_eval_results = self.evaluate(opponent)
                        eval_results['eval_reward'] += partial_eval_results['eval_reward']
                        eval_results[f'eval_win_rate_{name}'] = partial_eval_results['eval_win_rate']

                eval_results['eval_reward'] /= eval_count
                self.logger.log_metrics(eval_results)

            if episode % self.config['update_self_opponent_freq'] == 0:
                # only update self opponent if the current win rate > 55%
                if eval_results.get('eval_win_rate_self', 0) > 0.55:
                    tmp_opponent = torch.load(f'{out_folder}/sac_agent.pth', weights_only=False)
                    self.opponent_pooler.update_self_opponent(tmp_opponent)
