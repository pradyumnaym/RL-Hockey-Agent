import os
import torch

from .agent import SACAgent 
from common.replay_buffer import ReplayBuffer
from common.logger import Logger

class Trainer:
    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.agent = SACAgent(config, env.observation_space.shape[0], env.action_space.shape[0], env.action_space)
        self.logger = Logger(config)
        self.replay_buffer = ReplayBuffer(config['replay_buffer_size'])
        

    def evaluate(self, train_episode):
        self.agent.eval()
        mean_reward = 0
        win_rate = 0
        for episode in range(self.config['eval_episodes']):
            obs, _ = self.env.reset()
            episode_reward = 0
            for step in range(self.config['max_steps_in_episode']):
                action = self.agent.act(obs)
                next_state, reward, done, truncated, _info = self.env.step(action)

                episode_reward += reward

                if done or truncated:
                    # check for winner
                    # if _info['winner'] == 1:
                    #     self.logger.log({'eval_reward': reward})
                    if 'winner' in _info:
                        if _info['winner'] == 1:
                            win_rate += 1
                        break

            mean_reward += episode_reward
            
        mean_reward /= self.config['eval_episodes']
        win_rate /= self.config['eval_episodes']
        return {'train_episode': train_episode, 'eval_reward': mean_reward, 'eval_win_rate': win_rate}

    def train(self):
        out_folder = self.config['out_folder']
        os.makedirs(out_folder, exist_ok=True)
        self.agent.train()
        
        iteration = 0
        for episode in range(self.config['max_episodes']):
            obs, _ = self.env.reset()
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
            
            self.logger.log({'train_episode': episode, 'critic_loss': mean_critic_loss, 'actor_loss': mean_actor_loss, 'buffer_reward': mean_buffer_reward, 'alpha_loss': mean_alpha_loss})

            if episode % self.config['eval_freq'] == 0:
                # save model
                torch.save(self.agent, f'{out_folder}/sac_agent.pth')

                eval_results = self.evaluate(train_episode=episode)
                self.logger.log_metrics(eval_results)