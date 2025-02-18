import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from sac.custom_env import SinglePlayerHockeyEnv
import hockey.hockey_env as h_env

from sac.agent import SAC
from td3.agent import TD3


def evaluate(agent, env, config):
    agent.eval()
    mean_reward = 0
    win_rate = 0
    loss_rate = 0
    for episode in range(config['eval_episodes']):
        obs, _ = env.reset()
        episode_reward = 0
        for step in range(config['max_steps_in_episode']):
            action = agent.act(obs)
            obs, reward, done, truncated, _info = env.step(action)

            episode_reward += reward

            if done or truncated:
                if 'winner' in _info:
                    if _info['winner'] == 1:
                        win_rate += 1
                    elif _info['winner'] == -1:
                        loss_rate += 1
                    break

        mean_reward += episode_reward
        
    mean_reward /= config['eval_episodes']
    win_rate /= config['eval_episodes']
    loss_rate /= config['eval_episodes']
    return {'eval_reward': mean_reward, 'eval_win_rate': win_rate, 'eval_loss_rate': loss_rate}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SAC Agent")
    parser.add_argument('--model-path', type=str, help='Path to the model file')
    parser.add_argument('--opponent-type', choices=['weak', 'strong', 'custom'], 
                        help='Path to the opponent file')
    parser.add_argument('--opponent-path', type=str, help='Path to the opponent file. Only used if opponent-type is custom')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = dict()
    env = SinglePlayerHockeyEnv(weak_mode=False)
    if args.opponent_type == 'weak':
        opponent = h_env.BasicOpponent(weak=True)
    elif args.opponent_type == 'strong':
        opponent = h_env.BasicOpponent(weak=False)
    elif args.opponent_type == 'custom':
        opponent = torch.load(args.opponent_path, weights_only=False)
        opponent.eval()
    env.set_opponent(opponent)
    env.reset()
    agent = torch.load(args.model_path, weights_only=False)
    agent.eval()
    config['eval_episodes'] = 1000
    config['max_steps_in_episode'] = 250
    eval_results = evaluate(agent, env, config)
    print(eval_results)
    