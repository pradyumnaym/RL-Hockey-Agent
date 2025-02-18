import yaml
from sac.trainer import Trainer
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

from sac.custom_env import SinglePlayerHockeyEnv

from sac.agent import SACAgent

def load_config(config_path):
    # load config from yaml file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

def evaluate(agent, env, config):
    agent.eval()
    mean_reward = 0
    win_rate = 0
    loss_rate = 0
    count_reward_10 = 0
    for episode in range(config['eval_episodes']):
        # print(f"Evaluating episode {episode+1} of {config['eval_episodes']}")
        obs, _ = env.reset()
        episode_reward = 0
        for step in range(config['max_steps_in_episode']):
            action = agent.act(obs)
            obs, reward, done, truncated, _info = env.step(action)

            episode_reward += reward

            # if reward == 10:
            #     count_reward_10 += 1
            #     print(count_reward_10)

            if done or truncated:
                # print(done, truncated)
                # print(f"Episode {episode+1} finished")
                # check for winner
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
    parser.add_argument('--config', type=str, default='configs/sac_v1.yaml', help='Path to the configuration file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    # env = gym.make('Pendulum-v1', render_mode="rgb_array")
    env = SinglePlayerHockeyEnv(weak_mode=False)
    env.reset()
    # agent = SACAgent(config, env.observation_space.shape[0], env.action_space.shape[0], env.action_space)
    agent = torch.load(config['out_folder'] + '/sac_agent.pth')
    agent.eval()
    config['eval_episodes'] = 100
    config['max_steps_in_episode'] = 1200
    eval_results = evaluate(agent, env, config)
    print(eval_results)
    