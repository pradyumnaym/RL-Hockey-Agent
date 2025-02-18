import yaml
import os
from sac.trainer import Trainer
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import hockey.hockey_env as h_env
from sac.custom_env import SinglePlayerHockeyEnv

from sac.agent import SACAgent

def load_config():
    # load config from yaml file
    with open('configs/sac_v5.yaml', 'r') as f:
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

if __name__ == '__main__':
    config = load_config()
    # env = gym.make('Pendulum-v1', render_mode="rgb_array")
    # env = SinglePlayerHockeyEnv(weak_mode=True)
    env = h_env.HockeyEnv()
    env.reset()
    # agent = SACAgent(config, env.observation_space.shape[0], env.action_space.shape[0], env.action_space)
    agent = torch.load(config['out_folder'] + '/sac_agent.pth',  weights_only=False)
    agent.eval()

    opponent = h_env.BasicOpponent(weak=False)
    # opponent = torch.load("checkpoints/sac_v4/sac_agent.pth", weights_only=False)
    # opponent.eval()
    
    for i in range(10):
        print(f"Episode {i+1} of 10")
        frames = []
        obs, _ = env.reset()
        done = False
        step = 0

        while not done and step < 250:
            step += 1

            # action1 = opponent.act(obs)
            action1 = agent.act(obs)

            obs2 = env.obs_agent_two()
            # action2 = agent.act(obs2)
            action2 = opponent.act(obs2)

            obs, reward, done, truncated, info = env.step(np.hstack([action1, action2]))
            print(step, reward)
            # print("reward_closeness_to_puck", info["reward_closeness_to_puck"])
            # print("reward_touch_puck", info["reward_touch_puck"])
            # print("reward_puck_direction", info["reward_puck_direction"])
            frame = env.render(mode="rgb_array")
            frames.append(frame)


        env.close()
        os.makedirs('gifs', exist_ok=True)
        save_frames_as_gif(frames, path='./', filename=f'gifs/gym_animation{i}.gif')

        # break