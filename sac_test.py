import yaml
import os
from sac.trainer import Trainer
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from sac.custom_env import SinglePlayerHockeyEnv

from sac.agent import SACAgent

def load_config():
    # load config from yaml file
    with open('configs/sac_v0.yaml', 'r') as f:
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
    env = SinglePlayerHockeyEnv(weak_mode=True)
    env.reset()
    # agent = SACAgent(config, env.observation_space.shape[0], env.action_space.shape[0], env.action_space)
    agent = torch.load(config['out_folder'] + '/sac_agent.pth')
    agent.eval()
    
    for i in range(10):
        print(f"Episode {i+1} of 10")
        frames = []
        obs, _ = env.reset()
        done = False
        step = 0

        while not done and step < 250:
            step += 1
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            print(step, reward)
            frame = env.render(mode="rgb_array")
            frames.append(frame)


        env.close()
        os.makedirs('gifs', exist_ok=True)
        save_frames_as_gif(frames, path='./', filename=f'gifs/gym_animation{i}.gif')