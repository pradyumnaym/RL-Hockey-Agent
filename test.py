import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import hockey.hockey_env as h_env
from sac.custom_env import SinglePlayerHockeyEnv

from sac.agent import SAC
from td3.agent import TD3

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SAC Agent")
    parser.add_argument('--model-path', type=str, help='Path to the model file')
    parser.add_argument('--opponent-type', choices=['weak', 'strong', 'custom'], 
                        help='Path to the opponent file')
    parser.add_argument('--opponent-path', type=str, help='Path to the opponent file. Only used if opponent-type is custom')
    return parser.parse_args()


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
    args = parse_args()
    env = h_env.HockeyEnv()
    env.reset()

    agent = torch.load(args.model_path,  weights_only=False)
    agent.eval()

    if args.opponent_type == 'weak':
        opponent = h_env.BasicOpponent(weak=True)
    elif args.opponent_type == 'strong':
        opponent = h_env.BasicOpponent(weak=False)
    elif args.opponent_type == 'custom':
        opponent = torch.load(args.opponent_path, weights_only=False)
        opponent.eval()
    
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
            frame = env.render(mode="rgb_array")
            frames.append(frame)


        env.close()
        os.makedirs('gifs', exist_ok=True)
        save_frames_as_gif(frames, path='./', filename=f'gifs/gym_animation{i}.gif')

        # break