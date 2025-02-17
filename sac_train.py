import yaml
import argparse
from sac.trainer import Trainer
import gymnasium as gym

from sac.custom_env import SinglePlayerHockeyEnv

def load_config(config_path):
    # load config from yaml file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAC Training Script')
    parser.add_argument('--config', type=str, default='configs/sac_v2.yaml', help='Path to the config file')
    args = parser.parse_args()

    config = load_config(args.config)
    # env = gym.make('Pendulum-v1')
    env = SinglePlayerHockeyEnv(weak_mode=True)
    env.reset()
    trainer = Trainer(config, env)
    trainer.train()