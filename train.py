import yaml
from sac.trainer import Trainer
import gym

from common.single_player_env import SinglePlayerHockeyEnv

def load_config():
    # load config from yaml file
    with open('configs/sac.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == '__main__':
    config = load_config()
    # env = gym.make('Pendulum-v1')
    env = SinglePlayerHockeyEnv(weak_mode=True)
    env.reset()
    trainer = Trainer(config, env)
    trainer.train()