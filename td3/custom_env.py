import numpy as np
import hockey.hockey_env as h_env
from hockey.hockey_env import Mode
import gymnasium as gym
from importlib import reload
from stable_baselines3.common.env_checker import check_env


class SinglePlayerHockeyEnv(gym.Env):
    """
    A custom Gym environment for the hockey game. THis includes a wrapper for the hockey environment, 
    and the basic opponent agent to train against.

    This simplifies the training process by providing a simple interface to the environment, allowing 
    for easy integration with RL algorithms.
    """

    def __init__(self, weak_mode = False, reward_scheme = '1'):
        """
        Initialize the environment.

        Args:
        - weak_mode: a boolean indicating whether the opponent should be weak or strong
        - reward_scheme: a function that combines the intermediate rewards into a single scalar reward
        """

        self.env = h_env.HockeyEnv()
        self.opponent =  h_env.BasicOpponent(weak=weak_mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_scheme = reward_scheme

    def reward_scheme_1(self, info):
        """
        Combine the various rewards from the info dictionary into a single scalar reward.

        Args:
        - info: a dictionary containing the rewards from the environment

        Returns:
        - a scalar reward
        """
        # Note: we can also use the 'winner' key to determine the final reward (1 for win, -1 for loss)
        # these are just the intermediate rewards

        keys = ['reward_closeness_to_puck', 'reward_touch_puck', 'reward_puck_direction']

        final_reward = info['winner'] * 10          # 10 for win, -10 for loss, 0 for draw
        return sum([info[k] for k in keys]) + final_reward

    def step(self, action):
        """
        Take a step in the environment given an action.
        """

        obs2 = self.env.obs_agent_two()
        action2 = self.opponent.act(obs2)

        obs, r, d, t, info = self.env.step(np.hstack([action,action2]))

        return obs, getattr(self, 'reward_scheme_' + self.reward_scheme)(info), d, t, info
    
    def set_opponent(self, opponent):
        self.opponent = opponent
        self.reset()
    
    def reset(self, *args, **kwargs):
        """
        Reset the environment.
        """

        return self.env.reset(*args, **kwargs)
    
    def render(self, *args, **kwargs):
        """
        Render the environment.
        """

        return self.env.render(*args, **kwargs)

    def close(self):
        """
        Close the environment.
        """

        self.env.close()


if __name__ == '__main__':
    env = SinglePlayerHockeyEnv()
    check_env(env)
