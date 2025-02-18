import numpy as np
import hockey.hockey_env as h_env
from hockey.hockey_env import Mode
import gymnasium as gym
from importlib import reload
from stable_baselines3.common.env_checker import check_env

from gymnasium.spaces import Box

def basic_sum_intermediate(info):
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


class SinglePlayerHockeyEnv(gym.Env):
    """
    A custom Gym environment for the hockey game. THis includes a wrapper for the hockey environment, 
    and the basic opponent agent to train against.

    This simplifies the training process by providing a simple interface to the environment, allowing 
    for easy integration with RL algorithms.
    """

    def __init__(self, weak_mode = False):
        """
        Initialize the environment.

        Args:
        - weak_mode: a boolean indicating whether the opponent should be weak or strong
        - reward_scheme: a function that combines the intermediate rewards into a single scalar reward
        """

        self.env = h_env.HockeyEnv()
        self.opponent =  h_env.BasicOpponent(weak=weak_mode)
        self.action_space = Box(low=-1, high=1, shape=(4,))
        self.observation_space = self.env.observation_space
        self._step = 0
        self._first_time_touch = 1
        self._touched = 0

        self.reset()

    def reward_scheme(self, reward, _info):
        self._touched = max(self._touched, _info['reward_touch_puck'])

        step_reward = (
            reward  
            + 5 * _info['reward_closeness_to_puck']
            - (1 - self._touched) * 0.1
            + self._touched * self._first_time_touch * 0.1 * self._step
        )
        self._first_time_touch = 1 - self._touched
        return step_reward

    def step(self, action):
        """
        Take a step in the environment given an action.
        """

        obs2 = self.env.obs_agent_two()
        action2 = self.opponent.act(obs2)

        obs, r, d, t, info = self.env.step(np.hstack([action,action2]))

        step_reward = self.reward_scheme(r, info)
        self._step += 1

        return obs, step_reward, d, t, info
    
    def reset(self, *args, **kwargs):
        """
        Reset the environment.
        """
        self._step = 0
        self._touched = 0
        self._first_time_touch = 1
        
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
