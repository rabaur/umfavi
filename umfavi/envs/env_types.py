import gymnasium as gym
import abc

class TabularEnv(gym.Env, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_transition_matrix(self):
        pass
    
    @abc.abstractmethod
    def get_reward_matrix(self):
        pass