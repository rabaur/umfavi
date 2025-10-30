import gymnasium as gym
import numpy as np
from umfavi.envs.dct_grid_env import DCTGridEnv
from umfavi.utils.tabular import q_opt
from umfavi.utils.math import softmax

class UniformPolicy:
    """
    Generic uniform policy for any Gymnasium-compatible environment.
    """

    def __init__(self, action_space: gym.Space):
        self.action_space = action_space
    
    def __call__(self, observation=None):
        return self.action_space.sample()


class ExpertPolicy:
    """
    Expert policy for DCTGridEnv based on optimal Q-values.
    Uses softmax over Q-values: π(a|s) = exp(β*Q(s,a)) / Σ_a' exp(β*Q(s,a'))
    """

    def __init__(self, env: DCTGridEnv, rationality: float = 1.0, gamma: float = 0.99):
        """
        Initialize expert policy.
        
        Args:
            env: DCTGridEnv environment
            rationality: Rationality parameter (β) for softmax policy
            gamma: Discount factor for Q-value computation
        """
        self.env = env
        self.rationality = rationality
        self.gamma = gamma
        
        # Compute optimal Q-values
        self.Q_optimal = q_opt(env.P, env.R, self.gamma)

        # Create softmax policy: π(a|s) = exp(β*Q(s,a)) / Σ_a' exp(β*Q(s,a'))
        policy = softmax(self.rationality * self.Q_optimal, dims=1)
        self.policy = policy
    
    def __call__(self, observation):
        """
        Sample action from the expert policy.
        
        Args:
            observation: Observation from environment (dict with 'coord' and 'observation')
            
        Returns:
            Action sampled from the expert policy
        """
        # Extract state index from observation
        if isinstance(observation, tuple):
            # Handle reset return format (obs, info)
            observation = observation[0]
        
        coord = observation['coord']
        state_idx = coord[0] * self.env.grid_size + coord[1]
        
        # Sample action from policy distribution
        action = np.random.choice(self.env.action_space.n, p=self.policy[state_idx])
        
        return action