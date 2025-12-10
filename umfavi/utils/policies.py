import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from pathlib import Path
import os
from umfavi.utils.tabular import q_opt
from umfavi.utils.math import softmax
import stable_baselines3 as sb3
from umfavi.envs.env_types import TabularEnv
from umfavi.utils.torch import to_numpy


class QValueModel(ABC):
    """
    Abstract base class for Q-value models.
    
    Q-value models provide the expected return Q(s,a) for state-action pairs.
    """
    
    @abstractmethod
    def get_q_values(self, observation) -> NDArray:
        """
        Get Q-values for all actions given an observation.
        
        Args:
            observation: Environment observation
            
        Returns:
            Array of Q-values for each action
        """
        pass
    
    @property
    @abstractmethod
    def gamma(self) -> float:
        """Return the discount factor used by this Q-value model."""
        pass


class TabularQValueModel(QValueModel):
    """
    Q-value model for tabular environments.
    
    Computes optimal Q-values from transition dynamics P and rewards R.
    """
    
    def __init__(self, env: TabularEnv, gamma: float = 0.99):
        """
        Initialize tabular Q-value model.
        
        Args:
            env: Tabular environment with P and R attributes
            gamma: Discount factor for Q-value computation
        """
        
        self.env = env
        self._gamma = gamma
        R = env.get_reward_matrix()
        P = env.get_transition_matrix()
        self.Q_optimal = q_opt(P, R, gamma)
    
    def get_q_values(self, observation) -> NDArray:
        """Get Q-values for an observation."""
        # Handle reset return format (obs, info)
        if isinstance(observation, tuple):
            observation = observation[0]
        
        # Extract state index from observation
        coord = observation['state']
        state_idx = coord[0] * self.env.grid_size + coord[1]
        
        return self.Q_optimal[state_idx]
    
    @property
    def gamma(self) -> float:
        return self._gamma


class DQNQValueModel(QValueModel):
    """
    Q-value model using Deep Q-Networks.
    
    Wraps a trained DQN model from stable-baselines3.
    """
    
    def __init__(self, dqn_model: sb3.DQN):
        """
        Initialize DQN Q-value model.
        
        Args:
            dqn_model: Trained stable-baselines3 DQN model
        """
        self.dqn_model = dqn_model
    
    def get_q_values(self, observation) -> NDArray:
        """Get Q-values for an observation."""
        # Handle reset return format (obs, info)
        if isinstance(observation, tuple):
            observation = observation[0]
        
        # Convert observation to tensor format expected by stable-baselines3
        obs_tensor = self.dqn_model.policy.obs_to_tensor(observation)[0]
        
        # Get Q-values from the model
        q_values = self.dqn_model.q_net(obs_tensor)
        
        # Convert to numpy
        return to_numpy(q_values).flatten()
    
    @property
    def gamma(self) -> float:
        return self.dqn_model.gamma


class UniformPolicy:
    """
    Generic uniform policy for any Gymnasium-compatible environment.
    """

    def __init__(self, action_space: gym.Space):
        self.action_space = action_space
    
    def __call__(self, observation=None):
        return self.action_space.sample()


class ExpertPolicy(ABC):
    """
    Abstract base class for expert policies.
    
    Expert policies use softmax over Q-values: π(a|s) = exp(β*Q(s,a)) / Σ_a' exp(β*Q(s,a'))
    where β is the rationality parameter.
    """

    def __init__(self, q_model: QValueModel, rationality: float = 1.0):
        """
        Initialize expert policy.
        
        Args:
            q_model: Q-value model to use for action selection
            rationality: Rationality parameter (β) for softmax policy
        """
        self.q_model = q_model
        self.rationality = rationality
    
    @property
    def gamma(self) -> float:
        """Return the discount factor from the Q-value model."""
        return self.q_model.gamma
    
    @abstractmethod
    def __call__(self, observation):
        """
        Sample action from the expert policy.
        
        Args:
            observation: Observation from environment
            
        Returns:
            Action sampled from the expert policy
        """
        pass


class TabularExpertPolicy(ExpertPolicy):
    """
    Expert policy for tabular environments (e.g., GridEnv).
    
    Uses a TabularQValueModel and applies softmax over Q-values.
    """

    def __init__(self, q_model: TabularQValueModel, rationality: float = 1.0):
        """
        Initialize tabular expert policy.
        
        Args:
            q_model: TabularQValueModel for Q-value computation
            rationality: Rationality parameter (β) for softmax policy
        """
        super().__init__(q_model, rationality)
        
        # Precompute full policy for efficiency in tabular case
        if self.rationality == float('inf'):
            # Put probability mass uniformly on actions with the highest Q-value
            Q = self.q_model.Q_optimal
            max_q = Q.max(axis=1, keepdims=True)
            is_max = (Q == max_q).astype(float)
            self.policy = is_max / is_max.sum(axis=1, keepdims=True)
        else:
            self.policy = softmax(self.rationality * self.q_model.Q_optimal, dims=1)
        self.env = self.q_model.env
    
    def __call__(self, observation):
        # Handle reset return format (obs, info)
        if isinstance(observation, tuple):
            observation = observation[0]
        
        # Sample action from policy distribution
        dist = self.policy[observation]
        action = np.random.choice(self.env.action_space.n, p=dist)
        
        return action


class DQNExpertPolicy(ExpertPolicy):
    """
    Expert policy for non-tabular environments using Deep Q-Networks.
    
    Uses a DQN Q-value model and applies softmax with rationality parameter.
    """

    def __init__(self, q_model: DQNQValueModel, rationality: float = 1.0):
        """
        Initialize DQN expert policy.
        
        Args:
            q_model: DQNQValueModel for Q-value computation
            rationality: Rationality parameter (β) for softmax policy
        """
        super().__init__(q_model, rationality)
    
    def __call__(self, observation):
        # Get Q-values for this observation
        q_values = self.q_model.get_q_values(observation)
        
        # Apply softmax with rationality parameter
        if self.rationality != float('inf'):
            action_probs = softmax(self.rationality * q_values, dims=0)
        
            # Sample action from the distribution
            action = np.random.choice(len(action_probs), p=action_probs)
        else:
            # Deterministic policy
            action = np.argmax(q_values)
        
        return action


def create_expert_policy(q_model: QValueModel, rationality: float = 1.0) -> ExpertPolicy:
    """
    Factory function to create the appropriate expert policy based on environment type.
    
    Automatically detects whether the environment is tabular (has P and R attributes)
    or non-tabular, and creates the corresponding expert policy.
    
    Args:
        rationality: Rationality parameter (β) for softmax policy
        q_model: Optional pre-trained Q-value model. If provided, uses this directly.
    
    Returns:
        ExpertPolicy instance (TabularExpertPolicy or DQNExpertPolicy)
    """
    # If q_model is provided, use it directly
    if q_model is not None:
        if isinstance(q_model, TabularQValueModel):
            return TabularExpertPolicy(q_model, rationality)
        elif isinstance(q_model, DQNQValueModel):
            return DQNExpertPolicy(q_model, rationality)
        else:
            raise ValueError(f"Unknown Q-value model type: {type(q_model)}")