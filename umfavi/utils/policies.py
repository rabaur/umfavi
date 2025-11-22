import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
import os
from umfavi.utils.tabular import q_opt
from umfavi.utils.math import softmax
from umfavi.envs.grid_env.env import GridEnv


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

    def __init__(self, env: gym.Env, rationality: float = 1.0, gamma: float = 0.99):
        """
        Initialize expert policy.
        
        Args:
            env: Gymnasium environment
            rationality: Rationality parameter (β) for softmax policy
            gamma: Discount factor for Q-value computation
        """
        self.env = env
        self.rationality = rationality
        self.gamma = gamma
    
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
    
    Computes optimal Q-values from transition dynamics P and rewards R,
    then uses softmax over Q-values to define the policy.
    """

    def __init__(self, env: gym.Env, rationality: float = 1.0, gamma: float = 0.99):
        """
        Initialize tabular expert policy.
        
        Args:
            env: Tabular environment with P and R attributes
            rationality: Rationality parameter (β) for softmax policy
            gamma: Discount factor for Q-value computation
        """
        super().__init__(env, rationality, gamma)
        
        # Verify environment has required attributes
        if not (hasattr(env, 'P') and hasattr(env, 'R')):
            raise ValueError(
                "TabularExpertPolicy requires environment with P and R attributes. "
                "Use DQNExpertPolicy for non-tabular environments."
            )
        
        # Compute optimal Q-values
        self.Q_optimal = q_opt(env.P, env.R, self.gamma)

        # Create softmax policy: π(a|s) = exp(β*Q(s,a)) / Σ_a' exp(β*Q(s,a'))
        self.policy = softmax(self.rationality * self.Q_optimal, dims=1)
    
    def __call__(self, observation):
        """
        Sample action from the expert policy.
        
        Args:
            observation: Observation from environment (dict with 'state' key)
            
        Returns:
            Action sampled from the expert policy
        """
        # Handle reset return format (obs, info)
        if isinstance(observation, tuple):
            observation = observation[0]
        
        # Extract state index from observation
        coord = observation['state']
        state_idx = coord[0] * self.env.grid_size + coord[1]
        
        # Sample action from policy distribution
        action = np.random.choice(self.env.action_space.n, p=self.policy[state_idx])
        
        return action


class DQNExpertPolicy(ExpertPolicy):
    """
    Expert policy for non-tabular environments using Deep Q-Networks.
    
    Uses a pretrained or newly trained DQN model to get Q-values,
    then applies softmax with rationality parameter.
    """

    def __init__(
        self, 
        env: gym.Env, 
        rationality: float = 1.0, 
        gamma: float = 0.99,
        model_path: str = None,
        train_if_missing: bool = True,
        training_timesteps: int = 100000
    ):
        """
        Initialize DQN expert policy.
        
        Args:
            env: Gymnasium environment
            rationality: Rationality parameter (β) for softmax policy
            gamma: Discount factor for Q-value computation
            model_path: Path to pretrained DQN model (if None, uses default based on env name)
            train_if_missing: Whether to train a new model if none exists
            training_timesteps: Number of timesteps for training if needed
        """
        super().__init__(env, rationality, gamma)
        
        self.model_path = model_path
        self.training_timesteps = training_timesteps
        
        # Check for stable-baselines3
        try:
            from stable_baselines3 import DQN
            self.DQN = DQN
        except ImportError:
            raise ImportError(
                "stable-baselines3 is required for DQNExpertPolicy. "
                "Install with: pip install stable-baselines3"
            )
        
        # Determine model path
        if self.model_path is None:
            env_name = env.spec.id if hasattr(env, 'spec') and env.spec else 'unknown'
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            self.model_path = str(models_dir / f"{env_name}_dqn.zip")
        
        # Load or train model
        if os.path.exists(self.model_path):
            print(f"Loading pretrained DQN model from {self.model_path}")
            self.dqn_model = self.DQN.load(self.model_path, env=self.env)
        elif train_if_missing:
            print(f"No pretrained model found at {self.model_path}")
            self._train_dqn_model()
        else:
            raise FileNotFoundError(
                f"No pretrained model found at {self.model_path} and train_if_missing=False"
            )
    
    def _train_dqn_model(self):
        """Train a DQN model for the environment."""
        print(f"Training DQN for {self.training_timesteps} timesteps...")
        
        self.dqn_model = self.DQN(
            "MlpPolicy",
            self.env,
            gamma=self.gamma,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            target_update_interval=500,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            verbose=1
        )
        
        self.dqn_model.learn(total_timesteps=self.training_timesteps)
        
        # Save the trained model
        print(f"Saving trained model to {self.model_path}")
        self.dqn_model.save(self.model_path)
    
    def _get_q_values(self, observation):
        """
        Get Q-values for an observation.
        
        Args:
            observation: Environment observation
            
        Returns:
            Q-values as numpy array
        """
        # Convert observation to tensor format expected by stable-baselines3
        obs_tensor = self.dqn_model.policy.obs_to_tensor(observation)[0]
        
        # Get Q-values from the model
        q_values = self.dqn_model.q_net(obs_tensor)
        
        # Convert to numpy
        return q_values.detach().cpu().numpy().flatten()
    
    def __call__(self, observation):
        """
        Sample action from the expert policy using softmax over Q-values.
        
        Args:
            observation: Observation from environment
            
        Returns:
            Action sampled from the expert policy
        """
        # Handle reset return format (obs, info)
        if isinstance(observation, tuple):
            observation = observation[0]
        
        # Get Q-values for this observation
        q_values = self._get_q_values(observation)
        
        # Apply softmax with rationality parameter
        action_probs = softmax(self.rationality * q_values, dims=0)
        
        # Sample action from the distribution
        action = np.random.choice(len(action_probs), p=action_probs)
        
        return action


def create_expert_policy(
    env: gym.Env,
    rationality: float = 1.0,
    gamma: float = 0.99,
    **kwargs
) -> ExpertPolicy:
    """
    Factory function to create the appropriate expert policy based on environment type.
    
    Automatically detects whether the environment is tabular (has P and R attributes)
    or non-tabular, and creates the corresponding expert policy.
    
    Args:
        env: Gymnasium environment
        rationality: Rationality parameter (β) for softmax policy
        gamma: Discount factor for Q-value computation
        **kwargs: Additional arguments for DQNExpertPolicy:
            - model_path: Path to pretrained model
            - train_if_missing: Whether to train if model doesn't exist
            - training_timesteps: Number of timesteps for training
    
    Returns:
        ExpertPolicy instance (TabularExpertPolicy or DQNExpertPolicy)
    
    Examples:
        >>> # For tabular environment
        >>> env = GridEnv(...)
        >>> policy = create_expert_policy(env, rationality=5.0)
        >>> # Returns TabularExpertPolicy
        
        >>> # For non-tabular environment
        >>> env = gym.make("CartPole-v1")
        >>> policy = create_expert_policy(env, rationality=1.0, train_if_missing=True)
        >>> # Returns DQNExpertPolicy
    """
    # Check if environment is tabular
    is_tabular = hasattr(env, 'P') and hasattr(env, 'R')
    
    if is_tabular:
        # Filter out DQN-specific kwargs
        return TabularExpertPolicy(env=env, rationality=rationality, gamma=gamma)
    else:
        # Pass through all kwargs for DQN policy
        return DQNExpertPolicy(env=env, rationality=rationality, gamma=gamma, **kwargs)