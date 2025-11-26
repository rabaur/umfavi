import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
import os
from umfavi.utils.tabular import q_opt
from umfavi.utils.math import softmax
import stable_baselines3 as sb3
from umfavi.envs.env_types import TabularEnv
import matplotlib.pyplot as plt


class QValueModel(ABC):
    """
    Abstract base class for Q-value models.
    
    Q-value models provide the expected return Q(s,a) for state-action pairs.
    """
    
    @abstractmethod
    def get_q_values(self, observation) -> np.ndarray:
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
    
    def get_q_values(self, observation) -> np.ndarray:
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
    
    def get_q_values(self, observation) -> np.ndarray:
        """Get Q-values for an observation."""
        # Handle reset return format (obs, info)
        if isinstance(observation, tuple):
            observation = observation[0]
        
        # Convert observation to tensor format expected by stable-baselines3
        obs_tensor = self.dqn_model.policy.obs_to_tensor(observation)[0]
        
        # Get Q-values from the model
        q_values = self.dqn_model.q_net(obs_tensor)
        
        # Convert to numpy
        return q_values.detach().cpu().numpy().flatten()
    
    @property
    def gamma(self) -> float:
        return self.dqn_model.gamma


def load_or_train_dqn(
    env: gym.Env,
    model_path: str = None,
    train_if_missing: bool = True,
    force_train: bool = False,
    gamma: float = 0.99,
    training_timesteps: int = 100_000,
    verbose: int = 1,
    **dqn_kwargs
) -> sb3.DQN:
    """
    Load or train a DQN model for an environment.
    
    Args:
        env: Gymnasium environment
        model_path: Path to pretrained DQN model (if None, uses default based on env name)
        train_if_missing: Whether to train a new model if none exists
        force_train: If True, train a new model even if one exists
        gamma: Discount factor for Q-value computation
        training_timesteps: Number of timesteps for training if needed
        verbose: Verbosity level for training
        **dqn_kwargs: Additional arguments to pass to DQN constructor
        
    Returns:
        Trained DQN model
    """
    # Determine model path
    if model_path is None and not force_train:
        env_name = env.spec.id if hasattr(env, 'spec') and env.spec else 'unknown'
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = str(models_dir / f"{env_name}_dqn.zip")
    
    # Load existing model if not forcing training
    if not force_train and model_path and os.path.exists(model_path):
        print(f"Loading pretrained DQN model from {model_path}")
        return sb3.DQN.load(model_path, env=env)
    
    # Train new model
    if not train_if_missing and not force_train:
        raise FileNotFoundError(
            f"No pretrained model found at {model_path} and train_if_missing=False"
        )
    
    print(f"Training DQN for {training_timesteps} timesteps...")
    
    # Default DQN hyperparameters (can be overridden via dqn_kwargs)
    default_kwargs = {
        "learning_rate": 0.00063,
        "buffer_size": 50_000,
        "learning_starts": 0,
        "batch_size": 128,
        "target_update_interval": 250,
        "exploration_fraction": 0.12,
        "exploration_final_eps": 0.1,
        "train_freq": 4,
        "gradient_steps": -1,
        "policy_kwargs": dict(net_arch=[256, 256])
    }
    default_kwargs.update(dqn_kwargs)
    
    dqn_model = sb3.DQN(
        "MlpPolicy",
        env,
        gamma=gamma,
        verbose=verbose,
        **default_kwargs
    )
    
    dqn_model.learn(total_timesteps=training_timesteps)
    
    # Save the trained model
    if model_path:
        print(f"Saving trained model to {model_path}")
        dqn_model.save(model_path)
    
    return dqn_model


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
            # Put all probability mass on the action with the highest Q-value
            self.policy = np.zeros(self.q_model.Q_optimal.shape)
            self.policy[:, np.argmax(self.q_model.Q_optimal, axis=1)] = 1
        else:
            self.policy = softmax(self.rationality * self.q_model.Q_optimal, dims=1)
        self.env = self.q_model.env
    
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
        """
        Sample action from the expert policy using softmax over Q-values.
        
        Args:
            observation: Observation from environment
            
        Returns:
            Action sampled from the expert policy
        """
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


def create_expert_policy(
    env: gym.Env,
    rationality: float = 1.0,
    gamma: float = 0.99,
    q_model: QValueModel = None,
    **kwargs
) -> ExpertPolicy:
    """
    Factory function to create the appropriate expert policy based on environment type.
    
    Automatically detects whether the environment is tabular (has P and R attributes)
    or non-tabular, and creates the corresponding expert policy.
    
    Args:
        env: Gymnasium environment (only used if q_model is None)
        rationality: Rationality parameter (β) for softmax policy
        gamma: Discount factor for Q-value computation (only used if q_model is None)
        q_model: Optional pre-trained Q-value model. If provided, uses this directly.
        **kwargs: Additional arguments for DQN model loading/training (only used if q_model is None):
            - model_path: Path to pretrained model
            - train_if_missing: Whether to train if model doesn't exist
            - force_train: Whether to force training even if model exists
            - training_timesteps: Number of timesteps for training
    
    Returns:
        ExpertPolicy instance (TabularExpertPolicy or DQNExpertPolicy)
    
    Examples:
        >>> # For tabular environment
        >>> env = GridEnv(...)
        >>> policy = create_expert_policy(env, rationality=5.0)
        
        >>> # For non-tabular environment with auto-loading
        >>> env = gym.make("CartPole-v1")
        >>> policy = create_expert_policy(env, rationality=1.0, train_if_missing=True)
        
        >>> # Share Q-value model across multiple policies
        >>> q_model = DQNQValueModel(dqn_model)
        >>> policy1 = create_expert_policy(env, rationality=1.0, q_model=q_model)
        >>> policy2 = create_expert_policy(env, rationality=5.0, q_model=q_model)
    """
    # If q_model is provided, use it directly
    if q_model is not None:
        if isinstance(q_model, TabularQValueModel):
            return TabularExpertPolicy(q_model, rationality)
        elif isinstance(q_model, DQNQValueModel):
            return DQNExpertPolicy(q_model, rationality)
        else:
            raise ValueError(f"Unknown Q-value model type: {type(q_model)}")
    
    # Check if environment is tabular
    is_tabular = hasattr(env, 'P') and hasattr(env, 'R')
    
    if is_tabular:
        # Create tabular Q-value model
        q_model = TabularQValueModel(env, gamma)
        return TabularExpertPolicy(q_model, rationality)
    else:
        # Load or train DQN model
        dqn_model = load_or_train_dqn(env, gamma=gamma, **kwargs)
        q_model = DQNQValueModel(dqn_model)
        return DQNExpertPolicy(q_model, rationality)