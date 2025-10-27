import torch
import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset
from typing import Callable
import gymnasium as gym
from virel.utils.gym import rollout, get_obs_act_pairs, get_rewards
from virel.utils.math import sigmoid

class PreferenceDataset(Dataset):
    """
    Dataset for preference learning with trajectory pairs and simulated preferences.
    """
    def __init__(
        self, 
        n_samples: int,   
        n_steps: int,
        policy: Callable,
        env: gym.Env,
        device: str,
        rationality: float = 1.0,
        obs_transform: Callable = None,
        act_transform: Callable = None
    ):
        """
        Initialize preference dataset.
        
        Args:
            num_samples: Number of preference samples to generate
            init_state_dist: Initial state distribution
            init_policy: Initial policy
            env: Environment
        """
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.env = env
        self.rationality = rationality
        self.obs_transform = obs_transform
        self.act_transform = act_transform
        self.device = device

        # Generate trajectory pairs and preferences
        self.obs_seq_pairs, self.acts_seq_pairs, self.preferences = self.generate_preferences(policy=policy)
        
    def _compute_trajectory_return(self, trajectory: list[tuple[int, int, float]]) -> float:
        """
        Compute the total return of a trajectory.
        
        Args:
            trajectory: List of (state, action, reward) tuples
            
        Returns:
            Total return (sum of rewards)
        """
        return sum(reward for _, _, reward in trajectory)
    
    def add_preferences(self, policy: Callable) -> None:
        """
        Add preferences to the dataset.
        """
        obs_seq_pairs, acts_seq_pairs, new_preferences = self.generate_preferences(policy)
        if self.obs_seq_pairs is None:
            self.obs_seq_pairs, self.acts_seq_pairs, self.preferences = obs_seq_pairs, acts_seq_pairs, new_preferences
        else:
            self.obs_seq_pairs.extend(obs_seq_pairs)
            self.acts_seq_pairs.extend(acts_seq_pairs)
            self.preferences.extend(new_preferences)
        self.n_samples = len(self.preferences)
    
    def generate_preferences(self, policy: Callable) -> tuple[list[list[dict]], list[list[dict]], list[int]]:
        """
        Generate trajectory pairs and preferences.
        
        Returns:
            Tuple of (trajectory_pairs, preferences) where:
            - trajectory_pairs: List of (traj1, traj2) pairs
            - preferences: List of preferences (0 for traj1, 1 for traj2)
        """
        obs_seq_pairs = []
        acts_seq_pairs = []
        preferences = []
        
        for _ in range(self.n_samples):

            # Generate two trajectories using imported function
            traj1 = rollout(self.env, policy, n_steps=self.n_steps)
            traj2 = rollout(self.env, policy, n_steps=self.n_steps)

            # Extract state-action pairs from trajectories
            obs1, acts1 = get_obs_act_pairs(traj1)
            obs2, acts2 = get_obs_act_pairs(traj2)

            # Transform observations and actions
            if self.obs_transform:
                obs1 = list(map(self.obs_transform, obs1))
                obs2 = list(map(self.obs_transform, obs2))
            
            if self.act_transform:
                acts1 = list(map(self.act_transform, acts1))
                acts2 = list(map(self.act_transform, acts2))

            # Compute true returns
            rews1 = get_rewards(traj1)
            rews2 = get_rewards(traj2)
            rew_sum1 = sum(rews1)
            rew_sum2 = sum(rews2)

            # Generate preference using sigmoid
            preference_prob = sigmoid(self.rationality * (rew_sum2 - rew_sum1))
            pref = np.random.binomial(1, preference_prob)
            
            # Append the newly generated trajectories
            obs_seq_pairs.append((obs1, obs2))
            acts_seq_pairs.append((acts1, acts2))
            preferences.append(pref)
            
        return obs_seq_pairs, acts_seq_pairs, preferences
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs1, obs2 = self.obs_seq_pairs[idx]
        acts1, acts2 = self.acts_seq_pairs[idx]
        preference = self.preferences[idx]
        
        # Convert observations to tensors
        obs1_tensor = torch.tensor(obs1, dtype=torch.float32).to(self.device)
        obs2_tensor = torch.tensor(obs2, dtype=torch.float32).to(self.device)
        
        # Handle actions - if act_transform was applied, acts1/acts2 are lists of tensors
        if self.act_transform:
            # Stack the already transformed tensors
            acts1_tensor = torch.stack(acts1).to(self.device)
            acts2_tensor = torch.stack(acts2).to(self.device)
        else:
            # Convert to tensor if no transformation was applied
            acts1_tensor = torch.tensor(acts1, dtype=torch.float32).to(self.device)
            acts2_tensor = torch.tensor(acts2, dtype=torch.float32).to(self.device)
        
        # Concatenate obs and acts for easier handling
        obs_tensor = torch.stack([obs1_tensor, obs2_tensor], dim=0)
        acts_tensor = torch.stack([acts1_tensor, acts2_tensor], dim=0)
        preference = torch.tensor(preference, dtype=torch.float32).unsqueeze(0).to(self.device)
        return {
            "feedback_type": "preference",
            "obs": obs_tensor,
            "acts": acts_tensor,
            "targets": preference.squeeze(),
            "rationality": torch.tensor(self.rationality).to(self.device, dtype=torch.float32),
        }
    
    def get_all_data(self):
        """
        Get all trajectory pairs and preferences as numpy arrays.
        
        Returns:
            Tuple of (trajectory_pairs, preferences) as numpy arrays
        """
        return self.trajectory_pairs, np.array(self.preferences)
