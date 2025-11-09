import torch
import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset
from typing import Callable
import gymnasium as gym
from umfavi.utils.gym import rollout, extract_obs_state_actions, get_rewards
from umfavi.utils.math import sigmoid

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
        self.device = device

        # Generate trajectory pairs and preferences
        self.obs_seq_pairs, self.state_seq_pairs, self.acts_seq_pairs, self.preferences = self.generate_preferences(policy=policy)
        
    def _compute_trajectory_return(self, trajectory: list[tuple[int, int, float]]) -> float:
        """
        Compute the total return of a trajectory.
        
        Args:
            trajectory: List of (state, action, reward) tuples
            
        Returns:
            Total return (sum of rewards)
        """
        return sum(reward for _, _, reward in trajectory)
    
    def generate_preferences(self, policy: Callable) -> tuple[list[list[dict]], list[list[dict]], list[int]]:
        """
        Generate trajectory pairs and preferences.
        
        Returns:
            Tuple of (trajectory_pairs, preferences) where:
            - trajectory_pairs: List of (traj1, traj2) pairs
            - preferences: List of preferences (0 for traj1, 1 for traj2)
        """
        obs_seq_pairs = []
        state_seq_pairs = []
        acts_seq_pairs = []
        preferences = []
        
        for _ in range(self.n_samples):

            # Generate two trajectories using imported function
            # Add one step in case next_obs (next_state) are needed
            traj1 = rollout(self.env, policy, n_steps=self.n_steps + 1)
            traj2 = rollout(self.env, policy, n_steps=self.n_steps + 1)

            # Extract state-action pairs from trajectories
            obs1, states1, acts1 = extract_obs_state_actions(traj1)
            obs2, states2, acts2 = extract_obs_state_actions(traj2)

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
            state_seq_pairs.append((states1, states2))
            acts_seq_pairs.append((acts1, acts2))
            preferences.append(pref)
            
        return obs_seq_pairs, state_seq_pairs, acts_seq_pairs, preferences
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs1, obs2 = self.obs_seq_pairs[idx]
        states1, states2 = self.state_seq_pairs[idx]
        acts1, acts2 = self.acts_seq_pairs[idx]
        preference = self.preferences[idx]
        
        # Convert observations to tensors
        obs1_tensor = torch.tensor(obs1, dtype=torch.float32).to(self.device)
        obs2_tensor = torch.tensor(obs2, dtype=torch.float32).to(self.device)

        # Convert states to tensors
        states1_tensor = torch.tensor(states1, dtype=torch.int32).to(self.device)
        states2_tensor = torch.tensor(states2, dtype=torch.int32).to(self.device)

        # Convert actions to tensors
        acts1_tensor = torch.tensor(acts1, dtype=torch.float32).to(self.device)
        acts2_tensor = torch.tensor(acts2, dtype=torch.float32).to(self.device)
        
        # Concatenate obs and acts for easier handling
        obs_tensor = torch.stack([obs1_tensor[:-1], obs2_tensor[:-1]], dim=0)
        next_obs_tensor = torch.stack([obs1_tensor[1:], obs2_tensor[1:]], dim=0)
        state_tensor = torch.stack([states1_tensor[:-1], states2_tensor[:-1]], dim=0)
        next_state_tensor = torch.stack([states1_tensor[1:], states2_tensor[1:]], dim=0)
        acts_tensor = torch.stack([acts1_tensor, acts2_tensor], dim=0)
        preference = torch.tensor(preference, dtype=torch.float32).unsqueeze(0).to(self.device)
        return {
            "feedback_type": "preference",
            "state_features": obs_tensor,
            "next_obs": next_obs_tensor,
            "states": state_tensor,
            "next_states": next_state_tensor,
            "acts": acts_tensor,
            "targets": preference.squeeze(),
            "rationality": torch.tensor(self.rationality).to(self.device, dtype=torch.float32),
        }