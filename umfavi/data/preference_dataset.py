import torch
import numpy as np
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
        gamma: float = 0.99,
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
        self.gamma = gamma

        # Generate trajectory pairs and preferences
        prefs = self.generate_preferences(policy=policy)
        self.state_feats = self.state_feats = prefs["state_feats"]
        self.states = prefs["states"]
        self.act_feats = prefs["act_feats"]
        self.acts = prefs["acts"]
        self.preferences = prefs["preferences"]
    
    def generate_preferences(self, policy: Callable) -> tuple[list[list[dict]], list[list[dict]], list[int]]:
        """
        Generate trajectory pairs and preferences.
        
        Returns:
            Tuple of (trajectory_pairs, preferences) where:
            - trajectory_pairs: List of (traj1, traj2) pairs
            - preferences: List of preferences (0 for traj1, 1 for traj2)
        """
        preferences = {
            "state_feats": [],
            "states": [],
            "act_feats": [],
            "acts": [],
            "preferences": []
        }
        
        for _ in range(self.n_samples):

            # Generate two trajectories using imported function
            # Add one step in case next_obs (next_state) are needed
            traj1 = rollout(self.env, policy, n_steps=self.n_steps + 1)
            traj2 = rollout(self.env, policy, n_steps=self.n_steps + 1)

            # Extract state-action pairs
            traj1_data = extract_obs_state_actions(traj1, self.env)
            traj2_data = extract_obs_state_actions(traj2, self.env)

            # Compute true returns
            rews1 = get_rewards(traj1)
            rews2 = get_rewards(traj2)
            rew_sum1 = sum(rews1)
            rew_sum2 = sum(rews2)

            # Generate preference using sigmoid
            preference_prob = sigmoid(self.rationality * (rew_sum2 - rew_sum1))
            pref = np.random.binomial(1, preference_prob)
            
            # Append the newly generated trajectories
            preferences["state_feats"].append(np.stack([traj1_data["state_feats"], traj2_data["state_feats"]], axis=0))
            preferences["states"].append(np.stack([traj1_data["states"], traj2_data["states"]], axis=0))
            preferences["act_feats"].append(np.stack([traj1_data["act_feats"], traj2_data["act_feats"]], axis=0))
            preferences["acts"].append(np.stack([traj1_data["acts"], traj2_data["acts"]], axis=0))
            preferences["preferences"].append(preference_prob)
        
        preferences = {k: np.stack(v, axis=0) for k, v in preferences.items()}
            
        return preferences
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_feats_tensor = torch.tensor(self.state_feats[idx][..., :-1, :]).to(self.device)
        next_state_feats_tensor = torch.tensor(self.state_feats[idx][..., 1:, :]).to(self.device)
        states_tensor = torch.tensor(self.states[idx][..., :-1, :]).to(self.device)
        next_states_tensor = torch.tensor(self.states[idx][..., 1:, :]).to(self.device)
        act_feats_tensor = torch.tensor(self.act_feats[idx][..., :-1, :]).to(self.device)

        # Does not have feature dimension
        acts_tensor = torch.tensor(self.acts[idx][..., :-1]).to(self.device)
        preference_tensor = torch.tensor(self.preferences[idx]).to(self.device, dtype=torch.float32)
        rationality_tensor = torch.tensor(self.rationality).to(self.device, dtype=torch.float32)
        gamma_tensor = torch.tensor(self.gamma).to(self.device, dtype=torch.float32)
        return {
            "feedback_type": "preference",
            "states": states_tensor,
            "state_features": state_feats_tensor,
            "next_states": next_states_tensor,
            "next_state_features": next_state_feats_tensor,
            "actions": acts_tensor,
            "action_features": act_feats_tensor,
            "targets": preference_tensor,
            "rationality": rationality_tensor,
            "gamma": gamma_tensor,
        }