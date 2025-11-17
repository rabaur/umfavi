import torch
import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset
from typing import Callable
import gymnasium as gym
from umfavi.utils.gym import extract_obs_state_actions, rollout

class DemonstrationDataset(Dataset):
    """
    Dataset for demonstration learning with expert trajectories.
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
        td_error_weight: float = 1.0
    ):
        """
        Initialize demonstration dataset.
        
        Args:
            n_samples: Number of demonstration trajectories to generate
            n_steps: Length of each trajectory
            policy: Expert policy to generate demonstrations
            env: Gymnasium environment
            device: Device to store tensors on ('cpu' or 'cuda')
            obs_transform: Optional transformation for observations
            act_transform: Optional transformation for actions
            rationality: Rationality parameter for expert policy
            gamma: Discount factor for Q-value computation
            td_error_weight: Weight for TD-error constraint in demonstrations
        """
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.env = env
        self.device = device
        self.rationality = rationality
        self.gamma = gamma
        self.td_error_weight = td_error_weight
        
        # Generate demonstrations
        demos = self.generate_demonstrations(policy=policy)
        self.state_feats = demos["state_feats"]
        self.states = demos["states"]
        self.act_feats = demos["act_feats"]
        self.acts = demos["acts"]
    
    def generate_demonstrations(self, policy: Callable) -> dict:
        """
        Generate expert demonstration trajectories.
        
        Returns:
            Dictionary with:
            - state_feats: List of state feature sequences
            - states: List of state sequences
            - act_feats: List of action feature sequences
            - acts: List of action sequences
        """
        data = {
            "state_feats": [],
            "states": [],
            "act_feats": [],
            "acts": [],
        }
        
        for _ in range(self.n_samples):
            # Generate trajectory using the expert policy
            # Add one extra step to the trajectory to get the next observation
            trajectory = rollout(self.env, policy, n_steps=self.n_steps + 1)

            # Extract state-action pairs from trajectory
            traj_data = extract_obs_state_actions(trajectory, self.env)

            # Append the newly generated trajectory
            data["state_feats"].append(traj_data["state_feats"])
            data["states"].append(traj_data["states"])
            data["act_feats"].append(traj_data["act_feats"])
            data["acts"].append(traj_data["acts"])
        
        return data
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx) -> dict:
        """
        Get a demonstration sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with:
            - feedback_type: "demonstration"
            - state_features: State features tensor
            - acts: Action sequence tensor (targets for behavioral cloning)
            - targets: Same as acts (for consistency with other datasets)
        """

        # Get the demonstration sequence
        state_feats = self.state_feats[idx][:-1]
        next_state_feats = self.state_feats[idx][1:]
        states = self.states[idx][:-1]
        next_states = self.states[idx][1:]
        acts = self.acts[idx][:-1]
        act_feats = self.act_feats[idx][:-1]
        
        # Convert observations to tensors
        state_feats_tensor = torch.tensor(np.array(state_feats)).to(self.device)
        next_state_feats_tensor = torch.tensor(np.array(next_state_feats)).to(self.device)
        states_tensor = torch.tensor(np.array(states)).to(self.device)
        next_states_tensor = torch.tensor(np.array(next_states)).to(self.device)

        # Convert actions to tensors
        acts_tensor = torch.tensor(np.array(acts)).to(self.device)
        act_feats_tensor = torch.tensor(np.array(act_feats)).to(self.device)
        
        return {
            "feedback_type": "demonstration",
            "states": states_tensor,
            "state_features": state_feats_tensor,
            "next_states": next_states_tensor,
            "next_state_features": next_state_feats_tensor,
            "actions": acts_tensor,
            "action_features": act_feats_tensor,
            "targets": acts_tensor,
            "rationality": torch.tensor(self.rationality).to(self.device, dtype=torch.float32),
            "gamma": torch.tensor(self.gamma).to(self.device, dtype=torch.float32),
            "td_error_weight": torch.tensor(self.td_error_weight).to(self.device, dtype=torch.float32),
        }