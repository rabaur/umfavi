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
        obs_transform: Callable = None,
        act_transform: Callable = None,
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
        self.obs_transform = obs_transform
        self.act_transform = act_transform
        self.device = device
        self.rationality = rationality
        self.gamma = gamma
        # Generate demonstrations
        self.obs_seqs, self.states_seqs, self.acts_seqs = self.generate_demonstrations(policy=policy)
        self.td_error_weight = td_error_weight


    def add_demonstrations(self, policy: Callable) -> None:
        """
        Add demonstrations to the dataset.
        """
        obs_seqs, acts_seqs = self.generate_demonstrations(policy)
        if self.obs_seqs is None:
            self.obs_seqs, self.acts_seqs = obs_seqs, acts_seqs
        else:
            self.obs_seqs.extend(obs_seqs)
            self.acts_seqs.extend(acts_seqs)
        self.n_samples = len(self.obs_seqs)
    
    def generate_demonstrations(self, policy: Callable) -> tuple[list[list], list[list]]:
        """
        Generate expert demonstration trajectories.
        
        Returns:
            Tuple of (obs_seqs, acts_seqs) where:
            - obs_seqs: List of observation sequences
            - acts_seqs: List of action sequences
        """
        obs_seqs = []
        states_seqs = []
        acts_seqs = []
        
        for _ in range(self.n_samples):
            # Generate trajectory using the expert policy
            # Add one extra step to the trajectory to get the next observation
            trajectory = rollout(self.env, policy, n_steps=self.n_steps + 1)

            # Extract state-action pairs from trajectory
            obs, states, acts = extract_obs_state_actions(trajectory)

            # Transform observations and actions
            if self.obs_transform:
                obs = list(map(self.obs_transform, obs))
            
            if self.act_transform:
                acts = list(map(self.act_transform, acts))

            # Append the newly generated trajectory
            obs_seqs.append(obs)
            states_seqs.append(states)
            acts_seqs.append(acts)
            
        return obs_seqs, states_seqs, acts_seqs
    
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
            - obs: Observation sequence tensor
            - acts: Action sequence tensor (targets for behavioral cloning)
            - targets: Same as acts (for consistency with other datasets)
        """

        # Get the demonstration sequence
        obs = self.obs_seqs[idx][:-1]
        next_obs = self.obs_seqs[idx][1:]
        states = self.states_seqs[idx][:-1]
        next_states = self.states_seqs[idx][1:]
        acts = self.acts_seqs[idx][:-1]
        
        # Convert observations to tensors
        obs_tensor = torch.tensor(obs).to(self.device)
        next_obs_tensor = torch.tensor(next_obs).to(self.device)
        states_tensor = torch.tensor(states).to(self.device)
        next_states_tensor = torch.tensor(next_states).to(self.device)

        # Handle actions - if act_transform was applied, acts is a list of tensors
        if self.act_transform:
            # Stack the already transformed tensors
            acts_tensor = torch.stack(acts).to(self.device)
        else:
            # Convert to tensor if no transformation was applied
            acts_tensor = torch.tensor(acts).to(self.device)
        
        return {
            "feedback_type": "demonstration",
            "obs": obs_tensor,
            "next_obs": next_obs_tensor,
            "states": states_tensor,
            "next_states": next_states_tensor,
            "acts": acts_tensor,
            "targets": acts_tensor,
            "rationality": torch.tensor(self.rationality).to(self.device, dtype=torch.float32),
            "gamma": self.gamma,
            "td_error_weight": torch.tensor(self.td_error_weight).to(self.device, dtype=torch.float32),
        }