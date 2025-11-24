import torch
import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset
from typing import Callable, Optional
import gymnasium as gym
from umfavi.utils.gym import unpack_trajectory, rollout
from umfavi.utils.feature_transforms import apply_transform
from umfavi.types import TrajKeys, SampleKey, FeedbackType

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
        td_error_weight: float = 1.0,
        obs_transform: Optional[Callable] = None,
        act_transform: Optional[Callable] = None
    ):
        """
        Initialize demonstration dataset.
        
        Args:
            n_samples: Number of demonstration trajectories to generate
            n_steps: Length of each trajectory
            policy: Expert policy to generate demonstrations
            env: Gymnasium environment
            device: Device to store tensors on ('cpu' or 'cuda')
            rationality: Rationality parameter for expert policy
            gamma: Discount factor for Q-value computation
            td_error_weight: Weight for TD-error constraint in demonstrations
            obs_transform: Optional transformation for observations
            act_transform: Optional transformation for actions
        """
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.env = env
        self.device = device
        self.rationality = rationality
        self.gamma = gamma
        self.td_error_weight = td_error_weight
        self.obs_transform = obs_transform
        self.act_transform = act_transform
        
        # Generate demonstrations
        self.data = self.generate_demonstrations(policy=policy)
    
    def generate_demonstrations(self, policy: Callable) -> dict:
        """
        Generate expert demonstration trajectories.
        
        Returns:
            Dictionary with trajectory data using TrajectoryKey enums.
        """
        data = {
            TrajKeys.OBS: [],
            TrajKeys.ACTS: [],
        }
        
        # Also collect states and action features if available from env info
        collect_states = False
        collect_action_features = False
        
        for i in range(self.n_samples):
            # Generate trajectory using the expert policy
            # Add one extra step to the trajectory to get the next observation
            trajectory = rollout(self.env, policy, n_steps=self.n_steps + 1)

            # Extract state-action pairs from trajectory
            traj_data = unpack_trajectory(trajectory)

            # First iteration: check what keys are available
            if i == 0:
                collect_states = "states" in traj_data
                collect_action_features = "action_features" in traj_data
                if collect_states:
                    data["states"] = []
                if collect_action_features:
                    data["action_features"] = []

            # Append the newly generated trajectory
            data[TrajKeys.OBS].append(traj_data[TrajKeys.OBS])
            data[TrajKeys.ACTS].append(traj_data[TrajKeys.ACTS])
            
            if collect_states:
                data["states"].append(traj_data["states"])
            if collect_action_features:
                data["action_features"].append(traj_data["action_features"])
        
        # Stack all trajectories
        data = {k: np.stack(v, axis=0) for k, v in data.items()}
        
        # Apply transforms if provided
        if self.obs_transform:
            data[TrajKeys.OBS] = np.vectorize(self.obs_transform)(data[TrajKeys.OBS])
        
        if self.act_transform:
            # Use apply_transform to handle transforms that return arrays (e.g., one-hot encoding)
            transformed_actions = apply_transform(self.act_transform, data[TrajKeys.ACTS])
            data[SampleKey.ACT_FEATS] = transformed_actions
        
        return data
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx) -> dict:
        """
        Get a demonstration sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with demonstration data using SampleKey enums.
        """

        # Get the demonstration sequence (drop last for current, drop first for next)
        obs = self.data[TrajKeys.OBS][idx]
        obs_tensor = torch.tensor(np.array(obs[:-1])).to(self.device)
        next_obs_tensor = torch.tensor(np.array(obs[1:])).to(self.device)
        
        # Get states if available, otherwise use observations
        if "states" in self.data:
            states = self.data["states"][idx]
            states_tensor = torch.tensor(np.array(states[:-1])).to(self.device)
            next_states_tensor = torch.tensor(np.array(states[1:])).to(self.device)
        else:
            states_tensor = obs_tensor
            next_states_tensor = next_obs_tensor

        # Get actions
        actions = self.data[TrajKeys.ACTS][idx]
        actions_tensor = torch.tensor(np.array(actions[:-1])).to(self.device)
        
        # Get action features if available (from transform or environment info)
        if SampleKey.ACT_FEATS in self.data:
            # From transform
            action_feats = self.data[SampleKey.ACT_FEATS][idx]
            action_feats_tensor = torch.tensor(np.array(action_feats[:-1])).to(self.device)
        elif "action_features" in self.data:
            # From environment info
            action_feats = self.data["action_features"][idx]
            action_feats_tensor = torch.tensor(np.array(action_feats[:-1])).to(self.device)
        else:
            # Use raw actions as fallback
            action_feats_tensor = actions_tensor
        
        return {
            SampleKey.FEEDBACK_TYPE: FeedbackType.DEMONSTRATION,
            SampleKey.STATES: states_tensor,
            SampleKey.OBS: obs_tensor,
            SampleKey.NEXT_STATES: next_states_tensor,
            SampleKey.NEXT_OBS: next_obs_tensor,
            SampleKey.ACTS: actions_tensor,
            SampleKey.ACT_FEATS: action_feats_tensor,
            SampleKey.TARGETS: actions_tensor,
            SampleKey.RATIONALITY: torch.tensor(self.rationality).to(self.device, dtype=torch.float32),
            SampleKey.GAMMA: torch.tensor(self.gamma).to(self.device, dtype=torch.float32),
            SampleKey.TD_ERROR_WEIGHT: torch.tensor(self.td_error_weight).to(self.device, dtype=torch.float32),
        }