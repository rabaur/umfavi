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
        data = {k: [] for k in TrajKeys}
        
        for i in range(self.n_samples):

            # Generate trajectory using the expert policy
            # Add one extra step to the trajectory to get the next observation
            traj_demo = rollout(self.env, policy, num_steps=self.n_steps + 1)

            # Extract state-action pairs from trajectory
            traj_demo_data = unpack_trajectory(traj_demo)

            # Append the newly generated trajectory
            for k in traj_demo_data.keys():
                data[k].append(traj_demo_data[k])
        
        data = {k: np.stack(v, axis=0) for k, v in data.items()}

        # Apply transforms if provided. Transformations are applied per observation or action.
        if self.obs_transform:
            # Keep original states in the data
            data[SampleKey.STATES] = data[TrajKeys.OBS]

            # Apply transform to observations
            data[TrajKeys.OBS] = apply_transform(self.obs_transform, data[TrajKeys.OBS])

            # Keep original next states in the data
            data[SampleKey.NEXT_STATES] = data[TrajKeys.NEXT_OBS]

            # Apply transform to next observations
            data[TrajKeys.NEXT_OBS] = apply_transform(self.obs_transform, data[TrajKeys.NEXT_OBS])
        if self.act_transform:
            data[SampleKey.ACT_FEATS] = apply_transform(self.act_transform, data[TrajKeys.ACTS])
        
        return data
    
    def __len__(self):
        return self.n_samples
    
    def _to_torch(self, x: NDArray):
        return torch.tensor(x, dtype=torch.float32).to(self.device)
    
    def _drop_last_t(self, x: NDArray):
        """
        Drops data corresponding to last time-step.
        Assumes x has shape (..., T, F) where F is the feature dimension (may be 1).
        """
        return x[..., :-1, :]
    
    def _drop_first_t(self, x: NDArray):
        """
        Drops data corresponding to first time-step.
        Assumes x has shape (..., T, F) where F is the feature dimension (may be 1).
        """
        return x[..., 1:, :]
    
    def __getitem__(self, i) -> dict:
        """
        Get a demonstration sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with demonstration data using SampleKey enums.
        """

        # Get the demonstration sequence (drop last for current, drop first for next)
        obs = self.data[TrajKeys.OBS][i]
        obs_tensor = self._to_torch(self._drop_last_t(obs))
        next_obs_tensor = self._to_torch(self._drop_first_t(obs))
        
        # Get states if available, otherwise use observations
        if SampleKey.STATES in self.data:
            states = self.data[SampleKey.STATES][i]
            states_tensor = self._to_torch(self._drop_last_t(states))
            next_states_tensor = self._to_torch(self._drop_first_t(states))
        else:
            states_tensor = obs_tensor
            next_states_tensor = next_obs_tensor

        # Get actions
        actions = self.data[TrajKeys.ACTS][i]
        actions_tensor = self._to_torch(self._drop_last_t(actions))
        
        # Get action features if available (from transform or environment info)
        if SampleKey.ACT_FEATS in self.data:
            # From transform
            action_feats = self.data[SampleKey.ACT_FEATS][i]
            action_feats_tensor = self._to_torch(self._drop_last_t(action_feats)) 
        else:
            # Use raw actions as fallback
            actions_raw = self.data[TrajKeys.ACTS][i]
            action_feats_tensor = self._to_torch(self._drop_last_t(actions_raw))
        
        # Actions (discrete, no feature dimension)
        actions = self.data[TrajKeys.ACTS][i]
        actions_tensor = self._to_torch(self._drop_last_t(actions))
        
        # Dones (mask for valid/invalid timesteps)
        dones = self.data[TrajKeys.DONES][i]
        dones_tensor = self._to_torch(self._drop_last_t(dones))
        
        # Scalars
        rationality_tensor = self._to_torch(self.rationality)
        gamma_tensor = self._to_torch(self.gamma)
        td_error_weight_tensor = self._to_torch(self.td_error_weight)

        return {
            SampleKey.FEEDBACK_TYPE: FeedbackType.DEMONSTRATION,
            SampleKey.STATES: states_tensor,
            SampleKey.NEXT_STATES: next_states_tensor,
            SampleKey.OBS: obs_tensor,
            SampleKey.NEXT_OBS: next_obs_tensor,
            SampleKey.ACTS: actions_tensor,
            SampleKey.ACT_FEATS: action_feats_tensor,
            SampleKey.DONES: dones_tensor,
            SampleKey.RATIONALITY: rationality_tensor,
            SampleKey.GAMMA: gamma_tensor,
            SampleKey.TD_ERROR_WEIGHT: td_error_weight_tensor
        }