from numpy.typing import NDArray
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Callable, Optional, TypedDict
import gymnasium as gym
from umfavi.utils.gym import rollout, unpack_trajectory
from umfavi.utils.math import sigmoid
from umfavi.utils.feature_transforms import apply_transform
from umfavi.types import ObsType, ActType, TrajKeys, SampleKey, FeedbackType

class PreferenceSample(TypedDict):
    feedback_type: str
    states: torch.Tensor              # (2, T, state_dim)
    next_states: torch.Tensor         # (2, T, state_dim)
    obs: torch.Tensor                 # (2, T, obs_dim)
    next_obs: torch.Tensor            # (2, T, obs_dim)
    actions: torch.Tensor             # (2, T, 1) - discrete actions with feature_dim=1
    action_features: torch.Tensor     # (2, T, action_feature_dim)
    dones: torch.Tensor               # (2, T, 1) - boolean mask for valid timesteps
    rationality: torch.Tensor         # scalar
    gamma: torch.Tensor               # scalar
    preference: torch.Tensor          # scalar

class PreferenceDataset(Dataset):
    """
    Dataset for preference learning with trajectory pairs and simulated preferences.
    """
    def __init__(
        self, 
        n_samples: int,   
        n_steps: int,
        policy: Callable[[ObsType], ActType],
        env: gym.Env,
        device: str,
        rationality: float = 1.0,
        gamma: float = 0.99,
        obs_transform: Optional[Callable[[ObsType], ObsType]] = None,
        act_transform: Optional[Callable[[ActType], ActType]] = None,
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
        self.obs_transform = obs_transform
        self.act_transform = act_transform

        # Generate trajectory pairs and preferences
        self.data = self.generate_preferences(policy=policy)
    
    def generate_preferences(self, policy: Callable) -> tuple[list[list[dict]], list[list[dict]], list[int]]:
        """
        Generate trajectory pairs and preferences.
        
        Returns:
            Tuple of (trajectory_pairs, preferences) where:
            - trajectory_pairs: List of (traj1, traj2) pairs
            - preferences: List of preferences (0 for traj1, 1 for traj2)
        """
        data = {k: [] for k in TrajKeys}
        data[SampleKey.PREFERENCE] = []
        
        for _ in range(self.n_samples):

            # Generate two trajectories using imported function
            # Add one step in case next_obs (next_state) are needed
            traj1 = rollout(self.env, policy, n_steps=self.n_steps + 1)
            traj2 = rollout(self.env, policy, n_steps=self.n_steps + 1)

            # Extract state-action pairs
            traj1_dict = unpack_trajectory(traj1)
            traj2_dict = unpack_trajectory(traj2)

            # Compute true returns
            rews1 = np.array(traj1_dict[TrajKeys.REWS])
            rews2 = np.array(traj2_dict[TrajKeys.REWS])

            r1 = np.nansum(rews1, dtype=np.float32)
            r2 = np.nansum(rews2, dtype=np.float32)

            # Generate preference using sigmoid
            preference_prob = sigmoid(self.rationality * (r1 - r2))
            data[SampleKey.PREFERENCE].append(preference_prob)
            
            # Append the newly generated trajectories
            for k in traj1_dict.keys():
                data[k].append(np.stack([traj1_dict[k], traj2_dict[k]], axis=0))
        
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
            # Create action features by applying transform
            # Use apply_transform to handle transforms that return arrays (e.g., one-hot encoding)
            transformed_actions = apply_transform(self.act_transform, data[TrajKeys.ACTS])
            data[SampleKey.ACT_FEATS] = transformed_actions
            
        return data
    
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
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, i) -> PreferenceSample:
        """
        Gets a single preference feedback sample.

        Returns:
            The i-th PreferenceSample.
        """

        # Observations
        obs = self.data[TrajKeys.OBS][i]
        obs_tensor = self._to_torch(self._drop_last_t(obs))
        next_obs_tensor = self._to_torch(self._drop_first_t(obs))
        
        # States (from environment info dict, if available)
        if SampleKey.STATES in self.data:
            states = self.data[SampleKey.STATES][i]
            states_tensor = self._to_torch(self._drop_last_t(states))
            next_states_tensor = self._to_torch(self._drop_first_t(states))
        else:
            # If no separate states, use observations as states
            states_tensor = obs_tensor
            next_states_tensor = next_obs_tensor

        # Action features (from transform, if available)
        if SampleKey.ACT_FEATS in self.data:
            action_feats = self.data[SampleKey.ACT_FEATS][i]
            action_feats_tensor = self._to_torch(self._drop_last_t(action_feats))
        else:
            # If no action features, use raw actions
            actions_raw = self.data[TrajKeys.ACTS][i]
            action_feats_tensor = self._to_torch(self._drop_last_t(actions_raw))

        # Actions (discrete, no feature dimension)
        actions = self.data[TrajKeys.ACTS][i]
        actions_tensor = self._to_torch(self._drop_last_t(actions))
        
        # Dones (mask for valid/invalid timesteps)
        dones = self.data[TrajKeys.DONES][i]
        dones_tensor = self._to_torch(self._drop_last_t(dones))
        
        # Scalars
        preference_tensor = self._to_torch(self.data[SampleKey.PREFERENCE][i])
        rationality_tensor = self._to_torch(self.rationality)
        gamma_tensor = self._to_torch(self.gamma)
        
        return {
            SampleKey.FEEDBACK_TYPE: FeedbackType.PREFERENCE,
            SampleKey.STATES: states_tensor,
            SampleKey.NEXT_STATES: next_states_tensor,
            SampleKey.OBS: obs_tensor,
            SampleKey.NEXT_OBS: next_obs_tensor,
            SampleKey.ACTS: actions_tensor,
            SampleKey.ACT_FEATS: action_feats_tensor,
            SampleKey.DONES: dones_tensor,
            SampleKey.RATIONALITY: rationality_tensor,
            SampleKey.GAMMA: gamma_tensor,
            SampleKey.PREFERENCE: preference_tensor
        }