from numpy.typing import NDArray
from sympy import I
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Callable, Optional, TypedDict
import gymnasium as gym
from umfavi.utils.gym import rollout, get_undiscounted_return
from umfavi.utils.math import sigmoid
from umfavi.utils.feature_transforms import apply_transform
from umfavi.types import ObsType, ActType, TrajKeys, SampleKey, FeedbackType
import matplotlib.pyplot as plt
from tqdm import tqdm


def print_preference_stats(preferences: np.ndarray, cum_rews: np.ndarray, name: str, threshold: float = 0.3) -> None:
    """Print statistics about the preference dataset.
    
    Args:
        preferences: Array of preference probabilities (probability that traj1 > traj2)
        cum_rews: Array of cumulative rewards for each pair, shape (num_samples, 2)
        name: Dataset name for display
        threshold: Distance from 0.5 to consider a preference "meaningful" (default 0.3 means <0.2 or >0.8)
    """
    n_total = len(preferences)
    
    # Meaningful pairs: preference probability far from 0.5
    dist_from_half = np.abs(preferences - 0.5)
    meaningful_mask = dist_from_half >= threshold
    n_meaningful = meaningful_mask.sum()
    
    # Return difference statistics
    return_diffs = cum_rews[:, 0] - cum_rews[:, 1]
    
    print(f"{name} Preference Dataset: Pairs: {n_total} | Meaningful (|p-0.5|≥{threshold}): {n_meaningful} ({100*n_meaningful/n_total:.1f}%)")
    print(f"{name} Preference Dataset: p(traj1>traj2): {preferences.mean():.2f} ± {preferences.std():.2f} [{preferences.min():.2f}, {preferences.max():.2f}]")
    print(f"{name} Preference Dataset: Return diff (r1-r2): {return_diffs.mean():.1f} ± {return_diffs.std():.1f} [{return_diffs.min():.1f}, {return_diffs.max():.1f}]")

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
        num_samples: int,   
        num_steps: int,
        policy: Callable[[ObsType], ActType],
        env: gym.Env,
        device: str,
        rationality: float = 1.0,
        gamma: float = 0.99,
        obs_transform: Optional[Callable[[ObsType], ObsType]] = None,
        act_transform: Optional[Callable[[ActType], ActType]] = None,
        name: Optional[str] = "train",
        subsample_factor: int = 1,
    ):
        """
        Initialize preference dataset.
        
        Args:
            num_samples: Number of preference samples to generate
            init_state_dist: Initial state distribution
            init_policy: Initial policy
            env: Environment
        """
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.env = env
        self.rationality = rationality
        self.device = device
        self.gamma = gamma
        self.obs_transform = obs_transform
        self.act_transform = act_transform
        self.name = name
        self.subsample_factor = subsample_factor

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
        data[SampleKey.NEXT_ACTS] = []
        data[SampleKey.PREFERENCE] = []
        data["cum_rews"] = []
        
        for i in tqdm(range(self.num_samples), desc="Generating preferences"):

            # Generate two trajectories
            # Use different seeds for each trajectory to get diverse pairs
            traj1_data = rollout(self.env, policy, num_steps=self.num_steps, seed=i)
            traj2_data = rollout(self.env, policy, num_steps=self.num_steps, seed=i)

            # Check that states are different
            assert not np.allclose(traj1_data[TrajKeys.OBS], traj2_data[TrajKeys.OBS]), "States are the same"

            r1 = get_undiscounted_return(traj1_data)
            r2 = get_undiscounted_return(traj2_data)
            data["cum_rews"].append(np.array([r1, r2]))

            # Generate preference using sigmoid
            preference_prob = sigmoid(self.rationality * (r1 - r2))
            data[SampleKey.PREFERENCE].append(preference_prob)

            # Add next actions
            traj1_next_acts = np.concatenate([traj1_data[TrajKeys.ACTS][1:], np.array([-1])[:, None]])
            traj1_data[SampleKey.NEXT_ACTS] = traj1_next_acts
            traj2_next_acts = np.concatenate([traj2_data[TrajKeys.ACTS][1:], np.array([-1])[:, None]])
            traj2_data[SampleKey.NEXT_ACTS] = traj2_next_acts

            # Combine pairs
            combined_data = {k: np.stack([traj1_data[k], traj2_data[k]], axis=0) for k in traj1_data.keys()}
            
            # Sub-sample: keep every k-th transition
            if self.subsample_factor > 1:
                for k in combined_data.keys():
                    combined_data[k] = combined_data[k][::self.subsample_factor]

            # Add data
            for k in combined_data.keys():
                data[k].append(combined_data[k])
        
        data = {k: np.stack(v, axis=0) for k, v in data.items()}

                # Initialize states and action_features as copies of observations and actions
        # (they may be transformed later)
        data[SampleKey.STATES] = data[TrajKeys.OBS]
        data[SampleKey.NEXT_STATES] = data[TrajKeys.NEXT_OBS]
        data[SampleKey.ACT_FEATS] = data[SampleKey.ACTS]
        data[SampleKey.NEXT_ACT_FEATS] = data[SampleKey.NEXT_ACTS]

        # Apply transforms if provided. Transformations are applied per observation or action.
        # TODO: Make this more efficient.
        # For larger datasets this should be sped up by avoiding duplicate computation
        # (states and next_states only differ by one state each)
        if self.obs_transform:
            
            # Apply transform to observations
            data[TrajKeys.OBS] = \
                [apply_transform(self.obs_transform, obs) for obs in data[SampleKey.STATES]]
            
            # Apply transform to next observations
            data[TrajKeys.NEXT_OBS] = \
                [apply_transform(self.obs_transform, next_obs) for next_obs in data[SampleKey.NEXT_STATES]]

        if self.act_transform:
            # Apply transform to actions
            data[SampleKey.ACT_FEATS] = \
                [apply_transform(self.act_transform, acts) for acts in data[TrajKeys.ACTS]]
            
            # Apply transform to next actions
            data[SampleKey.NEXT_ACT_FEATS] = \
                [apply_transform(self.act_transform, next_acts) for next_acts in data[SampleKey.NEXT_ACTS]]
        
        # Print dataset statistics
        print_preference_stats(
            preferences=data[SampleKey.PREFERENCE],
            cum_rews=data["cum_rews"],
            name=self.name
        )
            
        return data
    
    def _to_torch(self, x: NDArray):
        return torch.tensor(x, dtype=torch.float32).to(self.device)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx) -> PreferenceSample:
        """
        Gets a single preference feedback sample.

        Returns:
            The i-th PreferenceSample.
        """
        # Scalars
        item_dict = {
            SampleKey.FEEDBACK_TYPE: FeedbackType.PREFERENCE,
            SampleKey.PREFERENCE: self._to_torch(self.data[SampleKey.PREFERENCE][idx]),
            SampleKey.RATIONALITY: self._to_torch(self.rationality),
            SampleKey.GAMMA: self._to_torch(self.gamma)
        }

        # Add remaining fields from data
        for k in self.data.keys():
            if k not in item_dict:
                item_dict[k] = self._to_torch(self.data[k][idx])
        
        return item_dict