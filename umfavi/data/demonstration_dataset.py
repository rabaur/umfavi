import torch
import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset
from typing import Callable, Optional
import gymnasium as gym
from umfavi.utils.gym import rollout, unpack_trajectory, get_undiscounted_return
from umfavi.utils.feature_transforms import apply_transform
from umfavi.types import TrajKeys, SampleKey, FeedbackType


def print_dataset_stats(returns: list[float], traj_lengths: list[int], name: str) -> None:
    """Print concise statistics about the demonstration dataset."""
    returns = np.array(returns)
    lengths = np.array(traj_lengths)
    print(f"{name} Dataset: Demos: {len(returns)} | Steps: {lengths.sum()}")
    print(f"{name} Dataset: Return: {returns.mean():.1f} ± {returns.std():.1f} [{returns.min():.1f}, {returns.max():.1f}]")
    print(f"{name} Dataset: Length: {lengths.mean():.1f} ± {lengths.std():.1f} [{lengths.min()}, {lengths.max()}]")

class DemonstrationDataset(Dataset):
    """
    Dataset for demonstration learning with expert trajectories.
    """
    def __init__(
        self, 
        num_demonstrations: int,   
        policy: Callable,
        env: gym.Env,
        device: str,
        rationality: float = 1.0,
        gamma: float = 0.99,
        td_error_weight: float = 1.0,
        num_steps: Optional[int] = None,
        obs_transform: Optional[Callable] = None,
        act_transform: Optional[Callable] = None,
        name: Optional[str] = "train"
    ):
        """
        Initialize demonstration dataset.
        
        Args:
            num_demonstrations: Number of demonstration trajectories to generate.
            n_steps: Length of each trajectory
            policy: Expert policy to generate demonstrations
            env: Gymnasium environment
            device: Device to store tensors on ('cpu' or 'cuda')
            rationality: Rationality parameter for expert policy
            gamma: Discount factor for Q-value computation
            td_error_weight: Weight for TD-error constraint in demonstrations
            num_steps: Number of time-steps (Optional).
                If not provided, the policy will be rolled out until `done` is received from the environment.
                If `done` is received before `num_steps` steps, the remaining datapoints will be padded with nan-equivalent value.
            obs_transform: Optional transformation for observations
            act_transform: Optional transformation for actions
        """
        self.num_demonstrations = num_demonstrations
        self.num_steps = num_steps
        self.env = env
        self.device = device
        self.rationality = rationality
        self.gamma = gamma
        self.td_error_weight = td_error_weight
        self.obs_transform = obs_transform
        self.act_transform = act_transform
        self.name = name
        
        # Generate demonstrations
        self.data = self.generate_demonstrations(policy=policy)
    
    def generate_demonstrations(self, policy: Callable) -> dict:
        """
        Generate expert demonstration trajectories.
        
        Returns:
            Dictionary with trajectory data using TrajectoryKey enums.
        """

        # Datastructure of ragged arrays
        data: dict[SampleKey: list[NDArray]] = {k: [] for k in TrajKeys}
        data[SampleKey.ACTS] = []
        data[SampleKey.NEXT_ACTS] = []
        
        # Add one extra step to the trajectory to get the next observation
        num_steps = self.num_steps + 1 if self.num_steps else None
        returns, traj_lengths = [], []
        for i in range(self.num_demonstrations):

            # Generate trajectory using the expert policy
            traj_demo = rollout(self.env, policy, num_steps=num_steps)

            # Extract state-action pairs from trajectory
            traj_demo_data = unpack_trajectory(traj_demo)

            returns.append(get_undiscounted_return(traj_demo))
            traj_lengths.append(len(traj_demo))

            # Differentiate between actions and next-actions, since not returned explicitly by the environment
            acts_full = traj_demo_data[SampleKey.ACTS]

            # Add dummy action for next-action
            next_acts = np.concatenate([acts_full[1:], np.array([-1])[:, None]], axis=0)
            data[SampleKey.NEXT_ACTS].append(next_acts)

            # Append the newly generated trajectory
            for k in traj_demo_data.keys():
                data[k.value].append(traj_demo_data[k])
        
        print_dataset_stats(returns, traj_lengths, self.name)
        
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

        # Assertion: Arrays belonging to the same trajectory should have the same length.
        # Save lengths for easier indexing later.
        self.demo_lengths = []
        arrays_by_traj = zip(*data.values())
        for i, As in enumerate(arrays_by_traj):
            first_len = len(As[0])
            assert all([len(a) == first_len for a in As]), f"Lengths of data for trajectory {i} don't match."
            self.demo_lengths.append(first_len)
        
        # Cumulative sum for easier indexing (prepend 0 for correct offset calculation)
        self.cumsum_demo_lengths = np.concatenate([[0], np.cumsum(self.demo_lengths)])
        
        return data
    
    def __len__(self):
        # Return total number of transitions (not trajectories)
        return int(self.cumsum_demo_lengths[-1])
    
    def _to_torch(self, x: NDArray):
        return torch.tensor(x, dtype=torch.float32).to(self.device)
    
    def __getitem__(self, i) -> dict:
        """
        Get a single (s, a, s', a') transition sample.
        Trajectories and time-steps are treated independently.
        
        Args:
            i: Index of the transition (0 to total_transitions - 1)
            
        Returns:
            Dictionary with demonstration data using SampleKey enums.
        """
        # Find which trajectory this transition belongs to
        demo_idx = np.searchsorted(self.cumsum_demo_lengths[1:], i, side='right')
        
        # Find offset within that trajectory
        offset = i - self.cumsum_demo_lengths[demo_idx]
        
        # Build the sample dictionary
        item_dict = {
            # Metadata
            SampleKey.FEEDBACK_TYPE: FeedbackType.DEMONSTRATION,
            SampleKey.RATIONALITY: self._to_torch(self.rationality),
            SampleKey.GAMMA: self._to_torch(self.gamma),
            SampleKey.TD_ERROR_WEIGHT: self._to_torch(self.td_error_weight),
        }
            
        # Add remaining fields from data
        for k in self.data.keys():
            if k not in item_dict:
                item_dict[k] = self._to_torch(self.data[k][demo_idx][offset])
        
        return item_dict