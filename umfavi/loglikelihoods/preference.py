import torch
from torch import nn
from umfavi.loglikelihoods.base import BaseLogLikelihood
from umfavi.types import SampleKey

class PreferenceDecoder(BaseLogLikelihood):
    """Head for predicting preferences."""

    def __init__(self):
        super().__init__()

    def forward(self, reward_samples: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            reward_samples: Float tensor of shape (batch_size, 2, num_steps, 1).
                reward_samples[:, 0, ...] corresponds to the rewards for the first trajectory, reward_samples[:, 1, ...] to the second
            **kwargs: Additional arguments including:
                - rationality: Rationality coefficient (default: 1.0)
                - targets: Preference targets tensor
                - dones: Boolean mask for valid timesteps (batch_size, 2, num_steps, 1)
        Returns:
            Preference tensor of shape (batch_size)
        """
        rationality = kwargs[SampleKey.RATIONALITY]
        targets = kwargs[SampleKey.PREFERENCE]
        dones = kwargs[SampleKey.DONES]
        
        # Mask out invalid timesteps (where done=True) before summing
        # Create mask: True for valid timesteps
        valid_mask = dones.bool()  # Shape: (batch_size, 2, num_steps, 1)
        
        # Zero out rewards at invalid timesteps
        reward_samples[..., valid_mask] = 0.0
        cum_rews_per_traj = reward_samples.sum(dim=-2)  # (batch_size, 2, 1)
        cum_rews1 = cum_rews_per_traj[:, 0].squeeze()  # (batch_size,)
        cum_rews2 = cum_rews_per_traj[:, 1].squeeze()  # (batch_size,)

        logits = rationality * (cum_rews1 - cum_rews2)
        return nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none').mean(), {}