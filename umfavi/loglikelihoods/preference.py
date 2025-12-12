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
        invalid = kwargs[SampleKey.INVALID]
        
        # Zero out rewards at invalid timesteps (use masked_fill to avoid in-place modification)
        reward_samples = reward_samples.masked_fill(invalid.unsqueeze(-1).bool(), 0.0)
        cum_rews_per_traj = torch.sum(reward_samples, dim=-2).squeeze(-1)  # (batch_size, 2)
        logits = rationality * (cum_rews_per_traj[..., 0] - cum_rews_per_traj[..., 1])
        return nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='mean'), {}