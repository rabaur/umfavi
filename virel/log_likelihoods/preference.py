import torch
from torch import nn
from virel.log_likelihoods.base_log_likelihood import BaseLogLikelihood

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
        Returns:
            Preference tensor of shape (batch_size)
        """
        rationality = kwargs.get("rationality", 1.0)
        targets = kwargs["targets"]
        
        cum_rews_per_traj = torch.sum(reward_samples, dim=2)
        cum_rews1 = cum_rews_per_traj[:, 0].squeeze()
        cum_rews2 = cum_rews_per_traj[:, 1].squeeze()
        logits = rationality * (cum_rews2 - cum_rews1)
        return nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='sum')