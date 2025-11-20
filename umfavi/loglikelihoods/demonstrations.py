import torch
import torch.nn.functional as F
from torch import nn
from umfavi.loglikelihoods.base import BaseLogLikelihood
from umfavi.utils.math import log_var_to_std
class DemonstrationsDecoder(BaseLogLikelihood):

    def __init__(self):
        """
        Args:
            Q_value_model: Q-value model, mapping states to Q-value estimates for all actions, i.e., Q : S -> R^{n_actions}.
        """
        super().__init__()

    def forward(self, reward_samples: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        The likelihood of the AVRIL demonstrations has two terms:
        - The likelihood of the demonstrations under the Boltzmann-rational expert policy:
            ∑_{s, a} log π(a | s) = ∑_{s, a} log (exp(βQ(s, a)) / ∑_{a'} exp(βQ(s, a')))
        Args:
            reward_samples: Float tensor of shape (batch_size, num_steps).
            **kwargs: Additional arguments including:
                - obs: Float tensor of shape (batch_size, num_steps, obs_dim).
                - acts: Float tensor of shape (batch_size, num_steps, act_dim).
                - reward_mean: Float tensor of shape (batch_size, num_steps) - mean of reward distribution.
                - reward_log_var: Float tensor of shape (batch_size, num_steps) - log variance of reward distribution.
                - rationality: Rationality coefficient (default: 1.0).
                - gamma: Discount factor (default: 0.99).
                - td_error_weight: Weight for the TD-error constraint (default: 1.0).
        Returns:
            ...
        """
        # Extract parameters from kwargs
        acts = kwargs["actions"].long()

        # Get the Q-value estimates
        q_values = kwargs["q_values"]  # (batch_size, num_steps, n_actions)

        # ------------------------------------------------------------------------------------------------
        # Boltzmann-rational expert policy likelihood
        # ------------------------------------------------------------------------------------------------

        # Compute the log-likelihood of the demonstrations under the Boltzmann-rational expert policy
        logits = q_values # rationality * q_values  # (batch_size, num_steps, n_actions)

        # Shuffle logits to (batch_size, n_actions, num_steps) since expects (N, C, d1, d2, ...) shape
        logits = logits.permute(0, 2, 1)
        demonstrations_nll = nn.functional.cross_entropy(logits, acts.squeeze(), reduction='none').mean()

        # Compute Q-value statistics for logging
        q_value_max = q_values.max().item()
        q_value_min = q_values.min().item()
        
        metrics = {
            "q_value_max": q_value_max,
            "q_value_min": q_value_min,
        }

        return demonstrations_nll, metrics