import torch
from torch import nn
from umfavi.loglikelihoods.base import BaseLogLikelihood
from umfavi.types import SampleKey

class DemonstrationsDecoder(BaseLogLikelihood):

    def __init__(self):
        """
        Args:
            Q_value_model: Q-value model, mapping states to Q-value estimates for all actions, i.e., Q : S -> R^{n_actions}.
        """
        super().__init__()

    def forward(self, reward_samples: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Computes the likelihood of demonstrations under a Boltzmann-rational policy.
        
        For transition-level data, each sample is a single (s, a) pair.
        The model predicts: π(a | s) ∝ exp(Q(s, a))
        
        Args:
            reward_samples: Float tensor of shape (batch_size,) or (batch_size, 1)
            **kwargs: Additional arguments including:
                - q_curr: Q-values tensor of shape (batch_size, n_actions)
                - acts: Actions tensor of shape (batch_size, 1) with long/int dtype
        
        Returns:
            Tuple of (nll, metrics_dict)
        """
        # Extract parameters from kwargs
        acts = kwargs[SampleKey.ACTS].long().squeeze(-1)  # (batch_size,)

        # Get the Q-value estimates
        q_curr = kwargs["q_curr"]  # (batch_size, n_actions)

        # ------------------------------------------------------------------------------------------------
        # Boltzmann-rational expert policy likelihood
        # ------------------------------------------------------------------------------------------------

        # Compute the log-likelihood of the demonstrations under the Boltzmann-rational expert policy
        # cross_entropy expects: (N, C) for input and (N,) for target
        demonstrations_nll = nn.functional.cross_entropy(q_curr, acts, reduction='mean')

        # Compute Q-value statistics for logging
        q_value_max = q_curr.max().item()
        q_value_min = q_curr.min().item()
        
        metrics = {
            "q_value_max": q_value_max,
            "q_value_min": q_value_min,
        }

        return demonstrations_nll, metrics