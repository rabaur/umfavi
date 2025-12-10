import torch
from torch import nn
from typing import Any
from umfavi.loglikelihoods.base import BaseLogLikelihood
from umfavi.priors import kl_divergence_std_normal
from umfavi.encoder.reward_encoder import BaseRewardEncoder
from umfavi.regularizer.td_error import td_error_regularizer
from umfavi.types import FeedbackType
from umfavi.types import SampleKey

class MultiFeedbackTypeModel(nn.Module):

    def __init__(self, encoder: BaseRewardEncoder, Q_value_model: nn.Module, decoders: dict[FeedbackType, BaseLogLikelihood]):
        super().__init__()
        self.encoder = encoder
        self.Q_value_model = Q_value_model
        self.decoders = nn.ModuleDict(decoders)
    
    def forward(self, **kwargs) -> Any:

        # Encode
        obs = kwargs[SampleKey.OBS]
        next_obs = kwargs[SampleKey.NEXT_OBS]
        action_feats = kwargs[SampleKey.ACT_FEATS]
        # next_action_feats = kwargs[SampleKey.NEXT_ACT_FEATS]
        next_action_feats = None
        dones = kwargs[SampleKey.DONES]
        
        mean, log_var = self.encoder(obs, action_feats, next_obs)
        reward_samples = self.encoder.sample(mean, log_var)
        kl_div = kl_divergence_std_normal(mean, log_var, dones)

        # Route to appropriate head with all kwargs
        head = self.decoders[kwargs[SampleKey.FEEDBACK_TYPE][0]]  # we can assume that all feedback types are the same per batch
        
        # Add reward mean and log_var to kwargs for decoders that need them
        kwargs["reward_mean"] = mean.squeeze(-1)
        kwargs["reward_log_var"] = log_var.squeeze(-1)

        # Compute Q-value estimates. Get gradients for q-value model only for demonstration feedback type.
        # Otherwise, it is not well defined.
        q_curr = self.Q_value_model(obs)
        q_next = self.Q_value_model(next_obs)
        kwargs["q_curr"] = q_curr
        kwargs["q_next"] = q_next
        
        result = head(reward_samples, **kwargs)
        
        # Handle decoders that return (loss, metrics) or just loss
        nll, metrics = result

        # Regularization
        td_error = td_error_regularizer(**kwargs)

        # Create final output
        output = {"negative_log_likelihood": nll, "kl_divergence": kl_div, "td_error": td_error}
        output.update(metrics)
        return output
