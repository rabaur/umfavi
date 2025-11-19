import torch
from torch import nn
from typing import Any
from umfavi.loglikelihoods.base import BaseLogLikelihood
from umfavi.priors import kl_divergence_std_normal
from umfavi.encoder.reward_encoder import BaseRewardEncoder
from umfavi.regularizer.td_error import td_error_regularizer

class MultiFeedbackTypeModel(nn.Module):

    def __init__(self, encoder: BaseRewardEncoder, Q_value_model: nn.Module, decoders: dict[FeedbackType, BaseLogLikelihood]):
        super().__init__()
        self.encoder = encoder
        self.Q_value_model = Q_value_model
        self.decoders = nn.ModuleDict(decoders)
    
    def forward(self, **kwargs) -> Any:

        # Encode
        state_feats = kwargs["state_features"]
        action_feats = kwargs["action_features"]
        next_state_feats = kwargs["next_state_features"]
        mean, log_var = self.encoder(state_feats, action_feats, next_state_feats)
        reward_samples = self.encoder.sample(mean, log_var)
        kl_div = kl_divergence_std_normal(mean, log_var)

        # Route to appropriate head with all kwargs
        head = self.decoders[kwargs["feedback_type"][0]]  # we can assume that all feedback types are the same per batch
        
        # Add reward mean and log_var to kwargs for decoders that need them
        kwargs["reward_mean"] = mean.squeeze(-1)
        kwargs["reward_log_var"] = log_var.squeeze(-1)

        # Compute Q-value estimates. Get gradients for q-value model only for demonstration feedback type.
        # Otherwise, it is not well defined.
        q_values = self.Q_value_model(state_feats)
        kwargs["q_values"] = q_values
        
        result = head(reward_samples.squeeze(-1), **kwargs)
        
        # Handle decoders that return (loss, metrics) or just loss
        nll, metrics = result

        # Regularization
        if kwargs["feedback_type"][0] != "demonstration":
            regularization = td_error_regularizer(q_values, kwargs["actions"], kwargs["reward_mean"], kwargs["reward_log_var"], kwargs["gamma"])
        else:
            regularization = td_error_regularizer(q_values, kwargs["actions"], kwargs["reward_mean"], kwargs["reward_log_var"], kwargs["gamma"])

        # Create final output
        output = {"negative_log_likelihood": nll, "kl_divergence": kl_div, "td_error": regularization}
        output.update(metrics)
        return output
