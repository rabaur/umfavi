from torch import nn
from typing import Any
from umfavi.loglikelihoods.base import BaseLogLikelihood
from umfavi.priors import kl_divergence_std_normal
from umfavi.encoder.reward_encoder import BaseRewardEncoder

class MultiFeedbackTypeModel(nn.Module):

    def __init__(self, encoder: BaseRewardEncoder, decoders: dict[str, BaseLogLikelihood]):
        super().__init__()
        self.encoder = encoder
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
        
        result = head(reward_samples, **kwargs)
        
        # Handle decoders that return (loss, metrics) or just loss
        if isinstance(result, tuple):
            nll, metrics = result
        else:
            nll = result
            metrics = {}

        output = {"negative_log_likelihood": nll, "kl_divergence": kl_div}
        output.update(metrics)
        return output
