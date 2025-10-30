from torch import nn
from typing import Any
from umfavi.log_likelihoods.base_log_likelihood import BaseLogLikelihood
from umfavi.priors import kl_divergence_std_normal

class MultiFeedbackTypeModel(nn.Module):

    def __init__(self, encoder: nn.Module, decoders: dict[str, BaseLogLikelihood]):
        super().__init__()
        self.encoder = encoder
        self.decoders = nn.ModuleDict(decoders)
    
    def forward(self, **kwargs) -> Any:

        # Encode
        obs = kwargs["obs"]
        acts = kwargs["acts"]
        mean, log_var = self.encoder(obs, acts)
        r_samples = self.encoder.sample(mean, log_var)
        kl_div = kl_divergence_std_normal(mean, log_var)

        # Route to appropriate head with all kwargs
        head = self.decoders[kwargs["feedback_type"][0]]  # we can assume that all feedback types are the same per batch
        
        # Add reward mean and log_var to kwargs for decoders that need them
        kwargs["reward_mean"] = mean
        kwargs["reward_log_var"] = log_var
        
        nll = head(r_samples, **kwargs)

        return {"negative_log_likelihood": nll, "kl_divergence": kl_div}
