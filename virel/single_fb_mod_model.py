from typing import Any
import torch
from torch import nn
from virel.priors import kl_divergence_normal
from virel.log_likelihoods.base_log_likelihood import BaseLogLikelihood


class SingleFeedbackTypeModel(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: BaseLogLikelihood):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, obs: torch.Tensor, acts: torch.Tensor, targets: Any) -> Any:
        mean, log_var = self.encoder(obs, acts)
        r_samples = self.encoder.sample(mean, log_var)
        preds = self.decoder(r_samples)
        ll = self.decoder.nll(preds, targets)
        kl_div = kl_divergence_normal(mean, log_var)
        loss_dict = {
            "negative_log_likelihood": ll,
            "kl_divergence": kl_div
        }
        return r_samples, preds, loss_dict