from typing import Optional
import torch
from torch import nn

class MLPFeatureModule(nn.Module):
    """Module for encoding features using a MLP."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: list[int],
        reward_domain: str = 's',
        activate_last_layer: bool = True,
        learn_embedding: Optional[bool] = False,
        state_embedding_size: Optional[int] = None,
        action_embedding_size: Optional[int] = None,
        n_states: Optional[int] = None,
        n_actions: Optional[int] = None
    ):
        """
        Args:
            state_dim: The dimension of the state features.
                Only used if `learn_embedding` is false.
            learn_embedding: Set to true to learn embedding.
        """
        super().__init__()
        
        self.learn_embedding = learn_embedding
        self.reward_domain = reward_domain

        if learn_embedding:
            assert state_embedding_size, "`state_embedding_size` cannot be None if `learn_embedding` is True"
            assert action_embedding_size, "`action_embedding_size` cannot be None if `learn_embedding` is True"
            assert n_states, "`n_states` cannot be None if `learn_embedding` is True"
            assert n_actions, "`n_actions` cannot be None if `learn_embedding` is True"
            self.state_embedding = nn.Embedding(n_states, state_embedding_size)
            self.action_embedding = nn.Embedding(n_actions, action_embedding_size)
        else:
            self.state_embedding = nn.Identity()
            self.action_embedding = nn.Identity()
        
        # If we learn the state embedding, the state feature dimensional is the state_embedding
        # Analogously for action embedding
        if learn_embedding:
            state_dim = state_embedding_size
            action_dim = action_embedding_size

        if reward_domain == 's':
            input_dim = state_dim
        elif reward_domain == 'sa':
            input_dim = state_dim + action_dim
        elif reward_domain == 'sas':
            input_dim = state_dim + action_dim + state_dim
        else:
            raise Exception(f"Unsupported reward domain '{reward_domain}'")
        hidden_sizes = [input_dim] + hidden_sizes
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if i < len(hidden_sizes) - 2 or activate_last_layer:
                layers.append(nn.SiLU())
        self.mlp = nn.Sequential(*layers)
        self.out_dim = hidden_sizes[-1]
    
    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        next_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        state_feats = self.state_embedding(states)
        
        if self.reward_domain == 's':
            feats = state_feats
        elif self.reward_domain == 'sa':
            action_feats = self.action_embedding(actions)
            feats = torch.cat([state_feats, action_feats], dim=-1)
        elif self.reward_domain == 'sas':
            action_feats = self.action_embedding(actions)
            next_state_feats = self.state_embedding(next_states)
            feats = torch.cat([state_feats, action_feats, next_state_feats], dim=-1)
        else:
            raise Exception(f"Unsupported reward domain '{self.reward_domain}'")
        
        return self.mlp(feats)