import torch
from umfavi.utils.math import log_var_to_std
from umfavi.types import SampleKey

def td_error_regularizer(**kwargs) -> torch.Tensor:
    """
    Computes TD error regularization for transition-level data.
    
    For each transition (s, a, s', a'), computes:
        TD_error = Q(s, a) - γ * Q(s', a')
    
    And enforces that the learned reward R(s,a) should match this TD error.
    
    Args:
        q_curr: Q-values at current state, shape (batch_size, n_actions)
        q_next: Q-values at next state, shape (batch_size, n_actions)
        acts_curr: Current actions, shape (batch_size, 1)
        acts_next: Next actions, shape (batch_size, 1)
        reward_mean: Mean of reward distribution, shape (batch_size,) or (batch_size, 1)
        reward_log_var: Log variance of reward distribution, shape (batch_size,) or (batch_size, 1)
        gamma: Discount factor (scalar tensor or float)
        dones: Boolean mask shape (batch_size, 1) indicating terminal states (Q_next should be 0)
    """

    # Unpack variables
    acts_curr = kwargs[SampleKey.ACTS].long()
    acts_next = kwargs[SampleKey.NEXT_ACTS].long()
    gamma = kwargs[SampleKey.GAMMA][0].item() if torch.is_tensor(kwargs[SampleKey.GAMMA]) else kwargs[SampleKey.GAMMA]
    dones = kwargs[SampleKey.DONES].squeeze(-1)  # (batch_size,)
    q_curr = kwargs["q_curr"]  # (batch_size, n_actions)
    q_next = kwargs["q_next"]  # (batch_size, n_actions)

    reward_mean = kwargs["reward_mean"]  # (batch_size,) or (batch_size, 1)
    reward_std = log_var_to_std(kwargs["reward_log_var"])  # (batch_size,) or (batch_size, 1)

    # Ensure reward_mean and reward_std are 1D
    if reward_mean.dim() > 1:
        reward_mean = reward_mean.squeeze(-1)
    if reward_std.dim() > 1:
        reward_std = reward_std.squeeze(-1)

    # Select Q(s_t, a_t) for current state-action pairs
    q_curr_a = torch.gather(q_curr, dim=-1, index=acts_curr).squeeze(-1)  # (batch_size,)
    
    # Select Q(s_{t+1}, a_{t+1}) for next state-action pairs
    q_next_a = torch.gather(q_next, dim=-1, index=acts_next).squeeze(-1)  # (batch_size,)

    # if acts_next is -1, set q_next_a to 0
    q_next_a = q_next_a * (1.0 - (acts_next == -1).float())

    # Compute TD-error: R(s,a) = Q(s,a) - γ * Q(s',a')
    td_error = q_curr_a - gamma * q_next_a  # (batch_size,)
    
    # Compute negative log-likelihood: the learned reward should explain the TD error
    normal_dist = torch.distributions.Normal(reward_mean, reward_std)
    td_error_nll = -normal_dist.log_prob(td_error).mean()
    
    return td_error_nll