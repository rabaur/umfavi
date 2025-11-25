import numpy as np
import torch
import gymnasium as gym
from typing import Callable
from umfavi.learned_reward_wrapper import LearnedRewardWrapper
from umfavi.utils.tabular import q_opt
from umfavi.utils.policies import (
    ExpertPolicy, 
    create_expert_policy,
    load_or_train_dqn,
    DQNQValueModel
)
from umfavi.envs.env_types import TabularEnv
from umfavi.encoder.reward_encoder import RewardEncoder
from umfavi.utils.feature_transforms import get_feature_combinations
from umfavi.utils.gym import rollout, get_discounted_return
from umfavi.utils.torch import to_numpy


def value_under_policy(P, R_true, gamma, pi):
    S, A, Sp = P.shape
    assert Sp == S
    # r_pi[s] = E[R(s, pi[s], S')]
    if R_true.ndim == 2:      # R(s,a)
        r_pi = R_true[np.arange(S), pi]
    else:                     # R(s,a,s')
        r_sa = R_true[np.arange(S), pi, :]          # (S, S)
        r_pi = np.sum(P[np.arange(S), pi, :] * r_sa, axis=1)

    # P_pi[s, s'] = P(s'|s, pi[s])
    P_pi = P[np.arange(S), pi, :]                   # (S, S)

    # Solve (I - gamma P_pi) V = r_pi
    I = np.eye(S)
    V = np.linalg.solve(I - gamma * P_pi, r_pi)
    return V  # shape (S,)


def evaluate_regret_tabular(
    env: TabularEnv,
    encoder: RewardEncoder,
    all_obs_features: torch.Tensor,
    all_act_features: torch.Tensor,
    gamma: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
    num_samples: int = 1000
) -> float:
    """
    Computes expected regret over states.
    """

    R_true = env.get_reward_matrix()
    P = env.get_transition_matrix()

    # Construct all state-action-next_state features to compute the estimated reward matrix
    num_states = P.shape[0]  # == num_next_states
    num_actions = P.shape[1]

    # Compute optimal Q-values for the true reward
    Q_true_opt = q_opt(P, R_true, gamma, max_iter=max_iter, tol=tol)
    V_true_star = np.max(Q_true_opt, axis=1)           # (S,)
    
    # Optimize batched inference based on reward domain
    reward_domain = encoder.features.reward_domain
    
    expanded_s_feats, expanded_a_feats, expanded_sp_feats = \
        get_feature_combinations(reward_domain, all_obs_features, all_act_features)
    

    with torch.no_grad():
        R_est_mean, _ = encoder.forward(expanded_s_feats, expanded_a_feats, expanded_sp_feats)
    
    R_est_mean = to_numpy(R_est_mean).squeeze()

    if reward_domain == 's':
        R_est = np.broadcast_to(R_est_mean[:, None, None], (num_states, num_actions, num_states))
    elif reward_domain == 'sa':
        R_est = np.reshape(R_est_mean, (num_states, num_actions))
        R_est = np.broadcast_to(R_est[:, :, None], (num_states, num_actions, num_states))
    else:
        R_est = np.reshape(R_est_mean, (num_states, num_actions, num_states))
    
    # Compute optimal Q-values for the estimated reward
    Q_est_opt = q_opt(P, R_est, gamma, max_iter=max_iter, tol=tol)
    
    pi_est = np.argmax(Q_est_opt, axis=1)              # (S,)
    V_est_pi = value_under_policy(P, R_true, gamma, pi_est)

    regret = float(np.mean(V_true_star - V_est_pi))
    return regret


def evaluate_regret_non_tabular(
    true_expert_policy: ExpertPolicy,
    base_env: gym.Env,
    wrapped_env: LearnedRewardWrapper,
    gamma: float,
    num_samples: int = 1000,
    max_num_steps: int = 100,
) -> tuple[float, float]:
    """
    MC estimate of the expected regret and the mean return of the estimated expert policy.
    """
    # Train a new DQN model on the wrapped environment with learned reward
    dqn_model = load_or_train_dqn(wrapped_env, gamma=gamma, force_train=True, training_timesteps=10000)
    q_model = DQNQValueModel(dqn_model)
    est_expert_policy = create_expert_policy(wrapped_env, rationality=float("inf"), q_model=q_model)
    
    regret = 0
    mean_rew = 0
    for i in range(num_samples):
        # Roll out both policies from the same initial state for fair comparison
        seed = i  # Use iteration index as seed for reproducibility
        
        # Rollout true expert policy
        traj_expert = rollout(base_env, true_expert_policy, n_steps=max_num_steps, seed=seed)
        ret_expert = get_discounted_return(traj_expert, gamma)
        
        # Rollout estimated expert policy from the same initial state
        traj_est = rollout(base_env, est_expert_policy, n_steps=max_num_steps, seed=seed)
        ret_est = get_discounted_return(traj_est, gamma)
        mean_rew += ret_est
        regret += ret_expert - ret_est
    
    return regret / num_samples, mean_rew / num_samples

    

