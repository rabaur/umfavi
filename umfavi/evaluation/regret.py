import numpy as np
import torch
import gymnasium as gym
from typing import Callable
from concurrent.futures import ThreadPoolExecutor
from umfavi.learned_reward_wrapper import LearnedRewardWrapper
from umfavi.utils.tabular import q_opt
from umfavi.utils.policies import (
    ExpertPolicy, 
    create_expert_policy,
    DQNQValueModel
)
from umfavi.envs.env_types import TabularEnv
from umfavi.encoder.reward_encoder import RewardEncoder
from umfavi.utils.feature_transforms import get_feature_combinations
from umfavi.utils.gym import rollout, get_discounted_return, get_undiscounted_return
from umfavi.utils.torch import to_numpy
from umfavi.utils.sb3 import train_dqn
from tqdm import tqdm
from umfavi.true_reward_callback import TrueRewardCallback


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


def regret_tabular(
    env: TabularEnv,
    encoder: RewardEncoder,
    all_obs_features: torch.Tensor,
    all_act_features: torch.Tensor,
    gamma: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
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


def compute_single_regret_sample(
    seed: int,
    env_id: str,
    true_expert_policy: ExpertPolicy,
    est_optimal_policy: ExpertPolicy,
    gamma: float,
    max_num_steps: int,
) -> tuple[float, float]:
    """Compute regret for a single sample (used for parallel execution)."""
    # Create fresh env instance per thread to avoid race conditions
    env = gym.make(env_id)
    
    traj_expert = rollout(env, true_expert_policy, num_steps=max_num_steps, seed=seed)
    ret_expert = get_discounted_return(traj_expert, gamma)
    
    traj_est = rollout(env, est_optimal_policy, num_steps=max_num_steps, seed=seed)
    ret_est = get_discounted_return(traj_est, gamma)
    cum_rew = get_undiscounted_return(traj_est)
    
    env.close()
    return ret_expert - ret_est, cum_rew


def regret_non_tabular(
    true_expert_policy: ExpertPolicy,
    base_env: gym.Env,
    wrapped_env: LearnedRewardWrapper,
    gamma: float,
    num_samples: int = 1000,
    max_num_steps: int = 100
) -> tuple[float, float, ExpertPolicy]:
    """
    MC estimate of the expected regret and the mean return of the estimated expert policy.
    
    Returns:
        tuple: (regret, mean_reward, estimated_expert_policy)
    """
    # Train a new DQN model on the wrapped environment with learned reward
    print(f"Training DQN model on estimated reward function...")
    dqn_model = train_dqn(wrapped_env, base_env.unwrapped.spec.id)
    q_model = DQNQValueModel(dqn_model)
    est_optimal_policy = create_expert_policy(q_model, rationality=float("inf"))
    
    env_id = base_env.unwrapped.spec.id
    
    regrets = np.empty(num_samples)
    rewards = np.empty(num_samples)
    for i in tqdm(range(num_samples), desc="Computing regret"):
        regrets[i], rewards[i] = compute_single_regret_sample(
            seed=i,
            env_id=env_id,
            true_expert_policy=true_expert_policy,
            est_optimal_policy=est_optimal_policy,
            gamma=gamma,
            max_num_steps=max_num_steps,
        )
    return np.mean(regrets), np.mean(rewards), est_optimal_policy

# Compute expected regret
def compute_regret(
    env: TabularEnv | gym.Env,
    reward_encoder: RewardEncoder,
    optimal_policy: ExpertPolicy,
    gamma: float,
    num_samples: int,
    max_num_steps: int,
    act_transform: Callable,
    obs_transform: Callable,
):
    regret = None
    mean_rew = None
    est_optimal_policy = None
    if isinstance(env, TabularEnv):
        regret = regret_tabular(
            env,
            reward_encoder,
            gamma=gamma
        )
    else:
        wrapped_env = LearnedRewardWrapper(env, reward_encoder, act_transform, obs_transform)
        regret, mean_rew, est_optimal_policy = regret_non_tabular(
            optimal_policy,
            env,
            wrapped_env,
            gamma=gamma,
            num_samples=num_samples,
            max_num_steps=max_num_steps
        )
    return regret, mean_rew, est_optimal_policy