import numpy as np
import gymnasium as gym
from numpy.typing import NDArray
from umfavi.learned_reward_wrapper import LearnedRewardWrapper
from umfavi.utils.tabular import q_opt
from umfavi.utils.policies import ExpertPolicy, create_expert_policy
from umfavi.utils.gym import rollout, get_discounted_return


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
    R_true: NDArray,
    R_est: NDArray,
    P: NDArray,
    gamma: float,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> float:
    """
    Computes expected regret over states.
    """
    
    # Compute optimal Q-values for the true reward
    Q_true_opt = q_opt(P, R_true, gamma, max_iter=max_iter, tol=tol)
    
    # Compute optimal Q-values for the estimated reward
    Q_est_opt = q_opt(P, R_est, gamma, max_iter=max_iter, tol=tol)
    
    # Extract optimal policies (greedy w.r.t. Q-values)
    V_true_star = np.max(Q_true_opt, axis=1)           # (S,)
    pi_est = np.argmax(Q_est_opt, axis=1)              # (S,)
    V_true_pi = value_under_policy(P, R_true, gamma, pi_est)

    regret = float(np.mean(V_true_star - V_true_pi))
    return regret


def evaluate_regret_non_tabular(
    true_expert_policy: ExpertPolicy,
    base_env: gym.Env,
    wrapped_env: LearnedRewardWrapper,
    gamma: float,
    num_samples: int = 100,
    max_num_steps: int = 100,
):
    est_expert_policy = create_expert_policy(wrapped_env, rationality=float("inf"), gamma=gamma, force_train=True)
    regret = 0
    for _ in range(num_samples):
        traj_expert = rollout(base_env, true_expert_policy, n_steps=max_num_steps)
        traj_est = rollout(base_env, est_expert_policy, n_steps=max_num_steps)
        ret_expert = get_discounted_return(traj_expert, gamma)
        ret_est = get_discounted_return(traj_est, gamma)
        regret += ret_expert - ret_est
    return regret / num_samples


    

