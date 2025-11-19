import numpy as np
from typing import Optional
from numpy.typing import NDArray
from umfavi.utils.reward import Rsa_to_Rsas

def canonically_shaped_reward(R_sas: NDArray, gamma: float, d_S: Optional[NDArray] = None, d_A: Optional[NDArray] = None) -> NDArray:
    """
    For a reward function R : S x A x S -> Reals the canonically shaped reward is defined as:
    C_{dist_S, dist_A}(R)(s,a,s') = R(s,a,s') + E[γR(s',A,S') - R(s,A,S') - γE[R(S,A,S')] 
                                  = R(s,a,s') + γE[R(s',A,S')] - E[R(s,A,S')] - γE[R(S,A,S')]

    Args:
        R_sas: A (|S|, |A|, |S|) array representing the tabular reward.
        gamma: The discount factor
        d_S: A (|S|,) vector representing a probability distribution over states. Uses uniform distribution if not passed.
        d_A: A (|A|,) vector representing a probability distribution over actions. Uses uniform distribution if not passed.

    Returns:
        The canonicalized reward.
    """
    assert len(R_sas.shape) == 3, f"Expect `R_sas` to be 3-dim, but was {R_sas.shape=}"
    S, A, Sp = R_sas.shape
    assert S == Sp, "Last dim must equal number of states"
    if not d_S:
        d_S = np.ones(S) / S
    if not d_A:
        d_A = np.ones(A) / A

    # M[s] = E_{A~D_a, S'~D_s}[ R(s, A, S') ]  shape (S,)
    M = np.einsum('a,p,sap->s', d_A, d_S, R_sas)

    # G = gamma * E_{S~D_s}[ M[S] ]  scalar (mean reward)
    G = np.dot(d_S, M)

    # C[s,a,s'] = R[s,a,s'] + gamma*M[s'] - M[s] - G
    C_sas = R_sas + gamma * M[None, None, :] - M[:, None, None] - gamma * G
    return C_sas


def pearson_correlation(X: NDArray, Y: NDArray) -> float:
    """
    Computes the Pearson correlation coefficient between two reward functions.
    """
    return np.corrcoef(X.flatten(), Y.flatten())[0, 1]


def pearson_distance(X: NDArray, Y: NDArray) -> float:
    """
    Computes the Pearson distance between two reward functions.
    """
    return np.sqrt(0.5 * (1 - pearson_correlation(X, Y)))


def epic_distance(R1: NDArray, R2: NDArray, gamma: float) -> float:
    """
    Computes the EPIC distance between two reward functions.
    """
    return pearson_distance(canonically_shaped_reward(R1, gamma), canonically_shaped_reward(R2, gamma))

def evaluate_epic_distance(
    R_true: NDArray,
    R_est: NDArray,
    gamma: float
) -> float:
    """
    Computes the epic distance between the predicted mean of the reward distribution
    and the ground-truth reward.

    Args:
        R_true: A (n_S, n_A) array containing the ground truth rewards.
        R_est: The feedback model to evaluate.
        state_feats: A (n_S, n_state_feats) array of state features.
        action_feats: A (n_A, n_act_feats) array of action features.
    """
    R_true_sas = Rsa_to_Rsas(R_true)
    R_est_sas = Rsa_to_Rsas(R_est)
    dist = epic_distance(R_true_sas, R_est_sas, gamma)
    return dist


