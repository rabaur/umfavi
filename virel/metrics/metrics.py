# Copyright 2019 DeepMind Technologies Limited, 2020 Adam Gleave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experiments with tabular (i.e. finite state) reward models."""

from typing import Callable, Optional

import numpy as np
import scipy.stats

DeshapeFn = Callable[[np.ndarray, float], np.ndarray]


def shape(reward: np.ndarray, potential: np.ndarray, discount: float) -> np.ndarray:
    """Adds a potential-based shaping to a reward.

    Args:
        reward: The three-dimensional reward array.
        potential: The state-only potential function.
        discount: The discount factor.

    Returns:
        reward shaped by potential.
    """
    assert reward.ndim == 3
    assert potential.ndim == 1
    new_pot = discount * potential[np.newaxis, np.newaxis, :]
    old_pot = potential[:, np.newaxis, np.newaxis]

    return reward + new_pot - old_pot


def closest_reward_am(
    source: np.ndarray, target: np.ndarray, n_iter: int = 100, discount: float = 0.99
) -> np.ndarray:
    """Finds the least squared-error reward to target that is equivalent to reward.

    Alternating minimization over `closest_potential` and `closest_affine`.

    Args:
        - source: the source reward.
        - target: the reward to match.
        - n_iter: the number of iterations of expectation-maximization.
        - discount: The discount rate of the MDP.

    Returns:
        A reward that is equivalent to `source` with minimal squared-error to `target`.
    """
    closest_reward = source
    # For undiscounted case, adding constant may cause return to not converge, and so does
    # not form part of the equivalence class.
    shift = discount < 1.0
    for _ in range(n_iter):
        potential = closest_potential(closest_reward, target, discount)
        closest_reward = shape(closest_reward, potential, discount)
        params = base.least_l2_affine(closest_reward.flatten(), target.flatten(), shift=shift)
        closest_reward = closest_reward * params.scale + params.shift
    return closest_reward


def _check_dist(dist: np.ndarray) -> None:
    assert np.allclose(np.sum(dist), 1)
    assert np.all(dist >= 0)


def lp_norm(arr: np.ndarray, p: int, dist: Optional[np.ndarray] = None) -> float:
    r"""Computes the L^{p} norm of arr, weighted by dist.

    Args:
        arr: The array to compute the norm of.
        p: The power to raise elements to.
        dist: A distribution to weight elements of array by.

    Returns:
        The L^{p} norm of arr with respect to the measure dist.
        That is, (\sum_i dist_i * |arr_i|^p)^{1/p}.
    """
    if dist is None:
        # Fast path: use optimized np.linalg.norm
        n = np.product(arr.shape)
        raw_norm = np.linalg.norm(arr.flatten(), ord=p)
        return raw_norm / (n ** (1 / p))

    # Otherwise, weighted; use our implementation (up to 2x slower).
    assert arr.shape == dist.shape
    _check_dist(dist)

    arr = np.abs(arr)
    arr **= p
    arr *= dist
    accum = np.sum(arr)
    accum **= 1 / p
    return accum


def direct_distance(
    rewa: np.ndarray, rewb: np.ndarray, p: int = 2, dist: Optional[np.ndarray] = None
) -> float:
    """L^p norm of the difference between `rewa` and `rewb` w.r.t. distribution `dist`."""
    delta = rewa - rewb
    return lp_norm(delta, p, dist)


def npec_distance(
    src_reward: np.ndarray,
    target_reward: np.ndarray,
    p: int = 2,
    dist: Optional[np.ndarray] = None,
    **kwargs,
) -> float:
    """Computes premetric NPEC distance."""
    closest = closest_reward_am(src_reward, target_reward, **kwargs)
    return direct_distance(closest, target_reward, p, dist)


def _center(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = np.average(x, weights=weights)
    return x - mean


def pearson_distance(
    rewa: np.ndarray, rewb: np.ndarray, dist: Optional[np.ndarray] = None
) -> float:
    """Computes pseudometric derived from the Pearson correlation coefficient.

    It is invariant to positive affine transformations like the Pearson correlation coefficient.

    Args:
        rewa: A reward array.
        rewb: A reward array.
        dist: Optionally, a probability distribution of the same shape as rewa and rewb.

    Returns:
        Computes the Pearson correlation coefficient rho, optionally weighted by dist.
        Returns the square root of 1 minus rho.
    """
    if dist is None:
        dist = np.ones_like(rewa) / np.prod(rewa.shape)
    _check_dist(dist)
    assert rewa.shape == dist.shape
    assert rewa.shape == rewb.shape

    dist = dist.flatten()
    rewa = _center(rewa.flatten(), dist)
    rewb = _center(rewb.flatten(), dist)

    vara = np.average(np.square(rewa), weights=dist)
    varb = np.average(np.square(rewb), weights=dist)
    cov = np.average(rewa * rewb, weights=dist)
    corr = cov / (np.sqrt(vara) * np.sqrt(varb))
    corr = min(corr, 1.0)  # floating point error sometimes rounds above 1.0

    return np.sqrt(0.5 * (1 - corr))


def spearman_distance(rewa: np.ndarray, rewb: np.ndarray) -> float:
    """Computes dissimilarity derived from Spearman correlation coefficient.

    Args:
        rewa: A reward array.
        rewb: A reward array.

    Returns:
        Computes the Spearman correlation coefficient rho. Returns the square root of 1 minus rho.
    """
    # TODO(adam): is this a pseudometric?
    # https://arxiv.org/pdf/1208.3145.pdf claims so but I do not follow their proof,
    # should try to derive it myself if this method looks useful.
    corr, _ = scipy.stats.spearmanr(rewa, rewb)
    return np.sqrt(0.5 * (1 - corr))


def asymmetric_distance(
    source: np.ndarray, target: np.ndarray, dist: Optional[np.ndarray] = None, **kwargs
) -> float:
    """Minimal Pearson distance over rewards equivalent to source. This is a premetric.

    Args:
        source: The three-dimensional source reward array.
        target: The three-dimensional target reward array.
        dist: Optionally, a probability distribution of the same shape as source and target.
        **kwargs: Passed through to `closest_reward_am`.

    Returns:
        The minimal distance to target over rewards equivalent to source.
    """
    source_matched = closest_reward_am(source, target, **kwargs)
    return pearson_distance(source_matched, target, dist)


def symmetric_distance(rewa: np.ndarray, rewb: np.ndarray, use_min: bool, **kwargs) -> float:
    """Symmetric version of `asymmetric_distance`. This is a premetric.

    Args:
        rewa: One three-dimensional reward array.
        rewb: One three-dimensional reward array.
        use_min: If true, take minimum of asymmetric distances; otherwise, take mean.
        **kwargs: Passed through to `asymmetric_distance`.

    Returns:
         The mean of `asymmetric_distance` from `rewa` to `rewb` and `rewb` to `rewa`.
    """
    dista = asymmetric_distance(rewa, rewb, **kwargs)
    distb = asymmetric_distance(rewb, rewa, **kwargs)
    if use_min:
        return min(dista, distb)
    else:
        return 0.5 * (dista + distb)


def singleton_shaping_canonical_reward(rew: np.ndarray, discount: float) -> np.ndarray:
    """
    Compute version of rew with canonicalized shaping.

    Args:
        rew: The three-dimensional reward array to canonicalize.
        discount: The discount rate of the MDP.

    Returns:
        Shaped version of rew. Specifically, the advantage of rew under transition dynamics
        where all states are absorbing, following an optimal policy (picking action greedily
        to maximize reward). This return value is the same for any shaped version of rew.

    Raises:
        ValueError if discount is not less than 1.
    """
    if discount >= 1:
        raise ValueError(f"discount '{discount}' >= 1: only undiscounted MDPs supported.")
    assert discount >= 0
    ns, _na, ns2 = rew.shape
    assert ns == ns2
    instantaneous_reward = rew[np.arange(ns), :, np.arange(ns)]
    greedy_reward = instantaneous_reward.max(1)
    value = 1 / (1 - discount) * greedy_reward
    return shape(rew, value, discount)


def fully_connected_random_canonical_reward(
    rew: np.ndarray,
    discount: float,
    state_dist: Optional[np.ndarray] = None,
    action_dist: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute version of rew with canonicalized shaping.

    Args:
        rew: The three-dimensional reward array to canonicalize.
        discount: The discount rate of the MDP.
        state_dist: Distribution over next states. Uniform if unspecified.
        action_dist: Distribution over actions. Uniform if unspecified.

    Returns:
        Shaped version of rew. Specifically, this corresponds to the advantage under
        transition dynamics where next states are chosen according to state_dist and a policy
        chooses actions according to action_dist. This return value is the same for any
        shaped version of rew.
    """
    assert 0 <= discount <= 1
    ns, _na, ns2 = rew.shape
    assert ns == ns2

    if state_dist is not None:
        _check_dist(state_dist)
    if action_dist is not None:
        _check_dist(action_dist)

    mean_rew_sa = np.average(rew, axis=2, weights=state_dist)
    mean_rew_s = np.average(mean_rew_sa, axis=1, weights=action_dist)
    mean_rew = np.average(mean_rew_s, axis=0, weights=state_dist)
    # In the infinite-horizon discounted case, the value function is:
    # V(s) = mean_rew_s + discount / (1 - discount) * mean_rew
    # So shaping gives:
    # R^{PC} = shape(rew, mean_rew_s, discount)
    #        + (discount - 1) * discount / (1 - discount) * mean_rew
    #        = shape(rew, mean_rew_s, discount) - mean_rew
    # In the finite-horizon undiscounted case, the value function is:
    # V_T(s) = mean_rew_s[s] + T*mean_rew
    # So shaping gives:
    # R^{PC}(s,a,s') = rew[s,a,s'] + V_{T - 1}(s') - V_{T-1}(s)
    #                = rew[s,a,s'] + mean_rew_s[s'] - mean_rew_s[s] - mean_rew
    #                = shape(rew, mean_rew, 1) - 1 * mean_rew
    # So pleasingly the same formula works for the discounted infinite-horizon and undiscounted
    # finite-horizon case.
    return shape(rew, mean_rew_s, discount) - discount * mean_rew


def fully_connected_greedy_canonical_reward(
    rew: np.ndarray,
    discount: float,
    state_dist: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute version of rew with canonicalized shaping.

    Args:
        rew: The three-dimensional reward array to canonicalize.
        discount: The discount rate of the MDP.
        state_dist: Distribution over next states. Uniform if unspecified.

    Returns:
        Shaped version of rew. Specifically, this corresponds to the advantage under
        transition dynamics where next states are chosen from state_dist, with an
        optimal policy. This return value is the same for any shaped version of rew.
    """
    assert 0 <= discount <= 1
    ns, _na, ns2 = rew.shape
    assert ns == ns2

    if state_dist is None:
        state_dist = np.ones(ns) / ns
    _check_dist(state_dist)

    mean_rew_sa = np.average(rew, axis=2, weights=state_dist)
    optimal_rew_s = np.max(mean_rew_sa, axis=1)
    mean_rew = np.average(optimal_rew_s, axis=0, weights=state_dist)
    # See `all_uniform_shaping_canonical_reward` for a discussion of how this expression
    # is derived from shaping (details differ but the overall argument is similar).
    return shape(rew, optimal_rew_s, discount) - discount * mean_rew


def canonical_scale_normalizer(
    rew: np.ndarray, p: int = 1, dist: Optional[np.ndarray] = None, eps: float = 1e-10
) -> float:
    """
    Compute coefficient by which to scale `rew` for it to have canonical scale.

    Coefficient is rounded down to `0` if computed scale is less than `eps`.

    Args:
        rew: The three-dimensional reward array to compute the normalizer for.
        p: The power to raise elements to.
        dist: The measure for the L^{p} norm.
        eps: Threshold to treat reward as zero (needed due to floating point error).

    Returns:
        Scaling coefficient by which to multiply `rew` to have unit norm.
    """
    scale = lp_norm(rew, p, dist)
    return 0 if abs(scale) < eps else 1 / scale


def canonical_reward(
    rew: np.ndarray,
    discount: float,
    deshape_fn: DeshapeFn,
    p: int = 1,
    dist: Optional[np.ndarray] = None,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute canonical version of rew, invariant to shaping and positive scaling.

    Args:
        rew: The three-dimensional reward array to canonicalize.
        discount: The discount rate of the MDP.
        deshape_fn: The function to canonicalize the shaping component of the reward.
        p: The power to raise elements to.
        dist: The measure for the L^{p} norm.
        eps: Threshold to treat reward as zero (needed due to floating point error).

    Returns:
        Canonical version of rew. Shaping is removed in accordance with `deshape_fn`.
        This is then rescaled to have unit norm.
    """
    res = deshape_fn(rew, discount)
    res *= canonical_scale_normalizer(res, p, dist, eps)
    return res


def canonical_reward_distance(
    rewa: np.ndarray,
    rewb: np.ndarray,
    discount: float,
    deshape_fn: DeshapeFn,
    p: int = 1,
    dist: Optional[np.ndarray] = None,
) -> float:
    """
    Computes direct distance between canonicalized versions of rewa and rewb.

    Args:
        rewa: A three-dimensional reward array.
        rewb: A three-dimensional reward array.
        discount: The discount rate of the MDP.
        deshape_fn: The function to canonicalize the shaping component of the reward.
        p: The power to raise elements to.
        dist: The measure for the L^{p} norm.

    Returns:
        The L^{p} norm of the difference between the canonicalized versions of `rewa` and `rewb`.
    """
    rewa_canon = canonical_reward(rewa, discount, deshape_fn, p, dist)
    rewb_canon = canonical_reward(rewb, discount, deshape_fn, p, dist)
    return 0.5 * direct_distance(rewa_canon, rewb_canon, p, dist)


def deshape_pearson_distance(
    rewa: np.ndarray,
    rewb: np.ndarray,
    discount: float,
    deshape_fn: DeshapeFn,
    dist: Optional[np.ndarray] = None,
) -> float:
    """
    Computes Pearson distance between deshaped versions of rewa and rewb.

    Args:
        rewa: A three-dimensional reward array.
        rewb: A three-dimensional reward array.
        discount: The discount rate of the MDP.
        deshape_fn: The function to canonicalize the shaping component of the reward.
        dist: The measure for the Pearson distance.

    Returns:
        The Pearson distance between the deshaped versions of `rewa` and `rewb`.
    """
    rewa = deshape_fn(rewa, discount)
    rewb = deshape_fn(rewb, discount)
    print(rewa.max())
    print(rewb.max())
    pdist = pearson_distance(rewa, rewb, dist)
    return pdist

def epic_distance(
    rewa: np.ndarray,
    rewb: np.ndarray,
    discount: float,
) -> float:
    """
    Computes EPIC distance between deshaped versions of rewa and rewb.
    """
    return deshape_pearson_distance(rewa, rewb, discount, singleton_shaping_canonical_reward)