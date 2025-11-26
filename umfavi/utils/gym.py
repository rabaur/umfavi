import gymnasium as gym
from typing import Any, Callable, Optional
import numpy as np
from umfavi.envs.grid_env.env import GridEnv
from umfavi.types import TrajectoryType, TrajKeys

INVALID_FLOAT = np.float32(np.nan)
INVALID_INT = -1


def _create_nan_like(value):
    """Create a NaN version of the given value, preserving structure."""
    if isinstance(value, dict):
        return {k: _create_nan_like(v) for k, v in value.items()}
    elif isinstance(value, (np.ndarray, list, tuple)):
        arr = np.asarray(value, dtype=np.float32)
        return np.full_like(arr, INVALID_FLOAT, dtype=np.float32)
    elif isinstance(value, (float, np.floating)):
        return INVALID_FLOAT
    elif isinstance(value, (int, np.integer)):
        return INVALID_INT
    else:
        # For other types, return np.nan as fallback
        return INVALID_FLOAT


def rollout(
    env: gym.Env,
    policy: Callable,
    num_steps: Optional[int] = None,
    pad: bool = True,
    seed: int = None
) -> TrajectoryType:
    """
    Rollout a policy in an environment for n_steps.
    
    Args:
        env: The environment
        policy: Policy function that maps observations to actions
        num_steps: Number of steps to rollout. If None, rollout until receives "done" from environment.
        pad: If True, pad trajectory to n_steps with NaN values if episode ends early
        seed: Optional seed for environment reset
    
    Returns:
        List of (obs, action, reward, next_obs, done, info) tuples
    """
    ep = []
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()
    done = False
    step = 0
    
    while not done:
        if step and step > num_steps:
            break
        action = policy(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep.append((obs, action, reward, next_obs, done, info))
        obs = next_obs
        step += 1
    
    # Pad trajectory if it ended early
    if pad and step < num_steps:
        nan_obs = _create_nan_like(obs)
        nan_action = _create_nan_like(action)
        nan_reward = INVALID_FLOAT
        nan_info = {}
        
        for _ in range(num_steps - step):
            ep.append((nan_obs, nan_action, nan_reward, nan_obs, True, nan_info))
    
    return ep


def unpack_trajectory(trajectory: TrajectoryType) -> dict[str, list[Any]]:
    """
    Unpack a trajectory into a dictionary of lists. Info dicts of the original trajectory are flattened.
    """
    traj_dict = {
        TrajKeys.OBS: [obs for obs, _, _, _, _, _ in trajectory],
        TrajKeys.ACTS: [act for _, act, _, _, _, _ in trajectory],
        TrajKeys.REWS: [r for _, _, r, _, _, _ in trajectory],
        TrajKeys.NEXT_OBS: [next_obs for _, _, _, next_obs, _, _ in trajectory],
        TrajKeys.DONES: [done for _, _, _, _, done, _ in trajectory]
    }
    infos = [info for _, _, _, _, _, info in trajectory]
    # Only unpack info keys if there are non-empty infos
    if infos and infos[0]:
        traj_dict.update({k: [info.get(k, INVALID_FLOAT) for info in infos] for k in infos[0].keys()})
    
    # Standardize: ensure all data has feature dimension for uniform handling
    # Convert scalars (actions, rewards, dones) to have explicit feature dimension
    for key in traj_dict.keys():
        arr = np.array(traj_dict[key])
        # If 1D (scalar per timestep), add feature dimension: (T,) -> (T, 1)
        if arr.ndim == 1:
            traj_dict[key] = arr[:, None]
        else:
            traj_dict[key] = arr
    
    return traj_dict


def get_undiscounted_return(trajectory: TrajectoryType):
    rewards = np.array([r for _, _, r, _, _, _ in trajectory])
    return np.nansum(rewards)


def get_discounted_return(trajectory: TrajectoryType, gamma: float):
    rewards = np.array([r for _, _, r, _, _, _ in trajectory])
    T = len(rewards)
    gammas = gamma ** np.arange(T)
    return np.nansum(rewards * gammas)


def is_registered_gym_env(env_name: str) -> bool:
    """
    Check if a Gym environment is registered.
    """
    return env_name in gym.envs.registry


def get_obs_dim(env: gym.Env, observation_transform: Callable = None) -> int:
    """
    Get the dimensionality of the observation representation by sampling a random observation and applying the observation-transform.
    """
    rand_obs = env.observation_space.sample()
    if observation_transform is not None:
        rand_obs = observation_transform(rand_obs)
    if isinstance(rand_obs, np.ndarray):
        return rand_obs.shape[0]
    elif isinstance(rand_obs, int) or isinstance(rand_obs, float):
        return 1
    else:
        raise ValueError(f"Invalid observation type: {type(rand_obs)}")


def get_act_dim(env: gym.Env, action_transform: Callable = None) -> int:
    """
    Get the dimensionality of the action representation by sampling a random action and applying the action-transform.
    """
    rand_action = env.action_space.sample()
    if action_transform is not None:
        rand_action = action_transform(rand_action)
    if isinstance(rand_action, np.ndarray):
        return rand_action.shape[0]
    elif isinstance(rand_action, int) or isinstance(rand_action, float):
        return 1
    else:
        raise ValueError(f"Invalid action type: {type(rand_action)}")