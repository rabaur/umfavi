import gymnasium as gym
from typing import Any, Callable, Optional
import numpy as np
from numpy.typing import NDArray
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
) -> dict[TrajKeys, NDArray[Any]]:
    """
    Rollout a policy in an environment for n_steps.
    
    Args:
        env: The environment
        policy: Policy function that maps observations to actions
        num_steps: Number of steps to rollout. If None, rollout until receives "done" from environment.
        pad: If True, pad trajectory to n_steps with NaN values if episode ends early
        seed: Optional seed for environment reset
    
    Returns:
        Dictionary mapping TrajKeys to numpy arrays of trajectory data.
    """
    traj = {k: [] for k in TrajKeys}
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()
    terminated = False
    truncated = False
    step = 0
    action = None
    info = {}
    
    while not terminated and not truncated:
        if num_steps and step >= num_steps:
            break
        action = policy(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        # if obs and actions are scalars, convert to 1D arrays
        obs = np.atleast_1d(obs)
        action = np.atleast_1d(action)
        next_obs = np.atleast_1d(next_obs)
        traj[TrajKeys.OBS].append(obs)
        traj[TrajKeys.ACTS].append(action)
        traj[TrajKeys.REWS].append(reward)
        traj[TrajKeys.NEXT_OBS].append(next_obs)
        traj[TrajKeys.TERMINATED].append(terminated)
        traj[TrajKeys.TRUNCATED].append(truncated)
        traj[TrajKeys.INVALID].append(False)
        for k, v in info.items():
            traj[k].append(v)
        obs = next_obs
        step += 1
    
    # Pad trajectory if it ended early
    if num_steps and pad and step < num_steps:
        nan_obs = _create_nan_like(obs)
        nan_obs = np.atleast_1d(nan_obs)
        # Use action from last step, or sample from action space if no steps taken
        nan_action = _create_nan_like(action) if action is not None else _create_nan_like(env.action_space.sample())
        nan_action = np.atleast_1d(nan_action)
        nan_reward = INVALID_FLOAT
        nan_info = {k: INVALID_FLOAT for k in info.keys()}
        
        for _ in range(num_steps - step):
            traj[TrajKeys.OBS].append(nan_obs)
            traj[TrajKeys.ACTS].append(nan_action)
            traj[TrajKeys.REWS].append(nan_reward)
            traj[TrajKeys.NEXT_OBS].append(nan_obs)
            traj[TrajKeys.TERMINATED].append(True)
            traj[TrajKeys.TRUNCATED].append(True)
            traj[TrajKeys.INVALID].append(True)
            for k, v in nan_info.items():
                traj[k].append(v)
    
    # Convert to numpy arrays
    for k in traj.keys():
        traj[k] = np.array(traj[k])

    return traj


def get_undiscounted_return(trajectory: dict[TrajKeys, NDArray]):
    return np.nansum(trajectory[TrajKeys.REWS])

def get_discounted_return(trajectory: dict[TrajKeys, NDArray], gamma: float):
    return np.nansum(trajectory[TrajKeys.REWS] * gamma ** np.arange(len(trajectory[TrajKeys.REWS])))

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
    elif isinstance(rand_obs, int) or isinstance(rand_obs, float) or isinstance(rand_obs, np.integer):
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

def get_env_name(env: gym.Env):
    return env.unwrapped.spec.id