import gymnasium as gym
from gymnasium import spaces
from typing import Any, Callable
from umfavi.envs.grid_env.env import GridEnv

ObsType = Any
ActType = Any
TrajectoryType = list[tuple[ObsType, ActType, float, ObsType, bool, dict[str, Any]]]

def rollout(env: gym.Env, policy: Callable, n_steps: int) -> TrajectoryType:
    ep = []
    obs, _ = env.reset()
    done = False
    ep = []
    step = 0
    while not done and step < n_steps:
        action = policy(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep.append((obs, action, reward, next_obs, done, info))
        obs = next_obs
        step += 1
    return ep

def deconstruct_trajectory(trajectory: TrajectoryType) -> dict[str, list[Any]]:
    """
    Deconstruct a trajectory into a dictionary of lists. Info dicts of the original trajectory are flattened.
    """
    traj_dict = {
        "obs": [obs for obs, _, _, _, _, _ in trajectory],
        "acts": [act for _, act, _, _, _, _ in trajectory],
        "rewards": [reward for _, _, reward, _, _, _ in trajectory],
        "next_obs": [next_obs for _, _, _, next_obs, _, _ in trajectory],
        "dones": [done for _, _, _, _, done, _ in trajectory]
    }
    infos = [info for _, _, _, _, _, info in trajectory]
    traj_dict.update({k: [info[k] for info in infos] for k in infos[0].keys()})
    return traj_dict


def is_registered_gym_env(env_name: str) -> bool:
    """
    Check if a Gym environment is registered.
    """
    return env_name in gym.envs.registry


def get_obs_dim(env: gym.Env):
    obs_space = env.observation_space
    if isinstance(obs_space, spaces.Discrete):
        obs_dim = obs_space.n
    elif isinstance(obs_space, spaces.Box):
        obs_dim = obs_space.shape[0]
    elif isinstance(obs_space, spaces.Dict):
        obs_dim = obs_space["state_features"].shape[0]
    else:
        raise NotImplementedError(f"Observation space {obs_space} not implemented")
    return obs_dim


def get_act_dim(env: gym.Env):
    act_space = env.action_space
    if isinstance(act_space, spaces.Discrete):
        act_dim = act_space.n
    elif isinstance(act_space, spaces.Box):
        act_dim = act_space.shape[0]
    else:
        raise NotImplementedError(f"Action space {act_space} not implemented")
    return act_dim