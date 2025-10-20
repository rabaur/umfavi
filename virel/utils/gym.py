import gymnasium as gym
from typing import Callable

def rollout(env: gym.Env, policy: Callable, n_steps: int) -> list[list[tuple[dict, int, bool, dict]]]:
    ep = []
    obs, _ = env.reset()
    done = False
    ep = []
    step = 0
    while not done and step < n_steps:
        action = policy(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep.append((obs, action, reward, next_obs, done))
        obs = next_obs
        step += 1
    return ep

def get_obs_act_pairs(trajectory: list[tuple[dict, int, bool, dict]]) -> list[tuple[int, int]]:
    """
    Get state-action pairs from a trajectory.
    """
    obs = [obs["observation"] for obs, _, _, _, _ in trajectory]
    acts = [act for _, act, _, _, _ in trajectory]
    return obs, acts

def get_rewards(trajectory: list[tuple[dict, int, bool, dict]]) -> list[float]:
    """
    Get rewards from a trajectory.
    """
    return [reward for _, _, reward, _, _ in trajectory]