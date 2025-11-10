import gymnasium as gym
from typing import Any, Callable
from umfavi.envs.grid_env.env import GridEnv

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

def extract_obs_state_actions(
    trajectory: list[tuple[dict, int, bool, dict]],
    env: GridEnv
) -> list[tuple[Any]]:
    """
    Get state-action pairs from a trajectory.
    """
    states = [state["state"] for state, _, _, _, _ in trajectory]
    state_feats = [state_feat["state_features"] for state_feat, _, _, _, _ in trajectory]
    acts = [act for _, act, _, _, _ in trajectory]
    act_feats = [env.A[act] for act in acts]
    return {
        "states": states,
        "state_feats": state_feats,
        "acts": acts,
        "act_feats": act_feats,
    }

def get_rewards(trajectory: list[tuple[dict, int, bool, dict]]) -> list[float]:
    """
    Get rewards from a trajectory.
    """
    return [reward for _, _, reward, _, _ in trajectory]