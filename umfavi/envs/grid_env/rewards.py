import numpy as np
from numpy.typing import NDArray
from umfavi.envs.grid_env.actions import Action, ACTION_DELTA
from umfavi.utils.reward import Rs_to_Rsas, shape

def convert_to_shift_and_axis(action_delta: NDArray):
    axis = np.argmax(np.abs(action_delta))
    shift = -action_delta[axis]
    return axis, shift


def succ_state(i: int, j: int, a: Action, grid_size: int):
    """Implements boundary checks and returns the successor state for a deterministic action."""
    s_diff = ACTION_DELTA[a]
    i_new = i + s_diff[0]
    j_new = j + s_diff[1]
    if i_new < 0 or i_new >= grid_size or j_new < 0 or j_new >= grid_size:
        return i, j
    return i_new, j_new

def to_flat_idx(i: int, j: int, grid_size):
    return i * grid_size + j

def reward_sparse(grid_size: int, goal_position: tuple[int, int]):
    R1d = np.zeros((grid_size, grid_size))
    R1d[goal_position[0], goal_position[1]] = 1
    R1d = np.reshape(R1d, (grid_size**2))
    R3d = Rs_to_Rsas(R1d, 5)
    return R3d

def reward_dense(grid_size: int, gamma: float):

    # Base reward
    R1d = np.zeros((grid_size, grid_size))
    R1d[-1, -1] = 1.0
    R1d = R1d.reshape(grid_size**2)
    R3d_base = Rs_to_Rsas(R1d, 5)

    # State-wise scalar
    ii, jj = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='ij')
    f = np.abs(ii) + np.abs(jj)
    f = -np.rot90(np.rot90(f))
    f = f.reshape(grid_size**2)

    # Shaping
    return shape(R3d_base, f, gamma)

def reward_factory(grid_size: int, reward_type: str, gamma: float):
    """
    Creates reward tables for the grid environment of different types.
    
    Returns:
        R (np.ndarray): The ground-truth reward function, s.t. R[s, a] is the reward for state s under action a.
    """
    n_S = grid_size ** 2
    n_A = 5
    R = np.zeros((n_S, n_A, n_S))
    if reward_type == "sparse":
        R = reward_sparse(grid_size, goal_position=(grid_size - 1, grid_size - 1))
    elif reward_type == "dense":
        R = reward_dense(grid_size, gamma)
    elif reward_type == "center":
        R = reward_sparse(grid_size, goal_position=(grid_size // 2, grid_size // 2))
    else:
        raise NotImplementedError(f"Reward type {reward_type} is not implemented")
    return R