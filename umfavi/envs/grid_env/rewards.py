import numpy as np
from umfavi.envs.grid_env.actions import Action, action_diffs_coords

def succ_state_deterministic(i: int, j: int, a: Action, grid_size: int):
    """Implements boundary checks and returns the successor state for a deterministic action."""
    s_diff = action_diffs_coords[a]
    i_new = i + s_diff[0]
    j_new = j + s_diff[1]
    if i_new < 0 or i_new >= grid_size or j_new < 0 or j_new >= grid_size:
        return i, j
    return i_new, j_new

def reward_sparse(grid_size: int, goal_state_offset: int = 2) -> np.ndarray:
    """
    Creates a sparse reward function for the grid environment.
    """
    n_S = grid_size ** 2
    n_A = 5
    R = np.full((n_S, n_A), -0.1)
    s_goal_idx = (grid_size - goal_state_offset) * grid_size + (grid_size - goal_state_offset)

    # All actions in the goal state have reward 1
    R[s_goal_idx, Action.STAY] = 1
    R[s_goal_idx, Action.LEFT] = 1
    R[s_goal_idx, Action.RIGHT] = 1
    R[s_goal_idx, Action.UP] = 1
    R[s_goal_idx, Action.DOWN] = 1

    return R

def reward_factory(grid_size: int, reward_type: str):
    """
    Creates reward tables for the grid environment of different types.
    
    Returns:
        R (np.ndarray): The ground-truth reward function, s.t. R[s, a] is the reward for state s under action a.
    """
    n_S = grid_size ** 2
    n_A = 5
    R = np.zeros((n_S, n_A))
    if reward_type == "sparse":
        R = reward_sparse(grid_size)
    elif reward_type == "dense":
        # Initialize with dense reward everywhere first, overwrite special cases later
        for s in range(n_S):
            i = s // grid_size
            j = s % grid_size

            R[s, Action.STAY] = -1

            # away from goal state
            R[s, Action.LEFT] = -4 if succ_state_deterministic(i, j, Action.LEFT, grid_size) != (i, j) else -1
            R[s, Action.UP] = -4 if succ_state_deterministic(i, j, Action.UP, grid_size) != (i, j) else -1

            # towards goal state
            R[s, Action.RIGHT] = 2 if succ_state_deterministic(i, j, Action.RIGHT, grid_size) != (i, j) else -1
            R[s, Action.DOWN] = 2 if succ_state_deterministic(i, j, Action.DOWN, grid_size) != (i, j) else -1
        
        # goal state
        s_goal_idx = n_S - 1
        R[s_goal_idx, Action.STAY] = 3
        R[s_goal_idx, Action.LEFT] = 0
        R[s_goal_idx, Action.RIGHT] = 3 # stays in goal state (out of bounds)
        R[s_goal_idx, Action.UP] = 0
        R[s_goal_idx, Action.DOWN] = 3 # stays in goal state (out of bounds)
    elif reward_type == "path":
        # top edge
        for j in range(1, grid_size):
            s_idx = j
            R[s_idx, Action.STAY] = -1
            R[s_idx, Action.LEFT] = -1
            R[s_idx, Action.RIGHT] = -1
            R[s_idx, Action.UP] = -1
            R[s_idx, Action.DOWN] = -1
        # bottom edge
        for j in range(0, grid_size - 1):
            s_idx = grid_size * (grid_size - 1) + j
            R[s_idx, Action.STAY] = -1
            R[s_idx, Action.LEFT] = -1
            R[s_idx, Action.RIGHT] = -1
            R[s_idx, Action.UP] = -1
            R[s_idx, Action.DOWN] = -1
        # goal state
        s_goal_idx = n_S - 1
        R[s_goal_idx, Action.STAY] = 4
        R[s_goal_idx, Action.LEFT] = 0
        R[s_goal_idx, Action.RIGHT] = 4 # stays in goal state (out of bounds)
        R[s_goal_idx, Action.UP] = 0
        R[s_goal_idx, Action.DOWN] = 4 # stays in goal state (out of bounds)
    elif reward_type == "cliff":
        # top edge (less punishing than bottom edge)
        for j in range(1, grid_size):
            s_idx = j
            R[s_idx, Action.STAY] = -1
            R[s_idx, Action.LEFT] = -1
            R[s_idx, Action.RIGHT] = -1
            R[s_idx, Action.UP] = -1
            R[s_idx, Action.DOWN] = -1
        # bottom edge
        for j in range(0, grid_size - 1):
            s_idx = grid_size * (grid_size - 1) + j
            R[s_idx, Action.STAY] = -4
            R[s_idx, Action.LEFT] = -4
            R[s_idx, Action.RIGHT] = -4
            R[s_idx, Action.UP] = -4
            R[s_idx, Action.DOWN] = -4
        # goal state
        s_goal_idx = n_S - 1
        R[s_goal_idx, Action.STAY] = 4
        R[s_goal_idx, Action.LEFT] = 0
        R[s_goal_idx, Action.RIGHT] = 4 # stays in goal state (out of bounds)
        R[s_goal_idx, Action.UP] = 0
        R[s_goal_idx, Action.DOWN] = 4 # stays in goal state (out of bounds)
    elif reward_type == "five_goals":
        # Define the 5 high-reward state coordinates
        # 4 corners + center
        goal_states = [
            (0, 0, 1),                              # top left
            (0, grid_size - 1, 1),                  # top right
            (grid_size - 1, 0, 1),                  # bottom left
            (grid_size - 1, grid_size - 1, 1),     # bottom right (very high reward)
            (grid_size // 2, grid_size // 2, 1)     # center
        ]
        
        # For each state in the grid, check if any action leads to a goal state
        for s in range(n_S):
            i = s // grid_size
            j = s % grid_size
            
            for a in Action:
                i_next, j_next = succ_state_deterministic(i, j, a, grid_size)
                
                # Check if this action leads to any goal state
                for goal_i, goal_j, reward_value in goal_states:
                    if (i_next, j_next) == (goal_i, goal_j):
                        # Reward for entering or staying in the goal state
                        R[s, a] = reward_value
                        break
    elif reward_type == "gaussian_goals":
        # Define the 5 high-reward state coordinates
        # 4 corners + center
        goal_states = [
            (0, 0, 1),                              # top left
            (0, 1, 1),                  # top right
            (1, 0, 1),                  # bottom left
            (1, 1, 2),     # bottom right (very high reward)
            (0.5, 0.5, 1)     # center
        ]
        
        # Add a Gaussian centered at each goal state with a standard deviation of 0.1
        xs, ys = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size), indexing='ij')
        n_S = grid_size ** 2
        R = np.zeros((grid_size, grid_size))
        for goal_x, goal_y, scale in goal_states:
            R += scale * np.exp(-((xs - goal_x) ** 2 + (ys - goal_y) ** 2) / (2 * 0.1 ** 2))
        R_flat = R.reshape(n_S, -1).squeeze()
        R = np.repeat(R_flat[:, None], n_A, axis=1) # Each action has same reward for all states

    return R