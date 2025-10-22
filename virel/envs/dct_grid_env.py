import re
import networkx as nx
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import IntEnum

class Action(IntEnum):
    # counter-clockwise order
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    STAY = 4

action_diffs_coords = {
    Action.RIGHT: np.array([0, 1]),
    Action.UP: np.array([-1, 0]),
    Action.LEFT: np.array([0, -1]),
    Action.DOWN: np.array([1, 0]),
    Action.STAY: np.array([0, 0])
}

def dct_features(grid_size: int, n_dct_basis_fns: int) -> np.ndarray:
    """
    Creates 2D DCT-II features on an N×N grid over [0,1]^2
    using the first n_dct_basis_fns in each dimension.
    """
    N = grid_size
    K = n_dct_basis_fns

    # grid coordinates
    x = np.arange(N)
    y = np.arange(N)
    xv, yv = np.meshgrid(x, y, indexing='ij')

    # DCT-II normalization factors
    def alpha(p):
        return np.where(p == 0, np.sqrt(1.0 / N), np.sqrt(2.0 / N))

    # compute basis functions
    feats = []
    for u in range(K):
        au = alpha(u)
        cos_u = np.cos(np.pi * (2 * xv + 1) * u / (2 * N))
        for v in range(K):
            av = alpha(v)
            cos_v = np.cos(np.pi * (2 * yv + 1) * v / (2 * N))
            feats.append(au * av * cos_u * cos_v)

    # stack into (N, N, K*K)
    return np.stack(feats, axis=-1)

def custom_layout(G: nx.Graph, grid_size: int):
    """
    Maps state node names in the format r"$s_{i}$" to [x, y] coordinates.
    Maps action node names in the format r"$s_{i}, a_{j}$" to [x, y] coordinates.
    State nodes are placed on the grid points, action nodes are placed in the middle of the edges between states.
    """
    rot_angle = np.pi / 16
    action_diffs_spatial = {
        Action.RIGHT: np.array([np.cos(rot_angle), np.sin(rot_angle)]),
        Action.UP: np.array([np.cos(np.pi / 2 + rot_angle), np.sin(np.pi / 2 + rot_angle)]),
        Action.LEFT: np.array([np.cos(np.pi + rot_angle), np.sin(np.pi + rot_angle)]),
        Action.DOWN: np.array([np.cos(-(np.pi / 2) + rot_angle), np.sin(-(np.pi / 2) + rot_angle)]),
        Action.STAY: np.array([0, 0])
    }

    pos = {}
    pattern_state = re.compile(r'\$s_\{(\d+)\}\$')
    pattern_action = re.compile(r'\$s_\{(\d+)\},\s*a_\{(\d+)\}\$')
    for node_name, node_data in G.nodes(data=True):

        if node_data['type'] == 'state':
            match = pattern_state.match(node_name)
            if match:
                s_idx = int(match.group(1)) - 1

                # Inverting the y coordinates makes the grid look like a normal grid, with (0, 0) in the top left corner
                pos[node_name] = np.array([s_idx % grid_size, -(s_idx // grid_size)])
        elif node_data['type'] == 'action':
            match = pattern_action.match(node_name)
            if match:
                s_idx = int(match.group(1)) - 1
                a_idx = int(match.group(2)) - 1
                
                # find base node position
                s_pos = np.array([s_idx % grid_size, -(s_idx // grid_size)])

                # determine action position relative to base node
                if a_idx == Action.STAY:
                    # Slightly offset the stay node to the left and up
                    pos[node_name] = s_pos + 1/6 * np.array([-1, 1])
                else:
                    pos[node_name] = s_pos + 1/3 * action_diffs_spatial[Action(a_idx)] # (grid_size, grid_size) is the bottom right corner on the plot, so we need to invert the action diffs
    return pos

def succ_state_deterministic(i: int, j: int, a: Action, grid_size: int):
    """Implements boundary checks and returns the successor state for a deterministic action."""
    s_diff = action_diffs_coords[a]
    i_new = i + s_diff[0]
    j_new = j + s_diff[1]
    if i_new < 0 or i_new >= grid_size or j_new < 0 or j_new >= grid_size:
        return i, j
    return i_new, j_new


def reward_factory(grid_size: int, reward_type: str):
    """
    Creates reward tables for the grid environment of different types.

    Args:
        grid_size (int): The number of grid points in each dimension.
        reward_type (str): The type of reward function to create. Options are:
            - "sparse": 1 reward for reaching and staying in state (grid_size - 1, grid_size - 1), 0 otherwise
            - "dense": This reward is a shaped version of the sparse environment, which a denser reward signal towards the goal
                - +3 reward for staying within state (grid_size - 1, grid_size - 1)
                - -1 reward for staying in any other state
                - +2 reward for moving towards the goal state (actions "right" and "down"), except when that action would move the agent out of bounds, then the reward is -1 (like stay)
                - -4 reward for moving away from the goal state (actions "left", and "up"), except in the goal state, where these actions have value 0, and when that action would move the agent out of bounds, then the reward is -1 (like stay)
            - "path": This environment forms a path enclosed between punishing "edges"
                - +4 reward for reaching and staying within state (grid_size - 1, grid_size - 1)
                - -1 reward for entering and staying in states (0, :) and (grid_size - 1, :) (top and bottom edges), except (grid_size - 1, grid_size - 1) and (0, 0)
                - 0 reward for all other states
            - "cliff": This environment is similar to "path", but the bottom edge is more punishing than the top edge
                - +4 reward for reaching and staying within state (grid_size - 1, grid_size - 1)
                - -1 reward for entering and staying in states (0, :) (top edge), except (0, 0)
                - -4 reward for entering and staying in states (grid_size - 1, :) (bottom edge), except (grid_size - 1, grid_size - 1)
                - 0 reward for all other transitions
            - "five_goals": 5 high-reward states (4 corners + center), all other states have 0 reward
                - +10 reward for reaching and staying within state (grid_size - 1, grid_size - 1) (bottom right corner)
                - +3 reward for reaching and staying within states (0, 0) (top left), (0, grid_size - 1) (top right), (grid_size - 1, 0) (bottom left), and (grid_size // 2, grid_size // 2) (center)
                - 0 reward for all other states and actions
    
    Returns:
        R (np.ndarray): The ground-truth reward function, s.t. R[s, a] is the reward for state s under action a.
    """
    n_S = grid_size ** 2
    n_A = 5
    R = np.zeros((n_S, n_A))
    if reward_type == "sparse":
        s_goal_idx = (grid_size - 1) * grid_size + (grid_size - 1)
        R[s_goal_idx, Action.STAY] = 1 # stays in goal state
        R[s_goal_idx, Action.LEFT] = 0 # leaves goal state
        R[s_goal_idx, Action.RIGHT] = 1 # stays in goal state (out of bounds)
        R[s_goal_idx, Action.UP] = 0 # leaves goal state
        R[s_goal_idx, Action.DOWN] = 1 # stays in goal state (out of bounds)
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
        s_goal_idx = (grid_size - 1) * grid_size + (grid_size - 1)
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
        s_goal_idx = (grid_size - 1) * grid_size + (grid_size - 1)
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
        s_goal_idx = (grid_size - 1) * grid_size + (grid_size - 1)
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
            (grid_size - 1, grid_size - 1, 2),     # bottom right (very high reward)
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
    return R

def dct_grid_env(grid_size: int, n_dct_basis_fns: int, reward_type: str, p_rand: float):
    """
    Creates a grid environment on the [0, 1]^2 square with DCT coefficents as state-features.
    Let n_S be the number of states.

    Args:
        grid_size (int): The number of grid points in each dimension.
        n_dct_basis_fns (int): The number of DCT basis functions.
        reward_type (str): The type of ground truth reward function. See `reward_factory` for options.
        p_rand (float): The probability of transitioning to a random state.

    Returns:
        P (np.ndarray): The transition probability matrix, s.t. P[s, a, s'] is the probability of transitioning from state s to state s' under action a.
        R (np.ndarray): The ground-truth reward function, s.t. R[s, a] is the reward for state s under action a.
        S (np.ndarray): The state-feature matrix, s.t. S[s, :] is the feature vector for state s.
    """
    # Create the transition probability matrix
    n_S = grid_size ** 2
    n_A = len(Action)
    P = np.zeros((n_S, n_A, n_S))

    # rows
    for i in range(grid_size):

        # cols
        for j in range(grid_size):

            # state index
            s = i * grid_size + j
            for a in range(n_A):
                for a_prime in range(n_A): # all the other actions that could be taken at random
                    i_prime, j_prime = succ_state_deterministic(i, j, Action(a_prime), grid_size)
                    s_prime = i_prime * grid_size + j_prime
                    
                    # If this is the intended action, we go to the new state with probability 1 - p_rand
                    if a == a_prime:
                        P[s, a, s_prime] = 1 - p_rand
                    # If this is a random action, we go to the new state with probability p_rand / (n_S - 1)
                    else:
                        P[s, a, s_prime] += p_rand / (n_S - 1)
    
    # Create reward matrix
    R = reward_factory(grid_size, reward_type)

    # Create state-feature matrix.
    S_square = dct_features(grid_size, n_dct_basis_fns)
    S = S_square.reshape((grid_size ** 2, -1))

    return P, R, S

class DCTGridEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size, n_dct_basis_fns, reward_type, p_rand):
        """
        Creates a discrete grid environment with Discrete-Cosine-Transform base-functions as observed features.

        Args:
            grid_size (int): The number of grid points in each dimension.
            n_dct_basis_fns (int): The number of DCT basis functions.
            reward_type (str): The type of ground truth reward function. Options are: "sparse", "dense", "path", "cliff", "five_goals".
            p_rand (float): The probability of transitioning to a random state. Must be in [0, 1].
        """
        super().__init__()
        self.grid_size = grid_size
        self.n_dct_basis_fns = n_dct_basis_fns
        self.reward_type = reward_type
        assert 0 <= p_rand <= 1, "p_rand must be in [0, 1]"
        assert reward_type in ["sparse", "dense", "path", "cliff", "five_goals"], "Invalid reward type"
        self.p_rand = p_rand

        # Compute S (features) etc. using existing code
        _, self.R, self.S = dct_grid_env(grid_size, n_dct_basis_fns, reward_type, p_rand)

        # Action space
        self.action_space = spaces.Discrete(len(Action))

        # Observation space
        self.observation_space = spaces.Dict({
            "coord": spaces.Box(low=np.array([0,0]), high=np.array([grid_size-1, grid_size-1]), dtype=np.int32, shape=(2,)),
            "observation": spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32, shape=(self.S.shape[1],))
        })
        self.state_coord = None  # will hold (i,j)
        self.state_idx = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Choose a random (i,j)
        i = self.np_random.integers(self.grid_size)
        j = self.np_random.integers(self.grid_size)
        self.state_coord = (i, j)
        self.state_idx = i * self.grid_size + j
        features = self.S[self.state_idx].astype(np.float32)
        obs = {"coord": np.array(self.state_coord, dtype=np.int32),
               "observation": features}
        info = {}
        return obs, info

    def step(self, action):
        # update (i,j) deterministically or with randomness
        i, j = self.state_coord
        i2, j2 = succ_state_deterministic(i, j, Action(action), self.grid_size)
        self.state_coord = (i2, j2)
        self.state_idx = i2 * self.grid_size + j2
        features = self.S[self.state_idx].astype(np.float32)
        reward = self.R[self.state_idx, action]
        terminated = False
        truncated = False
        info = {}
        obs = {"coord": np.array(self.state_coord, dtype=np.int32),
               "observation": features}
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        pass  # implement optional

    def close(self):
        pass
