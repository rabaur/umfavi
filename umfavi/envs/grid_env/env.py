import gymnasium as gym
from gymnasium import spaces
import numpy as np
from umfavi.envs.grid_env.actions import Action, action_diffs_coords
from umfavi.envs.grid_env.features import coordinate_features
from umfavi.envs.grid_env.rewards import reward_factory

def succ_state_deterministic(i: int, j: int, a: Action, grid_size: int):
    """Implements boundary checks and returns the successor state for a deterministic action."""
    s_diff = action_diffs_coords[a]
    i_new = i + s_diff[0]
    j_new = j + s_diff[1]
    if i_new < 0 or i_new >= grid_size or j_new < 0 or j_new >= grid_size:
        return i, j
    return i_new, j_new

def dct_grid_env(grid_size: int, reward_type: str, p_rand: float):
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
                    # If this is a random action, we go to the state corresponding
                    # to the random action with probability p_rand / (n_A - 1)
                    # (p_rand is distributed uniformly over n_A - 1 other random actions)
                    else:
                        P[s, a, s_prime] += p_rand / (n_A - 1)
    
    # Create reward matrix
    R = reward_factory(grid_size, reward_type)

    # Create state-feature matrix.
    # S_square = dct_features(grid_size, n_dct_basis_fns)
    S_square = coordinate_features(grid_size)
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
        assert reward_type in ["sparse", "dense", "path", "cliff", "five_goals", "gaussian_goals"], "Invalid reward type"
        self.p_rand = p_rand

        # Compute S (features) etc. using existing code
        self.P, self.R, self.S = dct_grid_env(grid_size, n_dct_basis_fns, reward_type, p_rand)
        # Action space
        self.action_space = spaces.Discrete(len(Action))

        # Observation space
        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=np.array([0,0]), high=np.array([grid_size-1, grid_size-1]), dtype=np.int32, shape=(2,)),
            "observation": spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32, shape=(self.S.shape[1],))
        })
        self.state_coord = None  # will hold (i,j)
        self.state_idx = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # # Choose a random (i,j)
        # i = self.np_random.integers(self.grid_size)
        # j = self.np_random.integers(self.grid_size)
        i = 0
        j = 0
        self.state_coord = (i, j)
        self.state_idx = i * self.grid_size + j
        features = self.S[self.state_idx].astype(np.float32)
        obs = {"state": np.array(self.state_coord, dtype=np.int32),
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
        obs = {"state": np.array(self.state_coord, dtype=np.int32),
               "observation": features}
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        pass  # implement optional

    def close(self):
        pass
