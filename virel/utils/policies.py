import gymnasium as gym
import numpy as np
from virel.utils.tabular import q_opt
from virel.utils.math import softmax

class UniformPolicy:
    """
    Generic uniform policy for any Gymnasium-compatible environment.
    """

    def __init__(self, action_space: gym.Space):
        self.action_space = action_space
    
    def __call__(self, observation=None):
        return self.action_space.sample()


class ExpertPolicy:
    """
    Expert policy for DCTGridEnv based on optimal Q-values.
    Uses softmax over Q-values: π(a|s) = exp(β*Q(s,a)) / Σ_a' exp(β*Q(s,a'))
    """

    def __init__(self, env, rationality: float = 1.0, gamma: float = 0.99):
        """
        Initialize expert policy.
        
        Args:
            env: DCTGridEnv environment
            rationality: Rationality parameter (β) for softmax policy
            gamma: Discount factor for Q-value computation
        """
        self.env = env
        self.rationality = rationality
        self.gamma = gamma
        
        # Compute transition matrix and get reward matrix from env
        # For DCTGridEnv, we need to construct the transition matrix
        self.T = self._build_transition_matrix()
        self.R = env.R
        
        # Compute optimal Q-values
        self.Q_optimal = q_opt(self.T, self.R, self.gamma)
        
        # Create softmax policy: π(a|s) = exp(β*Q(s,a)) / Σ_a' exp(β*Q(s,a'))
        self.policy = softmax(self.rationality * self.Q_optimal, dims=1)
    
    def _build_transition_matrix(self) -> np.ndarray:
        """
        Build transition matrix for DCTGridEnv.
        For now, assumes deterministic transitions (p_rand = 0).
        """
        grid_size = self.env.grid_size
        n_states = grid_size ** 2
        n_actions = self.env.action_space.n
        
        T = np.zeros((n_states, n_actions, n_states))
        
        # For deterministic case, copy logic from dct_grid_env
        from virel.envs.dct_grid_env import Action, succ_state_deterministic
        
        for i in range(grid_size):
            for j in range(grid_size):
                s = i * grid_size + j
                for a in range(n_actions):
                    # Deterministic case: agent always goes to intended state
                    i_next, j_next = succ_state_deterministic(i, j, Action(a), grid_size)
                    s_next = i_next * grid_size + j_next
                    T[s, a, s_next] = 1.0
        
        return T
    
    def __call__(self, observation):
        """
        Sample action from the expert policy.
        
        Args:
            observation: Observation from environment (dict with 'coord' and 'observation')
            
        Returns:
            Action sampled from the expert policy
        """
        # Extract state index from observation
        if isinstance(observation, tuple):
            # Handle reset return format (obs, info)
            observation = observation[0]
        
        coord = observation['coord']
        state_idx = coord[0] * self.env.grid_size + coord[1]
        
        # Sample action from policy distribution
        action = np.random.choice(self.env.action_space.n, p=self.policy[state_idx])
        
        return action