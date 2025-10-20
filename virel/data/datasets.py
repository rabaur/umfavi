from torch.utils.data import Dataset
import numpy as np
import torch
from virel.utils.gym import simulate_trajectory
from virel.utils.math import sigmoid, softmax
from virel.utils.tabular import q_opt

class IIDDCTGridDataset(Dataset):
    """
    This dataset corresponds to the mock task in which the
    state-features are provided iid (simple regression).
    """
    def __init__(self, state_features: np.ndarray, n_actions: np.ndarray, rewards: np.ndarray):

        # Create augmented dataset with one-hot encoded actions
        self.state_action_feats = np.zeros((5 * state_features.shape[0], state_features.shape[1] + n_actions))
        self.flat_rewards = np.zeros((5 * state_features.shape[0], 1))

        for i in range(state_features.shape[0]):
            for a in range(n_actions):
                action_one_hot = np.zeros(n_actions)
                action_one_hot[a] = 1
                self.state_action_feats[i * 5 + a] = np.concatenate([state_features[i], action_one_hot])
                self.flat_rewards[i * 5 + a] = rewards[i, a]

    def __len__(self):
        return self.state_action_feats.shape[0]
    
    def __getitem__(self, idx):
        return self.state_action_feats[idx], self.flat_rewards[idx]


class PreferenceDataset(Dataset):
    """
    Dataset for preference learning with trajectory pairs and simulated preferences.
    """
    def __init__(
        self, 
        num_samples: int,
        T: np.ndarray,
        R_true: np.ndarray, 
        policy: np.ndarray,
        init_state_dist: np.ndarray,
        S_features: np.ndarray,
        n_actions: int,
        rationality: float = 1.0,
        trajectory_length: int = 100,
        seed: int = 42
    ):
        """
        Initialize preference dataset.
        
        Args:
            num_samples: Number of preference samples to generate
            T: Transition probability matrix (n_states, n_actions, n_states)
            R_true: True reward matrix (n_states, n_actions)
            policy: Policy matrix (n_states, n_actions)
            init_state_dist: Initial state distribution
            S: State features matrix (n_states, n_dct_features)
            n_actions: Number of actions
            rationality: True rationality parameter for preference generation
            trajectory_length: Length of each trajectory
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.T = T
        self.R_true = R_true
        self.policy = policy
        self.init_state_dist = init_state_dist
        self.S_features = S_features
        self.n_actions = n_actions
        self.rationality = rationality
        self.trajectory_length = trajectory_length
        
        # Set random seed
        np.random.seed(seed)
        
        # Generate trajectory pairs and preferences
        self.trajectory_pairs, self.preferences = self._generate_preferences()
        
        # Pre-compute trajectory features
        self.trajectory_features = self._compute_trajectory_features()
        
    def _compute_trajectory_return(self, trajectory: list[tuple[int, int, float]]) -> float:
        """
        Compute the total return of a trajectory.
        
        Args:
            trajectory: List of (state, action, reward) tuples
            
        Returns:
            Total return (sum of rewards)
        """
        return sum(reward for _, _, reward in trajectory)
    
    def _trajectory_to_features(self, trajectory: list[tuple[int, int, float]]) -> np.ndarray:
        """
        Convert a trajectory to state-action features.
        
        Args:
            trajectory: List of (state, action, reward) tuples
            
        Returns:
            Array of shape (trajectory_length, n_dct_features + n_actions)
        """
        trajectory_features = []
        
        for state, action, _ in trajectory:
            # Get state features
            state_features = self.S_features[state]
            
            # Create one-hot encoded action
            action_one_hot = np.zeros(self.n_actions)
            action_one_hot[action] = 1
            
            # Concatenate state features and action
            state_action_features = np.concatenate([state_features, action_one_hot])
            trajectory_features.append(state_action_features)
        
        return np.array(trajectory_features)
    
    def _compute_trajectory_features(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Pre-compute trajectory features for all trajectory pairs.
        
        Returns:
            List of (traj1_features, traj2_features) tuples
        """
        trajectory_features = []
        
        for traj1, traj2 in self.trajectory_pairs:
            traj1_features = self._trajectory_to_features(traj1)
            traj2_features = self._trajectory_to_features(traj2)
            trajectory_features.append((traj1_features, traj2_features))
        
        return trajectory_features
    
    def _generate_preferences(self) -> tuple[list[tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]], list[int]]:
        """
        Generate trajectory pairs and preferences.
        
        Returns:
            Tuple of (trajectory_pairs, preferences) where:
            - trajectory_pairs: List of (traj1, traj2) pairs
            - preferences: List of preferences (0 for traj1, 1 for traj2)
        """
        trajectory_pairs = []
        preferences = []
        
        for _ in range(self.num_samples):
            # Generate two trajectories using imported function
            traj1 = simulate_trajectory(
                self.T, self.R_true, self.policy, self.init_state_dist, self.trajectory_length
            )
            traj2 = simulate_trajectory(
                self.T, self.R_true, self.policy, self.init_state_dist, self.trajectory_length
            )
            
            # Compute true returns
            r1 = self._compute_trajectory_return(traj1)
            r2 = self._compute_trajectory_return(traj2)
            
            # Generate preference using sigmoid
            preference_prob = sigmoid(self.rationality * (r2 - r1))
            pref = np.random.binomial(1, preference_prob)
            
            trajectory_pairs.append((traj1, traj2))
            preferences.append(pref)
            
        return trajectory_pairs, preferences
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a preference sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (traj1_features, traj2_features, preference) where:
            - traj1_features: Tensor of shape (trajectory_length, n_dct_features + n_actions)
            - traj2_features: Tensor of shape (trajectory_length, n_dct_features + n_actions)
            - preference: 0 for traj1, 1 for traj2
        """
        traj1_features, traj2_features = self.trajectory_features[idx]
        preference = self.preferences[idx]
        
        return (torch.tensor(traj1_features, dtype=torch.float32), 
                torch.tensor(traj2_features, dtype=torch.float32), 
                torch.tensor(preference, dtype=torch.float32).unsqueeze(0))
    
    def get_all_data(self):
        """
        Get all trajectory pairs and preferences as numpy arrays.
        
        Returns:
            Tuple of (trajectory_pairs, preferences) as numpy arrays
        """
        return self.trajectory_pairs, np.array(self.preferences)


class DemonstrationDataset(Dataset):
    """
    Creates a dataset containing expert demonstrations, sampled according to:
    π(a | s) = exp(βQ(s, a)) / ∑_{a'} exp(βQ(s, a'))
    """
    def __init__(
        self,
        T: np.ndarray,
        R_true: np.ndarray,
        init_state_dist: np.ndarray,
        S_features: np.ndarray,
        n_actions: int,
        rationality: float = 1.0,
        gamma: float = 0.99,
        num_steps: int = 100,
        seed: int = 42,
        num_samples: int = 1000
    ):
        """
        Initialize demonstration dataset.
        
        Args:
            T: Transition probability matrix (n_states, n_actions, n_states)
            R_true: True reward matrix (n_states, n_actions)
            init_state_dist: Initial state distribution
            S_features: State features matrix (n_states, n_dct_features)
            n_actions: Number of actions
            rationality: True rationality parameter for expert policy
            gamma: Discount factor for Q-value computation
            num_steps: Length of each demonstration trajectory
            seed: Random seed for reproducibility
            num_samples: Number of demonstration trajectories to generate
        """
        self.T = T
        self.R_true = R_true
        self.init_state_dist = init_state_dist
        self.S_features = S_features
        self.n_actions = n_actions
        self.rationality = rationality
        self.gamma = gamma
        self.num_samples = num_samples
        self.num_steps = num_steps
        
        # Set random seed
        np.random.seed(seed)
        
        # Generate demonstrations
        self.demonstrations = self._generate_demonstrations()
        
        # Pre-compute trajectory features and indices
        self.trajectory_features, self.trajectory_indices = self._compute_trajectory_features_and_indices()

    def _generate_demonstrations(self):
        """Generate expert demonstrations using optimal Q-values."""
        # Compute optimal Q-values
        q_optimal = q_opt(self.T, self.R_true, self.gamma)
        
        # Create expert policy using softmax
        expert_policy = softmax(self.rationality * q_optimal, dims=1)
        
        # Generate trajectories
        trajectories = []
        for _ in range(self.num_samples):
            trajectory = simulate_trajectory(
                self.T, self.R_true, expert_policy, self.init_state_dist, self.num_steps
            )
            trajectories.append(trajectory)
        
        return trajectories

    def _trajectory_to_features_and_indices(self, trajectory: list[tuple[int, int, float]]) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert a trajectory to state-action features and indices.
        
        Args:
            trajectory: List of (state, action, reward) tuples
            
        Returns:
            Tuple of (features, indices) where:
            - features: Array of shape (trajectory_length, n_dct_features + n_actions)
            - indices: Array of shape (trajectory_length, 2) with (state_idx, action_idx) pairs
        """
        trajectory_features = []
        trajectory_indices = []
        
        for state, action, _ in trajectory:
            # Get state features
            state_features = self.S_features[state]
            
            # Create one-hot encoded action
            action_one_hot = np.zeros(self.n_actions)
            action_one_hot[action] = 1
            
            # Concatenate state features and action
            state_action_features = np.concatenate([state_features, action_one_hot])
            trajectory_features.append(state_action_features)
            trajectory_indices.append([state, action])
        
        return np.array(trajectory_features), np.array(trajectory_indices)
    
    def _compute_trajectory_features_and_indices(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Pre-compute trajectory features and indices for all demonstrations.
        
        Returns:
            Tuple of (trajectory_features, trajectory_indices) where:
            - trajectory_features: List of feature arrays
            - trajectory_indices: List of index arrays
        """
        trajectory_features = []
        trajectory_indices = []
        
        for trajectory in self.demonstrations:
            traj_features, traj_indices = self._trajectory_to_features_and_indices(trajectory)
            trajectory_features.append(traj_features)
            trajectory_indices.append(traj_indices)
        
        return trajectory_features, trajectory_indices
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a demonstration sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (features, indices) where:
            - features: Trajectory features tensor of shape (trajectory_length, n_dct_features + n_actions)
            - indices: Trajectory indices tensor of shape (trajectory_length, 2) with (state_idx, action_idx) pairs
        """
        trajectory_features = self.trajectory_features[idx]
        trajectory_indices = self.trajectory_indices[idx]
        return (torch.tensor(trajectory_features, dtype=torch.float32), 
                torch.tensor(trajectory_indices, dtype=torch.long))
    
    def get_all_demonstrations(self):
        """
        Get all demonstrations as a list.
        
        Returns:
            List of demonstration trajectories
        """
        return self.demonstrations
