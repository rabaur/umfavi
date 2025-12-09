import copy
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# MiniGrid is an optional dependency; the wrapper is only used when it is installed.
from minigrid.wrappers import FullyObsWrapper, SymbolicObsWrapper, ReseedWrapper, FlatObsWrapper

from umfavi.envs.grid_env.action_features import action_feature_factory


class MiniGridWrapper(gym.Env):
    """
    A wrapper to adapt MiniGrid environments to the observation/action
    structure expected by the rest of this codebase.

    - Wraps the env with ReseedWrapper + FullyObsWrapper + SymbolicObsWrapper
      for deterministic, symbolic, fully observable state.
    - Exposes observations as a dict with keys {"state", "state_features"} to
      mirror GridEnv.
    - Builds a tabular transition/reward model (P, R) by exhaustive traversal
      of reachable states (useful for expert policy/regret metrics).
    - Provides action features `A` (one-hot by default) for downstream code.
    """

    def __init__(
        self,
        env: gym.Env,
        action_feature_type: str = "one_hot",
        seeds: list | None = None,
        max_states: int = 5000,
    ):
        super().__init__()
        # Ensure deterministic reset and fully observable symbolic observations
        env = ReseedWrapper(env, seeds=seeds or [0], seed_idx=0)
        env = FullyObsWrapper(env)
        env = SymbolicObsWrapper(env)
        env = FlatObsWrapper(env)
        self.env = env

        self.action_space = self.env.action_space

        # Derive grid size if available (for state bounds)
        self.grid_size = getattr(self.env.unwrapped, "width", None)

        # Obtain a sample observation to size spaces/features
        sample_obs, _ = self.env.reset()
        sample_feats = self._extract_state_features(sample_obs)

        # Observation space mirrors GridEnv: coordinates + features
        coord_high = np.array(
            [self.grid_size - 1, self.grid_size - 1] if self.grid_size else [np.iinfo(np.int32).max] * 2,
            dtype=np.int32,
        )
        '''self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=np.zeros(2, dtype=np.int32),
                    high=coord_high,
                    dtype=np.int32,
                    shape=(2,),
                ),
                "state_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    dtype=np.float32,
                    shape=sample_feats.shape,
                ),
            }
        )'''
        self.observation_space = self.env.observation_space  # --- IGNORE ---

        # Build tabular model over reachable states for expert policy/regret
        self._build_tabular_model(sample_obs=sample_obs, sample_feats=sample_feats, max_states=max_states)

        # Action features (one-hot or embedding)
        self.A = action_feature_factory(action_feature_type, n_actions=self.action_space.n)

        # Reset underlying env after model construction
        self.env.reset()

    # ------------- Observation helpers -------------
    def _extract_state_features(self, obs) -> np.ndarray:
        """
        Flatten the symbolic observation into a 1D float32 feature vector.
        Ignores non-array entries such as mission strings.
        """
        if isinstance(obs, dict):
            parts = []
            for _, val in obs.items():
                if isinstance(val, str):
                    continue  # skip mission text
                arr = np.asarray(val)
                parts.append(arr.reshape(-1))
            if not parts:
                return np.zeros((0,), dtype=np.float32)
            flat = np.concatenate(parts).astype(np.float32)
        else:
            flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        return flat

    def _extract_agent_pos(self, obs) -> np.ndarray:
        """Get the agent (row, col) position from observation or env state."""
        if isinstance(obs, dict) and "agent_pos" in obs:
            pos = np.asarray(obs["agent_pos"], dtype=np.int32)
            if pos.shape[0] >= 2:
                return pos[:2]
        if hasattr(self.env.unwrapped, "agent_pos"):
            pos = np.asarray(self.env.unwrapped.agent_pos, dtype=np.int32)
            if pos.shape[0] >= 2:
                return pos[:2]
        # Fallback
        return np.zeros(2, dtype=np.int32)

    def _wrap_obs(self, obs) -> dict:
        state_features = self._extract_state_features(obs)
        state_coord = self._extract_agent_pos(obs)
        return {
            "state": state_coord,
            "state_features": state_features,
        }

    def _obs_hash(self, wrapped_obs: dict) -> bytes:
        """Hash state_features for dictionary lookup."""
        return np.asarray(wrapped_obs["state_features"], dtype=np.float32).tobytes()

    # ------------- Tabular model construction -------------
    def _build_tabular_model(self, sample_obs, sample_feats, max_states: int):
        """
        Exhaustively traverse reachable states using deep copies to build a
        deterministic transition table. Adds an absorbing terminal state for
        episode terminations.
        """
        wrapped_init = self._wrap_obs(sample_obs)
        init_hash = self._obs_hash(wrapped_init)

        state_hash_to_idx: dict[bytes, int] = {init_hash: 0}
        state_features = [wrapped_init["state_features"]]
        state_coords = [self._extract_agent_pos(sample_obs)]
        env_snapshots: list[gym.Env | None] = [copy.deepcopy(self.env)]
        transitions = {}
        queue = deque([0])

        terminal_idx = None
        feature_dim = sample_feats.shape[0]

        while queue and len(state_features) < max_states:
            s_idx = queue.popleft()
            env_at_state = env_snapshots[s_idx]
            # If this is absorbing terminal, skip expansion
            if env_at_state is None:
                continue

            for a in range(self.action_space.n):
                env_copy = copy.deepcopy(env_at_state)
                obs_next, reward, terminated, truncated, _ = env_copy.step(a)
                wrapped_next = self._wrap_obs(obs_next)

                # Handle terminal transitions
                if terminated or truncated:
                    if terminal_idx is None:
                        terminal_idx = len(state_features)
                        state_hash_to_idx[b"__terminal__"] = terminal_idx
                        state_features.append(np.zeros(feature_dim, dtype=np.float32))
                        state_coords.append(np.array([np.nan, np.nan], dtype=np.float32))
                        env_snapshots.append(None)
                    # Map terminal observation hash to terminal index as well
                    h_next = self._obs_hash(wrapped_next)
                    state_hash_to_idx[h_next] = terminal_idx
                    s_prime = terminal_idx
                else:
                    h_next = self._obs_hash(wrapped_next)
                    if h_next not in state_hash_to_idx:
                        s_prime = len(state_features)
                        state_hash_to_idx[h_next] = s_prime
                        state_features.append(wrapped_next["state_features"])
                        state_coords.append(self._extract_agent_pos(obs_next))
                        env_snapshots.append(copy.deepcopy(env_copy))
                        queue.append(s_prime)
                    else:
                        s_prime = state_hash_to_idx[h_next]

                transitions[(s_idx, a)] = (s_prime, reward)

        n_states = len(state_features)
        n_actions = self.action_space.n
        P = np.zeros((n_states, n_actions, n_states), dtype=np.float32)
        R = np.zeros((n_states, n_actions, n_states), dtype=np.float32)

        for (s, a), (s_prime, r) in transitions.items():
            P[s, a, s_prime] = 1.0
            R[s, a, s_prime] = r

        # Terminal self-loops with zero reward
        if terminal_idx is not None:
            P[terminal_idx, :, terminal_idx] = 1.0

        self.S = np.stack(state_features, axis=0).astype(np.float32)
        self.P = P
        self.R = R
        self.transitions = transitions  # save for debugging
        self.state_hash_to_idx = state_hash_to_idx
        self.terminal_idx = terminal_idx
        self.state_coords = np.stack(state_coords, axis=0).astype(np.float32)

    def obs_to_state_idx(self, observation) -> int:
        """Map a wrapped observation to its tabular state index."""
        if isinstance(observation, tuple):
            observation = observation[0]
        if isinstance(observation, dict) and "state_features" in observation:
            key = self._obs_hash(observation)
        else:
            key = self._obs_hash(self._wrap_obs(observation))
        if key in self.state_hash_to_idx:
            return self.state_hash_to_idx[key]
        # Fall back to terminal if unknown
        if self.terminal_idx is not None:
            return self.terminal_idx
        raise KeyError("Observation not in tabular model and no terminal state defined.")

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        #wrapped_obs = self._wrap_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # wrapped_obs = self._wrap_obs(obs)
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        return self.env.render()

    def close(self):
        self.env.close()

    # ------------- Debug visualization -------------
    def debug_plot_tabular_model(self, figsize=(6, 6)):
        """
        Quick scatter/graph of reachable states and transitions for debugging.
        Returns (fig, ax).
        """
        import matplotlib.pyplot as plt

        coords = self.state_coords
        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(coords[:, 1], coords[:, 0], c="C0", label="states")
        if self.terminal_idx is not None:
            ax.scatter(
                coords[self.terminal_idx, 1],
                coords[self.terminal_idx, 0],
                c="red",
                marker="x",
                s=80,
                label="terminal",
            )

        for idx, (r, c) in enumerate(coords):
            if np.isnan(r) or np.isnan(c):
                continue
            ax.text(c + 0.05, r + 0.05, str(idx), fontsize=8)

        for (s, a), (s_prime, _) in self.transitions.items():
            if s == self.terminal_idx:
                continue
            src = coords[s]
            dst = coords[s_prime]
            if np.any(np.isnan(src)) or np.any(np.isnan(dst)):
                continue
            ax.arrow(
                src[1],
                src[0],
                dst[1] - src[1],
                dst[0] - src[0],
                head_width=0.1,
                head_length=0.1,
                length_includes_head=True,
                alpha=0.15,
                color="gray",
            )

        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        ax.set_title("MiniGrid tabular model (reachable states)")
        ax.legend(loc="best")
        return fig, ax

    def debug_plot_state_distribution(self, counts=None, figsize=(6, 6), cmap="Reds"):
        """
        Overlay a state distribution on top of the MiniGrid render.

        Args:
            counts: array-like of shape (n_states,). If None, uses uniform counts.
            figsize: matplotlib figsize.
            cmap: colormap for scatter intensity.
        Returns:
            (fig, ax)
        """
        import matplotlib.pyplot as plt

        # Render background
        img = self.env.render()
        if img is None:
            try:
                img = self.env.render()
            except TypeError:
                raise RuntimeError("Env render returned None and does not support mode='rgb_array'")
        img = np.asarray(img)

        coords = self.state_coords
        n_states = coords.shape[0]
        if counts is None:
            counts = np.ones(n_states, dtype=np.float32)
        counts = np.asarray(counts, dtype=np.float32)
        counts = counts / (counts.max() + 1e-8)

        h, w = img.shape[:2]
        # Infer grid cell size
        if self.grid_size is None:
            raise ValueError("grid_size is unknown; cannot map coordinates to pixels")
        cell_h = h / self.grid_size
        cell_w = w / self.grid_size

        xs = coords[:, 1] * cell_w + 0.5 * cell_w
        ys = coords[:, 0] * cell_h + 0.5 * cell_h

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)
        sc = ax.scatter(xs, ys, c=counts, cmap=cmap, s=200 * counts + 10, edgecolors="k")
        ax.set_axis_off()
        ax.set_title("State distribution over MiniGrid")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="normalized count")
        return fig, ax
