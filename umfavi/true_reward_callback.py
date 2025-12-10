import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback

class TrueRewardCallback(BaseCallback):
    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self.ep_true_rewards = np.zeros(n_envs, dtype=np.float32)
        self.ep_true_rewards_history = deque(maxlen=self.window_size)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]      # list of info dicts, one per env
        dones = self.locals["dones"]      # np.array shape (n_envs,)

        for i, info in enumerate(infos):
            # Accumulate true reward if present
            if "true_reward" in info:
                self.ep_true_rewards[i] += info["true_reward"]

            # If episode finished, store and reset
            if dones[i]:
                ep_tr = self.ep_true_rewards[i]
                self.ep_true_rewards_history.append(ep_tr)
                self.ep_true_rewards[i] = 0.0

        # Log moving average over last N episodes if we have any
        if len(self.ep_true_rewards_history) > 0:
            mean_true_rew = np.mean(self.ep_true_rewards_history)
            self.logger.record("rollout/ep_true_rew_mean", mean_true_rew)

        return True
