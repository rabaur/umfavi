import tempfile
from pathlib import Path
from typing import Any
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import yaml

from umfavi.true_reward_callback import TrueRewardCallback


def get_hyperparams(algo: str, env_name: str) -> dict[str, Any]:
    """
    Load hyperparameters for a given algorithm and environment from YAML files.
    
    Args:
        algo: Algorithm name (e.g., 'ppo', 'dqn', 'sac')
        env_name: Environment name (e.g., 'LunarLander-v3', 'CartPole-v1')
    
    Returns:
        Dictionary of hyperparameters suitable for the SB3 model constructor.
    """
    # Find the hyperparams directory relative to this file
    hyperparams_dir = Path(__file__).parent.parent.parent / "hyperparams"
    hyperparams_file = hyperparams_dir / f"{algo.lower()}.yml"
    
    if not hyperparams_file.exists():
        raise FileNotFoundError(f"Hyperparameters file not found: {hyperparams_file}")
    
    with open(hyperparams_file, "r") as f:
        all_hyperparams = yaml.safe_load(f)
    
    if env_name not in all_hyperparams:
        raise KeyError(f"Environment '{env_name}' not found in {hyperparams_file}")
    
    # Evaluate policy_kwargs string if present
    if "policy_kwargs" in all_hyperparams[env_name] and isinstance(all_hyperparams[env_name]["policy_kwargs"], str):
        all_hyperparams[env_name]["policy_kwargs"] = eval(all_hyperparams[env_name]["policy_kwargs"])
    
    hyperparams = all_hyperparams[env_name].copy()
    
    return hyperparams


def train_dqn(wrapped_env, reference_env_name, eval_freq: int = 10000, n_eval_episodes: int = 5):
    """
    Train a DQN model on the wrapped environment using the appropriate hyperparams for the reference environment.
    Returns the best model based on evaluation performance.
    
    Args:
        wrapped_env: The environment to train on (with learned reward).
        reference_env_name: Name of the original environment (for loading hyperparams).
        eval_freq: How often to evaluate the model (in timesteps).
        n_eval_episodes: Number of episodes to run for each evaluation.
    """
    hyperparams = get_hyperparams("dqn", reference_env_name)
    n_timesteps = hyperparams.pop("n_timesteps")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        true_reward_cb = TrueRewardCallback(window_size=100)
        eval_cb = EvalCallback(
            wrapped_env,
            best_model_save_path=tmpdir,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            verbose=0
        )
        callback = CallbackList([true_reward_cb, eval_cb])
        
        dqn_model = sb3.DQN(env=wrapped_env, **hyperparams, verbose=1)
        dqn_model.learn(total_timesteps=n_timesteps, callback=callback, progress_bar=True)
        
        # Load and return the best model
        best_model_path = Path(tmpdir) / "best_model.zip"
        if best_model_path.exists():
            return sb3.DQN.load(best_model_path, env=wrapped_env)
        
        # Fallback to final model if no best model was saved
        return dqn_model