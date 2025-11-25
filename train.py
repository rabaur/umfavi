import argparse
import torch
import wandb
import matplotlib.pyplot as plt
from umfavi.envs.get_env import get_env
from umfavi.data.get_dataset import get_dataset
from umfavi.learned_reward_wrapper import LearnedRewardWrapper
from umfavi.evaluation.epic import epic_distance
from umfavi.evaluation.regret import evaluate_regret_tabular, evaluate_regret_non_tabular
from umfavi.evaluation.val_loss import compute_eval_loss
from umfavi.multi_fb_model import MultiFeedbackTypeModel
from umfavi.utils.policies import (
    create_expert_policy,
    load_or_train_dqn,
    DQNQValueModel,
    TabularQValueModel
)
from umfavi.utils.gym import get_obs_dim, get_act_dim
from umfavi.encoder.reward_encoder import RewardEncoder
from umfavi.encoder.feature_modules import MLPFeatureModule
from umfavi.loglikelihoods.get_nll import get_nll
from umfavi.utils.reproducibility import seed_everything
from umfavi.utils.torch import get_device
from umfavi.utils.logging import update_epoch_log_dict, console_log_batch_metrics, console_log_eval_metrics
from umfavi.losses import elbo_loss
from umfavi.visualization.grid_visualizer import visualize_rewards as visualize_grid_rewards
from umfavi.visualization.cartpole_visualizer import visualize_cartpole_rewards
from umfavi.envs.grid_env.env import GridEnv
from umfavi.utils.feature_transforms import to_one_hot
from umfavi.types import FeedbackType
from umfavi.envs.env_types import TabularEnv
from umfavi.utils.feature_transforms import apply_transform
import numpy as np
from umfavi.utils.training import get_batch

def main(args):

    # Reproducibility
    seed_everything(args.seed)
    
    # Initialize wandb
    if args.log_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            tags=args.wandb_tags.split(",") if args.wandb_tags else None,
        )

    # Create environment
    if args.env_name == "GridWorld":
        env = GridEnv(
            **vars(args),
        )
    elif "MiniGrid" in args.env_name:
        import gymnasium as gym
        import minigrid # noqa: F401
        from umfavi.envs.minigrid.minigrid_wrapper import MiniGridWrapper

        gym_env = gym.make(args.env_name)
        env = MiniGridWrapper(gym_env)
    else:
        raise ValueError(f"Unsupported environment name: {args.env_name}")
    
    env = get_env(**vars(args))

    device = get_device()

    # Register feedback types and their sample counts
    feedback_config = {
        FeedbackType.PREFERENCE: args.num_pref_samples,
        FeedbackType.DEMONSTRATION: args.num_demo_samples,
    }
    
    # Filter out feedback types with 0 samples
    active_feedback_types = {k: v for k, v in feedback_config.items() if v > 0}
    
    if not active_feedback_types:
        raise ValueError("At least one feedback type must have samples > 0")
    
    # Create Q-value model (shared across all policies)
    is_tabular = isinstance(env, TabularEnv)
    if is_tabular:
        q_model = TabularQValueModel(env, gamma=args.gamma)
    else:
        dqn_model = load_or_train_dqn(env, gamma=args.gamma, train_if_missing=True)
        q_model = DQNQValueModel(dqn_model)
    
    # Create policies (all sharing the same Q-value model)
    policies = {}
    policies[FeedbackType.PREFERENCE] = create_expert_policy(
        env=env, 
        rationality=args.pref_trajectory_rationality, 
        q_model=q_model
    )
    policies[FeedbackType.DEMONSTRATION] = create_expert_policy(
        env=env, 
        rationality=args.demo_rationality, 
        q_model=q_model
    )
    
    # Define action-transform
    act_transform = None
    if args.act_transform:
        if args.act_transform == "one_hot":
            act_transform = lambda x: to_one_hot(x, env.action_space.n)
        else:
            raise NotImplementedError(f"Invalid action transform: {args.act_transform}")
    
    # Define observation-transform
    obs_transform = None
    if args.obs_transform:
        if args.obs_transform == "one_hot":
            obs_transform = lambda x: to_one_hot(x, env.observation_space.n)
        else:
            raise NotImplementedError(f"Invalid observation transform: {args.obs_transform}")

    # Dimensionality of the observation and action-space
    obs_dim = get_obs_dim(env, obs_transform)
    act_dim = get_act_dim(env, act_transform)

    # Create datasets and dataloaders
    _, train_dataloaders = get_dataset(active_feedback_types, args, env, policies, device, obs_transform, act_transform)
    _, val_dataloaders = get_dataset(active_feedback_types, args, env, policies, device, obs_transform, act_transform)


    feature_module = MLPFeatureModule(
        obs_dim,
        act_dim,
        args.encoder_hidden_sizes,
        reward_domain=args.reward_domain
    )
    reward_encoder = RewardEncoder(feature_module)

    # Create decoders only for active feedback types
    decoders = {}

    Q_value_model = MLPFeatureModule(
        state_dim=obs_dim,
        action_dim=act_dim,  # Not used since reward_domain='s'
        hidden_sizes=args.q_value_hidden_sizes + [act_dim],
        reward_domain='s',  # Q-value model only acts on state features
        activate_last_layer=False  # Q-values are in R
    )
    
    decoders = {fb_type: get_nll(fb_type) for fb_type in active_feedback_types}
    
    # Create multi-feedback model
    fb_model = MultiFeedbackTypeModel(
        encoder=reward_encoder,
        Q_value_model=Q_value_model,
        decoders=decoders
    )
    fb_model.to(device)
    
    # Watch model with wandb (log gradients and parameters)
    if args.log_wandb:
        wandb.watch(fb_model, log="all", log_freq=100)

    optimizer = torch.optim.Adam(fb_model.parameters(), lr=args.lr)

    # Calculate number of batches per epoch (use the max across all dataloaders)
    dataloader_lengths = {fb_type: len(dl) for fb_type, dl in train_dataloaders.items()}
    
    print(f"Training info:")
    for fb_type, length in dataloader_lengths.items():
        print(f"  {fb_type.value}: {length} batches per epoch")
    print(f"  Batches processed per update: {len(train_dataloaders)}")

    # Initialize dataloader iterators for each feedback type
    dloader_iters = {k: iter(train_dataloaders[k]) for k in active_feedback_types}
    
    # Steps per epoch is determined by the feedback type with the most batches
    steps_per_epoch = max(len(dl) for dl in train_dataloaders.values())

    print(f"\n{'='*60}")
    print(f"Starting training for {args.num_epochs} epochs")
    print(f"{'='*60}\n")

    regret_reference_policy = create_expert_policy(env, rationality=float("inf"), q_model=q_model)

    # If the environment is tabular, pre-compute all state-action features
    all_obs_features = None
    all_act_features = None
    if is_tabular:
        all_obs_features = torch.tensor(apply_transform(obs_transform, np.arange(env.observation_space.n)[:, None])).to(device)
        all_act_features = torch.tensor(apply_transform(act_transform, np.arange(env.action_space.n)[:, None])).to(device)

    for epoch in range(args.num_epochs):
        
        # Training
        fb_model.train()
        
        # Track losses per feedback type
        epoch_log_dict = {}
        
        print(f"Epoch {epoch}/{args.num_epochs-1} - Training...")
        
        for step in range(steps_per_epoch):

            # Calculate global step for wandb logging
            global_step = epoch * steps_per_epoch + step
            relative_step = global_step / steps_per_epoch

            # Process all feedback types in this step
            total_loss = 0.0
            aggregated_loss_dict = {}
            
            for fb_type in active_feedback_types:
    
                batch = get_batch(dloader_iters, train_dataloaders, fb_type)
                
                # Forward pass
                loss_dict = fb_model(**batch)
                
                # Compute ELBO loss for this feedback type
                elbo = elbo_loss(
                    loss_dict["negative_log_likelihood"], 
                    loss_dict["kl_divergence"], 
                    kl_weight=args.kl_weight
                )
                
                # Add regularization
                total_loss += elbo + args.td_error_weight * loss_dict["td_error"]
                
                # Aggregate loss dict for logging
                for key, value in loss_dict.items():
                    if key not in aggregated_loss_dict:
                        aggregated_loss_dict[key] = 0.0
                    aggregated_loss_dict[key] += value.item() if torch.is_tensor(value) else value
                    
                    # Also track per-feedback-type metrics
                    fb_key = f"{key}_{fb_type.value}"
                    if fb_key not in aggregated_loss_dict:
                        aggregated_loss_dict[fb_key] = 0.0
                    aggregated_loss_dict[fb_key] += value.item() if torch.is_tensor(value) else value
                
                # Update epoch log dict for this feedback type
                epoch_log_dict = update_epoch_log_dict(epoch_log_dict, loss_dict, fb_type)
            
            # Backpropagate combined loss
            total_loss.backward()
            
            # Single optimizer step after processing all feedback types
            optimizer.step()
            optimizer.zero_grad()
            
            # Log to wandb and console every N steps
            if (global_step) % args.log_every_n_steps == 0:
                # Console logging
                console_log_batch_metrics(aggregated_loss_dict, step, steps_per_epoch, total_loss)
                
                if args.log_wandb:
                    wandb_log_dict = {f"batch/{key}": value for key, value in aggregated_loss_dict.items()}
                    wandb_log_dict["batch/total_loss"] = total_loss.item()
                    wandb_log_dict["relative_step"] = relative_step
                    wandb.log(wandb_log_dict, step=global_step)

        # -----------------------------------------------
        # Evaluation
        # -----------------------------------------------
        if epoch % args.eval_every_n_epochs == 0:
            
            print(f"  Evaluating...")
            fb_model.eval()
            eval_metrics = {}

            # Compute epic distance
            # epic_dist = epic_distance(env.R, to_numpy(R_est), gamma=args.gamma)
            # eval_metrics["eval/epic_distance"] = epic_dist
            
            # Compute expected regret
            if is_tabular:
                regret = evaluate_regret_tabular(env, reward_encoder, all_obs_features, all_act_features, gamma=args.gamma, num_samples=1000)
            else:
                wrapped_env = LearnedRewardWrapper(env, fb_model.encoder, act_transform, obs_transform)
                regret, mean_rew = evaluate_regret_non_tabular(regret_reference_policy, env, wrapped_env, gamma=args.gamma)
                eval_metrics["eval/mean_rew"] = mean_rew
            eval_metrics["eval/regret"] = regret

            # Compute evaluation losses
            with torch.no_grad():
                eval_metrics |= compute_eval_loss(val_dataloaders, active_feedback_types, fb_model)

            # Log to wandb
            if args.log_wandb:
                eval_metrics |= {"epoch": epoch, "relative_step": relative_step}
                wandb.log(eval_metrics, step=global_step)

            # Console logging
            console_log_eval_metrics(eval_metrics)
            
            # Set model back to training mode
            fb_model.train()
        
        # Visualization 
        if epoch % args.vis_every_n_epochs == 0:
            print(f"  Generating visualization...")
            fb_model.eval()
            with torch.no_grad():
                # Get first available dataloader for trajectory visualization
                sample_dataloader = next(iter(train_dataloaders.values()))
                
                # Use appropriate visualizer based on environment type
                if isinstance(env, GridEnv):
                    fig = visualize_grid_rewards(env, fb_model.encoder, sample_dataloader, all_obs_features, all_act_features)
                else:
                    # Assume gymnasium environment (CartPole, etc.)
                    # Get number of actions for the environment
                    from gymnasium import spaces
                    if isinstance(env.action_space, spaces.Discrete):
                        num_actions = env.action_space.n
                    else:
                        num_actions = env.action_space.shape[0]
                    
                    fig = visualize_cartpole_rewards(
                        fb_model, device, sample_dataloader,
                        num_actions=num_actions
                    )
                
                # Log to wandb
                if args.log_wandb:
                    wandb.log({
                        "visualizations/rewards": wandb.Image(fig),
                        "epoch": epoch,
                    }, step=global_step)
                
                # Close the figure to free memory
                plt.close(fig)
            fb_model.train()
    
    # Finish wandb run
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}\n")
    
    if args.log_wandb:
        wandb.finish()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Reproducibility
    parser.add_argument("--seed", type=int, default=0, help="Global seed")

    # Environment
    parser.add_argument("--env_name", type=str, default="GridWorld", help="Name of the environment")
    
    # Dataset parameters
    parser.add_argument("--num_pref_samples", type=int, default=0, help="Number of preference samples (0 to disable)")
    parser.add_argument("--num_demo_samples", type=int, default=2, help="Number of demonstration samples (0 to disable)")
    parser.add_argument("--reward_domain", type=str, default="sa", help="Either state-only ('s'), state-action ('sa'), state-action-next-state ('sas')")
    parser.add_argument("--num_steps", type=int, default=19, help="Length of each trajectory")
    parser.add_argument("--td_error_weight", type=float, default=1.0, help="Weight for TD-error constraint in demonstrations")
    
    # Policy parameters
    parser.add_argument("--pref_rationality", type=float, default=1.0, help="Rationality for Bradley-Terry model")
    parser.add_argument("--pref_trajectory_rationality", type=float, default=10.0, help="Rationality of the expert policy generating the comparison trajectories")
    parser.add_argument("--demo_rationality", type=float, default=20.0, help="Rationality for expert policy")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--eval_every_n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--kl_weight", type=float, default=1.0, help="KL weight - use kl_restart_period for annealing")
    parser.add_argument("--vis_every_n_epochs", type=int, default=25, help="Frequency of visualizations (epochs)")
    parser.add_argument("--encoder_hidden_sizes", type=int, nargs="+", default=[64, 64], help="Hidden sizes for encoder MLP")
    parser.add_argument("--q_value_hidden_sizes", type=int, nargs="+", default=[64, 64], help="Hidden sizes for Q-value MLP")
    
    # Environment parameters
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--env_name", type=str, default="grid_sparse")
    parser.add_argument("--p_rand", type=float, default=0.0, help="Randomness in transitions (0 for deterministic)")
    parser.add_argument("--obs_transform", choices=["one_hot", "continuous_coordinate", "dct", None], default=None, help="Apply a transform to the observation space")
    parser.add_argument("--act_transform", choices=["one_hot", None], default=None, help="Apply a transform to the action space")
    parser.add_argument("--n_dct_basis_fns", type=int, default=8, help="Number of DCT basis functions (only for grid environment)")
        
    # Wandb parameters
    parser.add_argument("--log_wandb", action="store_true", help="Log to weights and biases")
    parser.add_argument("--log_every_n_steps", type=int, default=10, help="Log every n steps")
    parser.add_argument("--wandb_project", type=str, default="var-rew-learning", help="Wandb project name")
    parser.add_argument("--wandb_tags", type=str, default="", help="Comma-separated wandb tags")
    parser.add_argument("--wandb_watch", action="store_true", help="Enable wandb model watching (logs gradients and parameters)")
    
    args = parser.parse_args()
    main(args)
