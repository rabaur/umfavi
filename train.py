import argparse
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from torch.utils.data import DataLoader
from umfavi.envs.get_env import get_env
from umfavi.data.get_dataset import get_dataset
from umfavi.metrics.epic import epic_distance
from umfavi.metrics.regret import evaluate_regret
from umfavi.multi_fb_model import MultiFeedbackTypeModel
from umfavi.utils.policies import create_expert_policy
from umfavi.utils.gym import get_obs_dim, get_act_dim
from umfavi.encoder.reward_encoder import RewardEncoder
from umfavi.encoder.feature_modules import MLPFeatureModule
from umfavi.loglikelihoods.preference import PreferenceDecoder
from umfavi.loglikelihoods.demonstrations import DemonstrationsDecoder
from umfavi.utils.reproducibility import seed_everything
from umfavi.utils.torch import get_device, to_numpy
from umfavi.losses import elbo_loss
from umfavi.visualization.grid_visualizer import visualize_rewards as visualize_grid_rewards
from umfavi.visualization.cartpole_visualizer import visualize_cartpole_rewards
from umfavi.envs.grid_env.env import GridEnv
from umfavi.utils.feature_transforms import to_one_hot
from umfavi.types import FeedbackType


def get_batch(dloader_iters, dataloaders, fb_type):
    try:
        batch = next(dloader_iters[fb_type])
    except StopIteration:
        # Restart iterator if we've exhausted this dataloader
        dloader_iters[fb_type] = iter(dataloaders[fb_type])
        batch = next(dloader_iters[fb_type])
    return batch


def compute_eval_loss(val_dataloaders: dict[FeedbackType, DataLoader], active_feedback_types: list[FeedbackType], multi_fb_model: MultiFeedbackTypeModel):
    assert not multi_fb_model.training, "Model is not in evaluation mode"
    dloader_iters = {fb_type: iter(val_dataloaders[fb_type]) for fb_type in active_feedback_types}
    eval_loss_dict = {}
    for fb_type in active_feedback_types:
        batch = get_batch(dloader_iters, val_dataloaders, fb_type)
        loss_dict = multi_fb_model(**batch)

        # Log total loss metrics
        for k, v in loss_dict.items():
            if k not in eval_loss_dict:
                eval_loss_dict[k] = (0, 0.0)
            count, agg_val = eval_loss_dict[k]
            eval_loss_dict[k] = (count + 1, agg_val + v)
        
        # Log feedback-specific metrics
        for k, v in loss_dict.items():
            key = f"{k}_{fb_type.value}"
            if key not in eval_loss_dict:
                eval_loss_dict[key] = (0, 0.0)
            count, agg_val = eval_loss_dict[key]
            eval_loss_dict[key] = (count + 1, agg_val + v)
    
    # Average loss metrics
    final_dict = {}
    for k, (count, agg_val) in eval_loss_dict.items():
        if count > 0:
            final_dict[f"eval/{k}"] = agg_val / count
    return final_dict
        

def update_epoch_log_dict(epoch_log_dict: dict[str, tuple[int, Any]], metrics_dict: dict[str, Any], fb_type: FeedbackType):
    epoch_log_dict = dict(epoch_log_dict)
    for key, value in metrics_dict.items():
        if key in epoch_log_dict:
            prev_count, prev_val = epoch_log_dict[key]
            updated_vals = (prev_count + 1, prev_val + value)
            epoch_log_dict[key] = updated_vals
            epoch_log_dict[f"{key}_{fb_type.value}"] = updated_vals
        else:
            initial_value = (1, value)
            epoch_log_dict[key] = initial_value
            epoch_log_dict[f"{key}_{fb_type.value}"] = initial_value
    return epoch_log_dict


def epoch_log_dict_to_wandb(epoch_log_dict: dict[str, tuple[int, Any]]):
    wandb_dict = {}
    for key, (count, agg_value) in epoch_log_dict.items():
        if count > 0:
            wandb_dict[f"epoch/{key}"] = agg_value / count
        else:
            wandb_dict[f"epoch/{key}"] = np.nan
    return wandb_dict


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
    
    # Create policies
    policies = {}
    policies[FeedbackType.PREFERENCE] = create_expert_policy(env=env, rationality=args.pref_trajectory_rationality, gamma=args.gamma)
    policies[FeedbackType.DEMONSTRATION] = create_expert_policy(env=env, rationality=args.demo_rationality, gamma=args.gamma)
    
    # Define action-transform
    action_transform = None
    if args.action_feature_type == "one_hot":
        action_transform = lambda x: to_one_hot(x, env.action_space.n)
    else:
        raise NotImplementedError(f"Invalid action feature type: {args.action_feature_type}")
    obs_transform = None


    # Dimensionality of the observation and action-space
    obs_dim = get_obs_dim(env, obs_transform)
    act_dim = get_act_dim(env, action_transform)

    # Create datasets and dataloaders
    _, train_dataloaders = get_dataset(active_feedback_types, args, env, policies, device, obs_transform, action_transform)
    _, val_dataloaders = get_dataset(active_feedback_types, args, env, policies, device, obs_transform, action_transform)


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
    
    if FeedbackType.PREFERENCE in active_feedback_types:
        preference_decoder = PreferenceDecoder()
        decoders[FeedbackType.PREFERENCE] = preference_decoder
    
    if FeedbackType.DEMONSTRATION in active_feedback_types:
        # Q-value model is just MLPFeatureModule with reward_domain='s' and last layer = n_actions
        demonstration_decoder = DemonstrationsDecoder()
        decoders[FeedbackType.DEMONSTRATION] = demonstration_decoder
    
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
        print(f"  {fb_type}: {length} batches per epoch")
    print(f"  Batches processed per update: {len(train_dataloaders)}")

    # Initialize dataloader iterators for each feedback type
    dloader_iters = {k: iter(train_dataloaders[k]) for k in active_feedback_types}
    
    # Steps per epoch is determined by the feedback type with the most batches
    steps_per_epoch = max(len(dl) for dl in train_dataloaders.values())

    print(f"\n{'='*60}")
    print(f"Starting training for {args.num_epochs} epochs")
    print(f"{'='*60}\n")

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
                    fb_key = f"{key}_{fb_type}"
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
                nll = aggregated_loss_dict.get("negative_log_likelihood", 0.0)
                kl = aggregated_loss_dict.get("kl_divergence", 0.0)
                td = aggregated_loss_dict.get("td_error", 0.0)
                print(f"  Step {step}/{steps_per_epoch-1} | Loss: {total_loss.item():.4f} | NLL: {nll:.4f} | KL: {kl:.4f} | TD: {td:.4f}")
                
                if args.log_wandb:
                    wandb_log_dict = {f"batch/{key}": value for key, value in aggregated_loss_dict.items()}
                    wandb_log_dict["batch/total_loss"] = total_loss.item()
                    wandb_log_dict["relative_step"] = relative_step
                    wandb.log(wandb_log_dict, step=global_step)

        # -----------------------------------------------
        # Evaluation
        # -----------------------------------------------
        if False and epoch % args.eval_every_n_epochs == 0:
            
            print(f"  Evaluating...")
            fb_model.eval()
            eval_metrics = {}
            with torch.no_grad():

                # Compute the mean estimated reward per state-action pair
                if args.reward_domain == "s":
                    R_est_mean, _ = fb_model.encoder(state_feats_flat, None, None)
                    R_est_mean = R_est_mean.squeeze()
                    R_est = torch.broadcast_to(R_est_mean[:, None, None], (n_states, n_actions, n_states))
                else:
                    raise NotImplementedError()

                # Compute epic distance
                epic_dist = epic_distance(env.R, to_numpy(R_est), gamma=args.gamma)
                eval_metrics["eval/epic_distance"] = epic_dist
                
                # Compute expected regret
                regret = evaluate_regret(R_est=to_numpy(R_est), R_true=env.R, P=env.P, gamma=args.gamma)
                eval_metrics["eval/regret"] = regret

                # Compute evaluation losses
                eval_metrics |= compute_eval_loss(val_dataloaders, active_feedback_types, fb_model)

            # Console logging of evaluation metrics
            print(f"  Evaluation Results:")
            if "eval/epic_distance" in eval_metrics:
                print(f"    EPIC Distance: {eval_metrics['eval/epic_distance']:.6f}")
            if "eval/regret" in eval_metrics:
                print(f"    Regret: {eval_metrics['eval/regret']:.6f}")
            if "eval/negative_log_likelihood" in eval_metrics:
                print(f"    Eval NLL: {eval_metrics['eval/negative_log_likelihood']:.4f}")
            if "eval/kl_divergence" in eval_metrics:
                print(f"    Eval KL: {eval_metrics['eval/kl_divergence']:.4f}")
            if "eval/td_error" in eval_metrics:
                print(f"    Eval TD Error: {eval_metrics['eval/td_error']:.4f}")
            print()

            # Log to wandb
            if args.log_wandb:
                eval_metrics |= {"epoch": epoch, "relative_step": relative_step}
                wandb.log(eval_metrics, step=global_step)
            
            # Set model back to training mode
            fb_model.train()
        
        # Visualization 
        if epoch % args.vis_freq == 0:
            print(f"  Generating visualization...")
            fb_model.eval()
            with torch.no_grad():
                # Get first available dataloader for trajectory visualization
                sample_dataloader = next(iter(train_dataloaders.values()))
                
                # Use appropriate visualizer based on environment type
                if isinstance(env, GridEnv):
                    fig = visualize_grid_rewards(env, fb_model, device, sample_dataloader)
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
    
    # Dataset parameters
    parser.add_argument("--num_pref_samples", type=int, default=256, help="Number of preference samples (0 to disable)")
    parser.add_argument("--num_demo_samples", type=int, default=0, help="Number of demonstration samples (0 to disable)")
    parser.add_argument("--reward_domain", type=str, default="sa", help="Either state-only ('s'), state-action ('sa'), state-action-next-state ('sas')")
    parser.add_argument("--num_steps", type=int, default=32, help="Length of each trajectory")
    parser.add_argument("--td_error_weight", type=float, default=1.0, help="Weight for TD-error constraint in demonstrations")
    
    # Policy parameters
    parser.add_argument("--pref_rationality", type=float, default=1.0, help="Rationality for Bradley-Terry model")
    parser.add_argument("--pref_trajectory_rationality", type=float, default=0.1, help="Rationality of the expert policy generating the comparison trajectories")
    parser.add_argument("--demo_rationality", type=float, default=5.0, help="Rationality for expert policy")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--eval_every_n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--kl_weight", type=float, default=1.0, help="KL weight - use kl_restart_period for annealing")
    parser.add_argument("--vis_freq", type=int, default=10, help="Frequency of visualizations (epochs)")
    parser.add_argument("--encoder_hidden_sizes", type=int, nargs="+", default=[64, 64], help="Hidden sizes for encoder MLP")
    parser.add_argument("--q_value_hidden_sizes", type=int, nargs="+", default=[64, 64], help="Hidden sizes for Q-value MLP")
    
    # Environment parameters
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--env_name", type=str, default="CartPole-v1")
    parser.add_argument("--p_rand", type=float, default=0.0, help="Randomness in transitions (0 for deterministic)")
    parser.add_argument("--state_feature_type", type=str, default="one_hot", help="Type of state feature encoding (one_hot, continuous_coordinate, dct)")
    parser.add_argument("--n_dct_basis_fns", type=int, default=8, help="Number of DCT basis functions")
    parser.add_argument("--action_feature_type", type=str, default="one_hot", help="Type of action feature encoding (one_hot)")
    
    # Visualization parameters
    parser.add_argument("--visualize_dataset", action="store_true", help="Visualize dataset state-action visitation before training")
    
    # Wandb parameters
    parser.add_argument("--log_wandb", action="store_true", help="Log to weights and biases")
    parser.add_argument("--log_every_n_steps", type=int, default=1, help="Log every n steps")
    parser.add_argument("--wandb_project", type=str, default="var-rew-learning", help="Wandb project name")
    parser.add_argument("--wandb_tags", type=str, default="", help="Comma-separated wandb tags")
    parser.add_argument("--wandb_watch", action="store_true", help="Enable wandb model watching (logs gradients and parameters)")
    
    args = parser.parse_args()
    main(args)