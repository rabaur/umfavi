import argparse
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Any
from umfavi.envs.grid_env.env import GridEnv
from umfavi.data.preference_dataset import PreferenceDataset
from umfavi.data.demonstration_dataset import DemonstrationDataset
from umfavi.metrics.epic import evaluate_epic_distance
from umfavi.metrics.regret import evaluate_regret
from umfavi.multi_fb_model import MultiFeedbackTypeModel
from umfavi.utils.policies import ExpertPolicy
from umfavi.encoder.reward_encoder import RewardEncoder
from umfavi.encoder.features import MLPFeatureModule
from umfavi.loglikelihoods.preference import PreferenceDecoder
from umfavi.loglikelihoods.demonstrations import DemonstrationsDecoder
from umfavi.utils.reproducibility import seed_everything
from umfavi.utils.torch import get_device, to_numpy
from umfavi.losses import elbo_loss
from umfavi.grid_env_visualizer import visualize_rewards
from umfavi.feedback_types import FeedbackType


def dataset_factory(active_feedback_types, args, env, policies, device):
    datasets = {}
    dataloaders = {}
    if FeedbackType.Preference in active_feedback_types:
        pref_dataset = PreferenceDataset(
            n_samples=args.num_pref_samples,
            n_steps=args.num_steps,
            env=env,
            policy=policies[FeedbackType.Preference],
            device=device,
            rationality=args.pref_rationality,
            gamma=args.gamma,
        )
        datasets[FeedbackType.Preference] = pref_dataset
        dataloaders[FeedbackType.Preference] = DataLoader(pref_dataset, batch_size=args.batch_size, shuffle=True)
        print(f"Created preference dataset with {len(pref_dataset)} samples")

    if FeedbackType.Demonstration in active_feedback_types:
        demo_dataset = DemonstrationDataset(
            n_samples=args.num_demo_samples,
            n_steps=args.num_steps,
            env=env,
            policy=policies[FeedbackType.Demonstration],
            device=device,
            rationality=args.expert_rationality,
            gamma=args.gamma,
            td_error_weight=args.td_error_weight,
        )
        datasets[FeedbackType.Demonstration] = demo_dataset
        dataloaders[FeedbackType.Demonstration] = DataLoader(demo_dataset, batch_size=args.batch_size, shuffle=True)
        print(f"Created demonstration dataset with {len(demo_dataset)} samples")


def get_batch(dloader_iters, dataloaders, fb_type):
    try:
        batch = next(dloader_iters[fb_type])
    except StopIteration:
        # Restart iterator if we've exhausted this dataloader
        dloader_iters[fb_type] = iter(dataloaders[fb_type])
        batch = next(dloader_iters[fb_type])
    return batch


def compute_eval_loss(val_dataloaders, active_feedback_types, multi_fb_model: MultiFeedbackTypeModel):
    assert not multi_fb_model.training, "Model is not in evaluation mode"
    dloader_iters = {fb_type: iter(val_dataloaders[fb_type]) for fb_type in active_feedback_types}
    eval_loss_dict
    for fb_type in active_feedback_types:
        batch = get_batch(dloader_iters, val_dataloaders, fb_type)
        loss



def update_epoch_log_dict(epoch_log_dict: dict[str, tuple[int, Any]], metrics_dict: dict[str, Any], fb_type: str):
    epoch_log_dict = dict(epoch_log_dict)
    for key, value in metrics_dict.items():
        if key in epoch_log_dict:
            prev_count, prev_val = epoch_log_dict[key]
            updated_vals = (prev_count + 1, prev_val + value)
            epoch_log_dict[key] = updated_vals
            epoch_log_dict[f"{key}_{fb_type}"] = updated_vals
        else:
            initial_value = (1, value)
            epoch_log_dict[key] = initial_value
            epoch_log_dict[f"{key}_{fb_type}"] = initial_value
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

    env = GridEnv(
        **vars(args),
    )

    device = get_device()
        
    # Get all state-action features for evaluation
    n_states = env.S.shape[0]
    n_actions = env.action_space.n
    state_feats_flat = torch.tensor(env.S).to(device=device)
    action_feats = torch.tensor(env.A).to(device=device)

    # Register feedback types and their sample counts
    feedback_config = {
        FeedbackType.Preference: args.num_pref_samples,
        FeedbackType.Demonstration: args.num_demo_samples,
    }
    
    # Filter out feedback types with 0 samples
    active_feedback_types = {k: v for k, v in feedback_config.items() if v > 0}
    
    if not active_feedback_types:
        raise ValueError("At least one feedback type must have samples > 0")
    
    # Create policies
    policies = {}
    policies[FeedbackType.Preference] = ExpertPolicy(env=env, rationality=args.pref_trajectory_rationality, gamma=args.gamma)
    policies[FeedbackType.Demonstration] = ExpertPolicy(env=env, rationality=args.demo_rationality, gamma=args.gamma)
    
    # Create datasets and dataloaders
    _, train_dataloaders = dataset_factory(active_feedback_types, args, env, policies)
    _, eval_dataloaders = dataset_factory(active_feedback_types, args, env, policies)

    # Create feature module and encoder
    obs_dim = env.observation_space["state_features"].shape[0]
    act_dim = env.action_space.n
    learn_embedding = args.state_feature_type == "embedding"

    feature_module = MLPFeatureModule(
        obs_dim,
        act_dim,
        args.encoder_hidden_sizes,
        reward_domain=args.reward_domain,
        learn_embedding=learn_embedding,
        state_embedding_size=args.state_embedding_size,
        action_embedding_size=args.action_embedding_size,
        n_actions=n_actions,
        n_states=n_states
    )
    reward_encoder = RewardEncoder(feature_module)

    # Create decoders only for active feedback types
    decoders = {}

    Q_value_model = MLPFeatureModule(
        state_dim=obs_dim,
        action_dim=act_dim,  # Not used since reward_domain='s'
        learn_embedding=learn_embedding,
        hidden_sizes=args.q_value_hidden_sizes + [act_dim],
        state_embedding_size=args.state_embedding_size,
        action_embedding_size=args.action_embedding_size,
        n_states=n_states,
        n_actions=n_actions,
        reward_domain='s',  # Q-value model only acts on state features
        activate_last_layer=False  # Q-values are in R
    )
    
    if FeedbackType.Preference in active_feedback_types:
        preference_decoder = PreferenceDecoder()
        decoders[FeedbackType.Preference] = preference_decoder
    
    if FeedbackType.Demonstration in active_feedback_types:
        # Q-value model is just MLPFeatureModule with reward_domain='s' and last layer = n_actions
        demonstration_decoder = DemonstrationsDecoder()
        decoders[FeedbackType.Demonstration] = demonstration_decoder
    
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

    for epoch in range(args.num_epochs):
        
        # Training
        fb_model.train()
        
        # Track losses per feedback type
        epoch_log_dict = {}
        
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
                fb_loss = elbo_loss(
                    loss_dict["negative_log_likelihood"], 
                    loss_dict["kl_divergence"], 
                    kl_weight=args.kl_weight
                )
                
                # Add regularization
                total_loss = fb_loss + args.td_error_weight * loss_dict["td_error"]
                
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
            
            # Log to wandb every N steps
            if (global_step) % args.log_every_n_steps == 0:
                if args.log_wandb:
                    wandb_log_dict = {f"batch/{key}": value for key, value in aggregated_loss_dict.items()}
                    wandb_log_dict["batch/total_loss"] = total_loss.item()
                    wandb_log_dict["global_step"] = global_step
                    wandb.log(wandb_log_dict, step=relative_step)

        # -----------------------------------------------
        # Evaluation
        # -----------------------------------------------
        if epoch % args.eval_every_n_epochs == 0:
            
            fb_model.eval()
            eval_metrics = {}
            with torch.no_grad():

                # Compute the mean estimated reward per state-action pair
                R_est = np.empty((n_states, n_actions))
                for a in range(n_actions):
                    a_feats_tiled = torch.tile(action_feats[a], (n_states, 1))
                    R_mean_a, _ = fb_model.encoder(state_feats_flat, a_feats_tiled, state_feats_flat)
                    R_est[:, a] = to_numpy(R_mean_a).squeeze()

                # Compute epic distance
                epic_dist = evaluate_epic_distance(
                    R_true=env.R,
                    R_est=R_est,
                    gamma=args.gamma)
                eval_metrics["epic_distance"] = epic_dist
                
                # Compute expected regret
                regret = evaluate_regret(R_est=R_est, R_true=env.R, P=env.P, gamma=args.gamma)
                eval_metrics["regret"] = regret

            # Log to wandb
            if args.log_wandb:
                wandb.log({
                    "eval/epic_distance": epic_dist,
                    "eval/expected_regret": regret,
                    "epoch": epoch,
                    "global_step": global_step,
                }, step=relative_step)

                epoch_log_dict_wandb = epoch_log_dict_to_wandb(epoch_log_dict)
                wandb.log(epoch_log_dict_wandb, step=global_step)
        
            # Evaluation
            if (epoch + 1) % args.vis_freq == 0:
                with torch.no_grad():
                    fig = visualize_rewards(env, fb_model, device, train_dataloaders[fb_type])
                    # Log to wandb
                    if args.log_wandb:
                        wandb.log({
                            "visualizations/rewards": wandb.Image(fig),
                            "epoch": epoch,
                        }, step=global_step)
                    
                    # Close the figure to free memory
                    plt.close(fig)
            
            # Set model back to training mode
            fb_model.train()
    
    # Finish wandb run
    wandb.finish()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Reproducibility
    parser.add_argument("--seed", type=int, default=0, help="Global seed")
    
    # Dataset parameters
    parser.add_argument("--num_pref_samples", type=int, default=69, help="Number of preference samples (0 to disable)")
    parser.add_argument("--num_demo_samples", type=int, default=0, help="Number of demonstration samples (0 to disable)")
    parser.add_argument("--reward_domain", type=str, default="s", help="Either state-only ('s'), state-action ('sa'), state-action-next-state ('sas')")
    parser.add_argument("--num_steps", type=int, default=32, help="Length of each trajectory")
    parser.add_argument("--td_error_weight", type=float, default=1.0, help="Weight for TD-error constraint in demonstrations")
    
    # Policy parameters
    parser.add_argument("--pref_rationality", type=float, default=1.0, help="Rationality for Bradley-Terry model")
    parser.add_argument("--pref_trajectory_rationality", type=float, default=0.1, help="Rationality for the expert policy generating the comparison trajectories")
    parser.add_argument("--demo_rationality", type=float, default=5.0, help="Rationality for expert policy")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--eval_every_n_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--kl_weight", type=float, default=1.0, help="KL weight - use kl_restart_period for annealing")
    parser.add_argument("--vis_freq", type=int, default=10, help="Frequency of visualizations (epochs)")
    parser.add_argument("--encoder_hidden_sizes", type=int, nargs="+", default=[64, 64], help="Hidden sizes for encoder MLP")
    parser.add_argument("--q_value_hidden_sizes", type=int, nargs="+", default=[64, 64], help="Hidden sizes for Q-value MLP")
    
    # Environment parameters
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--reward_type", type=str, default="sparse")
    parser.add_argument("--p_rand", type=float, default=0.0, help="Randomness in transitions (0 for deterministic)")
    parser.add_argument("--state_feature_type", type=str, default="one_hot", help="Type of state feature encoding (one_hot, continuous_coordinate, dct, embedding)")
    parser.add_argument("--n_dct_basis_fns", type=int, default=8, help="Number of DCT basis functions")
    parser.add_argument("--state_embedding_size", type=int, default=32, help="Only used if state_feature_type=='embedding'")
    parser.add_argument("--action_feature_type", type=str, default="one_hot", help="Type of action feature encoding (one_hot, embedding)")
    parser.add_argument("--action_embedding_size", type=int, default=8, help="Only used if state_feature_type=='embedding")
    
    # Visualization parameters
    parser.add_argument("--visualize_dataset", action="store_true", help="Visualize dataset state-action visitation before training")
    
    # Wandb parameters
    parser.add_argument("--log_wandb", action="store_true", help="Log to weights and biases")
    parser.add_argument("--log_every_n_steps", type=int, default=10, help="Log every n steps")
    parser.add_argument("--wandb_project", type=str, default="var-rew-learning", help="Wandb project name")
    parser.add_argument("--wandb_tags", type=str, default="", help="Comma-separated wandb tags")
    parser.add_argument("--wandb_watch", action="store_true", help="Enable wandb model watching (logs gradients and parameters)")
    
    args = parser.parse_args()
    main(args)