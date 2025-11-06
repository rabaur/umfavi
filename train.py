import argparse
import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from umfavi.envs.dct_grid_env import DCTGridEnv
from umfavi.data.preference_dataset import PreferenceDataset
from umfavi.data.demonstration_dataset import DemonstrationDataset
from umfavi.metrics.epic import evaluate_epic_distance
from umfavi.metrics.regret import evaluate_regret
from umfavi.multi_fb_model import MultiFeedbackTypeModel
from umfavi.utils.policies import ExpertPolicy
from umfavi.encoder.reward_encoder import RewardEncoder
from umfavi.encoder.features import MLPFeatureModule, QValueModel
from umfavi.loglikelihoods.preference import PreferenceDecoder
from umfavi.loglikelihoods.demonstrations import DemonstrationsDecoder
from umfavi.utils.torch import get_device, to_numpy
from umfavi.losses import elbo_loss
from umfavi.visualization.dct_grid_env_visualizer import (
    visualize_rewards,
    visualize_state_action_dist
)

def create_run_name(args):
    run_name_parts = []
    for key, value in args.__dict__.items():
        if isinstance(value, int):
            run_name_parts.append(f"{key}{value}")
        elif isinstance(value, float):
            run_name_parts.append(f"{key}{value:.2f}")
        else:
            run_name_parts.append(f"{key}{value}")
    return "-".join(run_name_parts)


# Define function that performs one-hot encoding of actions
def one_hot_encode_actions(actions, n_actions: int) -> torch.Tensor:
    """Convert integer action to one-hot encoded tensor."""
    if not isinstance(actions, torch.Tensor):
        actions = torch.tensor(actions, dtype=torch.long)
    return torch.nn.functional.one_hot(actions, num_classes=n_actions).float()

def main(args):
    
    args.wandb_run_name = create_run_name(args)
    
    # Initialize wandb
    if args.log_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            tags=args.wandb_tags.split(",") if args.wandb_tags else None,
        )
        print(f"Wandb run: {wandb.run.name if wandb.run else 'N/A'}")
        print(f"Wandb URL: {wandb.run.url if wandb.run else 'N/A'}\n")

    env = DCTGridEnv(
        grid_size=args.grid_size,
        n_dct_basis_fns=args.n_dct_basis_fns,
        reward_type=args.reward_type,
        p_rand=args.p_rand,
    )

    device = get_device()
    print(f"Using device: {device}")
        
    # Get all state-action features for evaluation
    n_states = env.S.shape[0]
    n_actions = env.action_space.n
    state_feats_flat = torch.tensor(env.S, dtype=torch.float32).to(device=device)
    action_feats = torch.zeros((n_actions, n_actions), dtype=torch.float32).to(device=device)
    for a in range(n_actions):
        action_feats[a, :] = one_hot_encode_actions(a, env.action_space.n)

    # Register feedback types and their sample counts
    feedback_config = {
        "preference": args.num_pref_samples,
        "demonstration": args.num_demo_samples,
    }
    
    # Filter out feedback types with 0 samples
    active_feedback_types = {k: v for k, v in feedback_config.items() if v > 0}
    
    if not active_feedback_types:
        raise ValueError("At least one feedback type must have samples > 0")
    
    print(f"\nActive feedback types: {list(active_feedback_types.keys())}")
    for fb_type, n_samples in active_feedback_types.items():
        print(f"  {fb_type}: {n_samples} samples")
    print()
    
    # Create policies (only if needed)
    policies_created = set()

    preference_policy = ExpertPolicy(env=env, rationality=0.5, gamma=args.gamma)
    policies_created.add("preference")
    demonstration_policy = ExpertPolicy(env=env, rationality=args.expert_rationality, gamma=args.gamma)
    policies_created.add("demonstration")
    
    # Create datasets and dataloaders
    datasets = {}
    dataloaders = {}
    
    if "preference" in active_feedback_types:
        pref_dataset = PreferenceDataset(
            n_samples=args.num_pref_samples,
            n_steps=args.num_steps,
            env=env,
            policy=preference_policy,
            device=device,
            act_transform=one_hot_encode_actions,
            rationality=args.pref_rationality,
        )
        datasets["preference"] = pref_dataset
        dataloaders["preference"] = DataLoader(pref_dataset, batch_size=args.batch_size, shuffle=True)
        print(f"Created preference dataset with {len(pref_dataset)} samples")
    
    if "demonstration" in active_feedback_types:
        demo_dataset = DemonstrationDataset(
            n_samples=args.num_demo_samples,
            n_steps=args.num_steps,
            env=env,
            policy=demonstration_policy,
            device=device,
            act_transform=one_hot_encode_actions,
            rationality=args.expert_rationality,
            gamma=args.gamma,
            td_error_weight=args.td_error_weight,
        )
        datasets["demonstration"] = demo_dataset
        dataloaders["demonstration"] = DataLoader(demo_dataset, batch_size=args.batch_size, shuffle=True)
        print(f"Created demonstration dataset with {len(demo_dataset)} samples")

    # Visualize dataset visitation if requested
    visualize_state_action_dist(env, dataloaders["demonstration"])

    # Create feature module and encoder
    obs_dim = env.observation_space["observation"].shape[0]
    act_dim = env.action_space.n
    feature_module = MLPFeatureModule(
        obs_dim, act_dim, [128, 128], reward_domain=args.reward_domain
    )
    reward_encoder = RewardEncoder(feature_module)

    # Create decoders only for active feedback types
    decoders = {}
    
    if "preference" in active_feedback_types:
        preference_decoder = PreferenceDecoder()
        decoders["preference"] = preference_decoder
        print("Created preference decoder")
    
    if "demonstration" in active_feedback_types:
        q_value_model = QValueModel(obs_dim, [128, 128, act_dim])
        demonstration_decoder = DemonstrationsDecoder(q_value_model)
        decoders["demonstration"] = demonstration_decoder
        print("Created demonstration decoder")
    
    # Create multi-feedback model
    fb_model = MultiFeedbackTypeModel(
        encoder=reward_encoder,
        decoders=decoders
    )
    fb_model.to(device)
    
    # Watch model with wandb (log gradients and parameters)
    if args.usewb:
        wandb.watch(fb_model, log="all", log_freq=100)

    optimizer = torch.optim.Adam(fb_model.parameters(), lr=args.lr)

    # Calculate number of batches per epoch (use the max across all dataloaders)
    dataloader_lengths = {fb_type: len(dl) for fb_type, dl in dataloaders.items()}
    
    
    print(f"Training info:")
    for fb_type, length in dataloader_lengths.items():
        print(f"  {fb_type}: {length} batches per epoch")
    print(f"  Batches processed per update: {len(dataloaders)}")

    dloader_iters = {k: iter(dataloaders[k]) for k in active_feedback_types}
    total_data_len = sum(len(dset) for dset in datasets.values())
    steps_per_epoch = total_data_len // args.batch_size
    sampling_probs = [len(datasets[k]) / total_data_len for k in active_feedback_types]

    for epoch in range(args.num_epochs):
        
        # Training
        fb_model.train()
        
        # Track losses per feedback type
        total_loss = 0
        loss_sums = {fb_type: 0.0 for fb_type in dataloaders.keys()}
        kl_div_sum = 0.0
        nll_sum = 0.0
        
        for step in range(steps_per_epoch):

            fb_type = np.random.choice(list(active_feedback_types), p=sampling_probs)

            try:
                batch = next(dloader_iters[fb_type])
            except StopIteration:
                dloader_iters[fb_type] = iter(dataloaders[fb_type])
                batch = next(dloader_iters[fb_type])
            
            # Forward pass
            loss_dict = fb_model(**batch)
            loss = elbo_loss(
                loss_dict["negative_log_likelihood"], 
                loss_dict["kl_divergence"], 
                kl_weight=args.kl_weight
            )
            
            # Backpropagate
            loss.backward()
            
            # Single optimizer step after processing all feedback types
            optimizer.step()
            optimizer.zero_grad()

            # Track losses
            loss_sums[fb_type] += loss.item()
            
            kl_div_sum += loss_dict["kl_divergence"].item()
            nll_sum += loss_dict["negative_log_likelihood"].item()
            
            # Calculate global step for wandb logging
            global_step = epoch * steps_per_epoch + step
            
            # Log to wandb every batch
            wandb_log = {
                "batch/total_loss": loss.item(),
                "batch/kl_divergence": loss_dict["kl_divergence"].item(),
                "batch/negative_log_likelihood": loss_dict["negative_log_likelihood"].item(),
                "batch/learning_rate": optimizer.param_groups[0]['lr'],
            }
            if args.usewb:
                wandb.log(wandb_log, step=global_step)
            
            # Log every few batches to console
            if (step + 1) % max(1, steps_per_epoch // 10) == 0:
                avg_kl = kl_div_sum / (step + 1)
                avg_nll = nll_sum / (step + 1)
                log_parts = [f"Epoch {epoch}, Batch {step + 1}/{steps_per_epoch}"]
                for fb_type in sorted(dataloaders.keys()):
                    avg_loss = loss_sums[fb_type] / (step + 1)
                    log_parts.append(f"{fb_type.capitalize()}: {avg_loss:.4f}")
                log_parts.append(f"KL: {avg_kl:.4f}")
                log_parts.append(f"NLL: {avg_nll:.4f}")
                print(" - ".join(log_parts))
        
        # Epoch summary
        avg_total_loss = total_loss / steps_per_epoch
        avg_kl_div = kl_div_sum / steps_per_epoch
        avg_nll = nll_sum / steps_per_epoch
        
        summary_parts = [f"\nEpoch {epoch} Summary - Total: {avg_total_loss:.4f}"]
        for fb_type in sorted(dataloaders.keys()):
            avg_loss = loss_sums[fb_type] / steps_per_epoch
            summary_parts.append(f"{fb_type.capitalize()}: {avg_loss:.4f}")
        summary_parts.append(f"KL: {avg_kl_div:.4f}")
        summary_parts.append(f"NLL: {avg_nll:.4f}")
        print(", ".join(summary_parts) + "\n")
        
        # -----------------------------------------------
        # Evaluation
        # -----------------------------------------------
        if epoch % args.eval_every_n_epochs == 0:
            fb_model.eval()
            with torch.no_grad():

                # Compute the mean estimated reward per state-action pair
                R_est = np.empty((n_states, n_actions))
                for a in range(n_actions):
                    a_feats_repped = action_feats[a, :].repeat(n_states, 1)
                    R_mean_a, _ = fb_model.encoder(state_feats_flat, a_feats_repped)
                    R_est[:, a] = to_numpy(R_mean_a).squeeze()

                # Compute epic distance
                epic_dist = evaluate_epic_distance(
                    R_true=env.R,
                    R_est=R_est,
                    gamma=args.gamma)
                
                # Compute expected regret
                regret = evaluate_regret(R_est=R_est, R_true=env.R, P=env.P, gamma=args.gamma)


            # Log to wandb
            if args.usewb:
                wandb.log({
                    "eval/epic_distance": epic_dist,
                    "eval/expected_regret": regret,
                    "epoch": epoch,
                })

                epoch_log = {
                    "epoch/loss": avg_total_loss,
                    "epoch/kl_divergence": avg_kl_div,
                    "epoch/negative_log_likelihood": avg_nll,
                    "epoch/learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                }
                # Log per-feedback-type epoch averages
                for fb_type in dataloaders.keys():
                    epoch_log[f"epoch/loss_{fb_type}"] = loss_sums[fb_type] / steps_per_epoch
                wandb.log(epoch_log, step=global_step)
        
            # Evaluation
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    visualize_rewards(env, one_hot_encode_actions, fb_model, device)
            
            # Set model back to training mode
            fb_model.train()
    
    # Finish wandb run
    wandb.finish()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument("--num_pref_samples", type=int, default=64, help="Number of preference samples (0 to disable)")
    parser.add_argument("--num_demo_samples", type=int, default=64, help="Number of demonstration samples (0 to disable)")
    parser.add_argument("--reward_domain", type=str, default="sa", help="Either state-only ('s'), state-action ('sa'), state-action-next-state ('sas')")
    parser.add_argument("--num_steps", type=int, default=32, help="Length of each trajectory")
    parser.add_argument("--td_error_weight", type=float, default=1.0, help="Weight for TD-error constraint in demonstrations")
    
    # Policy parameters
    parser.add_argument("--pref_rationality", type=float, default=2.0, help="Rationality for preference generation")
    parser.add_argument("--expert_rationality", type=float, default=6.0, help="Rationality for expert policy")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--eval_every_n_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl_weight", type=float, default=0.01, help="KL weight - use kl_restart_period for annealing")
    
    # Environment parameters
    parser.add_argument("--grid_size", type=int, default=16)
    parser.add_argument("--n_dct_basis_fns", type=int, default=12)
    parser.add_argument("--reward_type", type=str, default="cliff")
    parser.add_argument("--p_rand", type=float, default=0.0, help="Randomness in transitions (0 for deterministic)")
    
    # Visualization parameters
    parser.add_argument("--visualize_dataset", action="store_true", help="Visualize dataset state-action visitation before training")
    
    # Wandb parameters
    parser.add_argument("--log_wandb", action="store_true", help="Log to weights and biases")
    parser.add_argument("--wandb_project", type=str, default="var-rew-learning", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name (default: auto-generated)")
    parser.add_argument("--wandb_tags", type=str, default="", help="Comma-separated wandb tags")
    parser.add_argument("--wandb_watch", action="store_true", help="Enable wandb model watching (logs gradients and parameters)")
    
    args = parser.parse_args()
    main(args)