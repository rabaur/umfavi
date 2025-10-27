import argparse
import torch
import itertools
import wandb
import numpy as np
from torch.utils.data import DataLoader
from virel.envs.dct_grid_env import DCTGridEnv
from virel.data.preference_dataset import PreferenceDataset
from virel.data.demonstration_dataset import DemonstrationDataset
from virel.metrics.epic import evaluate_epic_distance
from virel.metrics.regret import evaluate_regret
from virel.multi_fb_model import MultiFeedbackTypeModel
from virel.utils.policies import UniformPolicy, ExpertPolicy
from virel.encoder.reward_encoder import RewardEncoder
from virel.encoder.features import MLPFeatureModule, QValueModel
from virel.log_likelihoods.preference import PreferenceDecoder
from virel.log_likelihoods.demonstrations import DemonstrationsDecoder
from virel.utils.torch import get_device, to_numpy
from virel.losses import elbo_loss
from virel.visualization.dct_grid_env_visualizer import visualize_rewards, visualize_state_action_visitation

def main(args):
    
    # Initialize wandb
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

    
    # Define function that performs one-hot encoding of actions
    def one_hot_encode_actions(actions) -> torch.Tensor:
        """Convert integer action to one-hot encoded tensor."""
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.long)
        return torch.nn.functional.one_hot(actions, num_classes=env.action_space.n).float()
        
    # Get all state-action features for evaluation
    n_states = env.S.shape[0]
    n_actions = env.action_space.n
    state_feats_flat = torch.tensor(env.S, dtype=torch.float32).to(device=device)
    action_feats = torch.zeros((n_actions, n_actions), dtype=torch.float32).to(device=device)
    for a in range(n_actions):
        action_feats[a, :] = one_hot_encode_actions(a)

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

    uniform_policy = UniformPolicy(env.action_space)
    policies_created.add("uniform")
    expert_policy = ExpertPolicy(env=env, rationality=args.expert_rationality, gamma=args.gamma)
    policies_created.add("expert")
    
    # Test policies if created
    if policies_created:
        print("Testing policies...")
        if "uniform" in policies_created:
            print(f"Uniform policy action: {uniform_policy(env.reset())}")
        if "expert" in policies_created:
            print(f"Expert policy action: {expert_policy(env.reset())}")
        print()
    
    # Create datasets and dataloaders
    dataloaders = {}
    
    if "preference" in active_feedback_types:
        pref_dataset = PreferenceDataset(
            n_samples=args.num_pref_samples,
            n_steps=args.num_steps,
            env=env,
            policy=uniform_policy,
            device=device,
            act_transform=one_hot_encode_actions,
            rationality=args.pref_rationality,
        )
        dataloaders["preference"] = DataLoader(pref_dataset, batch_size=args.batch_size, shuffle=True)
        print(f"Created preference dataset with {len(pref_dataset)} samples")
    
    if "demonstration" in active_feedback_types:
        demo_dataset = DemonstrationDataset(
            n_samples=args.num_demo_samples,
            n_steps=args.num_steps,
            env=env,
            policy=expert_policy,
            device=device,
            act_transform=one_hot_encode_actions,
            rationality=args.expert_rationality,
            gamma=args.gamma,
        )
        dataloaders["demonstration"] = DataLoader(demo_dataset, batch_size=args.batch_size, shuffle=True)
        print(f"Created demonstration dataset with {len(demo_dataset)} samples")
    
    print()

    # Visualize dataset visitation if requested
    if args.visualize_dataset:
        print("\nVisualizing state-action visitation distribution...")
        datasets_dict = {}
        if "preference" in active_feedback_types:
            datasets_dict["preference"] = pref_dataset
        if "demonstration" in active_feedback_types:
            datasets_dict["demonstration"] = demo_dataset
        visualize_state_action_visitation(env, datasets_dict, normalize=True)

    # Create feature module and encoder
    obs_dim = env.observation_space["observation"].shape[0]
    act_dim = env.action_space.n
    feature_module = MLPFeatureModule(obs_dim, act_dim, [128, 128])
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
    
    print()
    
    # Create multi-feedback model
    fb_model = MultiFeedbackTypeModel(
        encoder=reward_encoder,
        decoders=decoders
    )
    fb_model.to(device)
    
    # Watch model with wandb (log gradients and parameters)
    wandb.watch(fb_model, log="all", log_freq=100)

    optimizer = torch.optim.Adam(fb_model.parameters(), lr=args.lr)

    # Create KL weight scheduler using cosine annealing with warm restarts
    # We use a dummy parameter to leverage PyTorch's scheduler
    dummy_param = torch.nn.Parameter(torch.tensor(1.0))
    # Calculate number of batches per epoch (use the max across all dataloaders)
    dataloader_lengths = {fb_type: len(dl) for fb_type, dl in dataloaders.items()}
    batches_per_epoch = max(dataloader_lengths.values())
    dummy_optimizer = torch.optim.SGD([dummy_param], lr=1.0)
    
    
    print(f"Training info:")
    for fb_type, length in dataloader_lengths.items():
        print(f"  {fb_type}: {length} batches per epoch")
    print(f"  Total update steps per epoch: {batches_per_epoch}")
    print(f"  Batches processed per update: {len(dataloaders)}")
    print(f"\nKL Weight Annealing:")
    print(f"  Target KL weight: ~1.0000 (annealing towards maximum)")
    print()

    for epoch in range(args.num_epochs):
        
        # Training
        fb_model.train()
        
        # Create cyclic iterators for all active dataloaders
        dataloader_iters = {
            fb_type: itertools.cycle(dl) 
            for fb_type, dl in dataloaders.items()
        }
        
        # Track losses per feedback type
        total_loss = 0
        loss_sums = {fb_type: 0.0 for fb_type in dataloaders.keys()}
        kl_div_sum = 0.0
        nll_sum = 0.0
        
        for batch_idx in range(batches_per_epoch):
            
            # Accumulate losses from all feedback types
            batch_total_loss = 0
            batch_kl_div = 0
            batch_nll = 0
            
            for fb_type, data_iter in dataloader_iters.items():
                # Get next batch for this feedback type
                batch = next(data_iter)
                
                # Forward pass
                loss_dict = fb_model(**batch)
                loss = elbo_loss(
                    loss_dict["negative_log_likelihood"], 
                    loss_dict["kl_divergence"], 
                    kl_weight=args.kl_weight
                )
                
                # Accumulate gradients (don't step yet)
                loss /= len(dataloaders) # normalize by the number of feedback types
                loss.backward()
                
                # Track losses
                loss_sums[fb_type] += loss.item()
                batch_total_loss += loss.item()
                batch_kl_div += loss_dict["kl_divergence"].item()
                batch_nll += loss_dict["negative_log_likelihood"].item()
            
            # Single optimizer step after processing all feedback types
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += batch_total_loss
            kl_div_sum += batch_kl_div
            nll_sum += batch_nll
            
            # Calculate global step for wandb logging
            global_step = epoch * batches_per_epoch + batch_idx
            
            # Log to wandb every batch
            wandb_log = {
                "batch/loss": batch_total_loss,
                "batch/kl_divergence": batch_kl_div,
                "batch/negative_log_likelihood": batch_nll,
                "batch/learning_rate": optimizer.param_groups[0]['lr'],
            }
            # Log per-feedback-type losses
            for fb_type in dataloaders.keys():
                wandb_log[f"batch/loss_{fb_type}"] = loss_sums[fb_type] / (batch_idx + 1)
            wandb.log(wandb_log, step=global_step)
            
            # Log every few batches to console
            if (batch_idx + 1) % max(1, batches_per_epoch // 10) == 0:
                avg_kl = kl_div_sum / (batch_idx + 1)
                avg_nll = nll_sum / (batch_idx + 1)
                log_parts = [f"Epoch {epoch}, Batch {batch_idx + 1}/{batches_per_epoch}"]
                for fb_type in sorted(dataloaders.keys()):
                    avg_loss = loss_sums[fb_type] / (batch_idx + 1)
                    log_parts.append(f"{fb_type.capitalize()}: {avg_loss:.4f}")
                log_parts.append(f"KL: {avg_kl:.4f}")
                log_parts.append(f"NLL: {avg_nll:.4f}")
                print(" - ".join(log_parts))
        
        # Epoch summary
        avg_total_loss = total_loss / batches_per_epoch
        avg_kl_div = kl_div_sum / batches_per_epoch
        avg_nll = nll_sum / batches_per_epoch
        
        summary_parts = [f"\nEpoch {epoch} Summary - Total: {avg_total_loss:.4f}"]
        for fb_type in sorted(dataloaders.keys()):
            avg_loss = loss_sums[fb_type] / batches_per_epoch
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
                regret = evaluate_regret(
                    R_est=R_est,
                    R_true=env.R,
                    P=env.P,
                    gamma=args.gamma
                )


            # Log to wandb
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
                epoch_log[f"epoch/loss_{fb_type}"] = loss_sums[fb_type] / batches_per_epoch
            wandb.log(epoch_log, step=global_step)
        
            # Evaluation
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    visualize_rewards(env, one_hot_encode_actions, fb_model, device)
    
    # Finish wandb run
    wandb.finish()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument("--num_pref_samples", type=int, default=0, help="Number of preference samples (0 to disable)")
    parser.add_argument("--num_demo_samples", type=int, default=128, help="Number of demonstration samples (0 to disable)")
    parser.add_argument("--num_steps", type=int, default=32, help="Length of each trajectory")
    
    # Policy parameters
    parser.add_argument("--pref_rationality", type=float, default=1.0, help="Rationality for preference generation")
    parser.add_argument("--expert_rationality", type=float, default=1.0, help="Rationality for expert policy")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--eval_every_n_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl_weight", type=float, default=0.01, help="KL weight - use kl_restart_period for annealing")
    parser.add_argument("--kl_restart_epochs", type=float, default=1, help="Number of epochs for KL weight restarts (0 = no restarts, standard cosine annealing)")
    parser.add_argument("--kl_restart_mult", type=float, default=0.5, help="Multiplier for KL weight restarts (T_mult parameter)")
    
    # Environment parameters
    parser.add_argument("--grid_size", type=int, default=16)
    parser.add_argument("--n_dct_basis_fns", type=int, default=11)
    parser.add_argument("--reward_type", type=str, default="path")
    parser.add_argument("--p_rand", type=float, default=0.0, help="Randomness in transitions (0 for deterministic)")
    
    # Visualization parameters
    parser.add_argument("--visualize_dataset", action="store_true", help="Visualize dataset state-action visitation before training")
    
    # Wandb parameters
    parser.add_argument("--wandb_project", type=str, default="var-rew-learning", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name (default: auto-generated)")
    parser.add_argument("--wandb_tags", type=str, default="", help="Comma-separated wandb tags")
    parser.add_argument("--wandb_watch", action="store_true", help="Enable wandb model watching (logs gradients and parameters)")
    
    args = parser.parse_args()
    main(args)