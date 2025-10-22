import argparse
import torch
import itertools
from torch.utils.data import DataLoader
from virel.envs.dct_grid_env import DCTGridEnv
from virel.data.preference_dataset import PreferenceDataset
from virel.data.demonstration_dataset import DemonstrationDataset
from virel.multi_fb_model import MultiFeedbackTypeModel
from virel.utils.policies import UniformPolicy, ExpertPolicy
from virel.encoder.reward_encoder import RewardEncoder
from virel.encoder.features import MLPFeatureModule, QValueModel
from virel.log_likelihoods.preference import PreferenceDecoder
from virel.log_likelihoods.demonstrations import DemonstrationsDecoder
from virel.utils.torch import get_device
from virel.losses import elbo_loss
from virel.visualization.dct_grid_env_visualizer import visualize_rewards

def main(args):

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
    if "preference" in active_feedback_types:
        uniform_policy = UniformPolicy(env.action_space)
        policies_created.add("uniform")
    if "demonstration" in active_feedback_types:
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
        q_value_model = QValueModel(obs_dim, [128, act_dim])
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

    optimizer = torch.optim.Adam(fb_model.parameters(), lr=args.lr)

    # Calculate number of batches per epoch (use the max across all dataloaders)
    dataloader_lengths = {fb_type: len(dl) for fb_type, dl in dataloaders.items()}
    batches_per_epoch = max(dataloader_lengths.values())
    
    print(f"Training info:")
    for fb_type, length in dataloader_lengths.items():
        print(f"  {fb_type}: {length} batches per epoch")
    print(f"  Total update steps per epoch: {batches_per_epoch}")
    print(f"  Batches processed per update: {len(dataloaders)}")
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
            
            # Log every few batches
            if (batch_idx + 1) % max(1, batches_per_epoch // 100) == 0:
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
        
        # Evaluation
        fb_model.eval()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                visualize_rewards(env, one_hot_encode_actions, fb_model, device)

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument("--num_pref_samples", type=int, default=0, help="Number of preference samples (0 to disable)")
    parser.add_argument("--num_demo_samples", type=int, default=2, help="Number of demonstration samples (0 to disable)")
    parser.add_argument("--num_steps", type=int, default=32, help="Length of each trajectory")
    
    # Policy parameters
    parser.add_argument("--pref_rationality", type=float, default=1.0, help="Rationality for preference generation")
    parser.add_argument("--expert_rationality", type=float, default=2.0, help="Rationality for expert policy")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--kl_weight", type=float, default=1.0)
    
    # Environment parameters
    parser.add_argument("--grid_size", type=int, default=8)
    parser.add_argument("--n_dct_basis_fns", type=int, default=8)
    parser.add_argument("--reward_type", type=str, default="five_goals")
    parser.add_argument("--p_rand", type=float, default=0.0, help="Randomness in transitions (0 for deterministic)")
    
    args = parser.parse_args()
    main(args)