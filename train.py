import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset
from virel.envs.dct_grid_env import DCTGridEnv
from virel.data.preference_dataset import PreferenceDataset
from virel.data.demonstration_dataset import DemonstrationDataset
from virel.multi_fb_model import MultiFeedbackTypeModel
from virel.single_fb_mod_model import SingleFeedbackTypeModel
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

    uniform_policy = UniformPolicy(env.action_space)
    expert_policy = ExpertPolicy(
        env=env,
        rationality=args.expert_rationality,
        gamma=args.gamma
    )

    print("Testing policies...")
    print(f"Uniform policy action: {uniform_policy(env.reset())}")
    print(f"Expert policy action: {expert_policy(env.reset())}")

    device = get_device()
    print(f"Using device: {device}")

    # Define function that performs one-hot encoding of actions
    def one_hot_encode_actions(actions) -> torch.Tensor:
        """Convert integer action to one-hot encoded tensor."""
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.long)
        return torch.nn.functional.one_hot(actions, num_classes=env.action_space.n).float()

    # Create preference dataset with uniform policy
    pref_dataset = PreferenceDataset(
        n_samples=args.num_pref_samples,
        n_steps=args.num_steps,
        rationality=args.pref_rationality,
        env=env,
        policy=uniform_policy,
        device=device,
        act_transform=one_hot_encode_actions,
    )
    print(f"Created preference dataset with {len(pref_dataset)} samples")
    
    # Create demonstration dataset with expert policy
    demo_dataset = DemonstrationDataset(
        n_samples=args.num_demo_samples,
        n_steps=args.num_steps,
        env=env,
        policy=expert_policy,
        device=device,
        act_transform=one_hot_encode_actions,
    )
    print(f"Created demonstration dataset with {len(demo_dataset)} samples")
    
    # Combine datasets
    preference_dataloader = DataLoader(pref_dataset, batch_size=args.batch_size, shuffle=True)
    demonstration_dataloader = DataLoader(demo_dataset, batch_size=args.batch_size, shuffle=True)

    obs_dim = env.observation_space["observation"].shape[0]
    act_dim = env.action_space.n
    feature_module = MLPFeatureModule(obs_dim, act_dim, [128, 128])

    # Create feedback model
    reward_encoder = RewardEncoder(feature_module)

    # preference decoder
    preference_decoder = PreferenceDecoder()

    # demonstration decoder
    q_value_model = QValueModel(obs_dim, [128, act_dim])
    demonstration_decoder = DemonstrationsDecoder(q_value_model)

    fb_model = MultiFeedbackTypeModel(
        encoder=reward_encoder,
        decoders={
            "preference": preference_decoder,
            "demonstration": demonstration_decoder,
        }
    )
    fb_model.to(device)

    optimizer = torch.optim.Adam(fb_model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):

        # Training
        fb_model.train()
        for batch_idx, batch in enumerate(demonstration_dataloader):

            # Forward pass
            loss_dict = fb_model(**batch)

            # Backward pass
            loss = elbo_loss(loss_dict["negative_log_likelihood"], loss_dict["kl_divergence"], kl_weight=args.kl_weight)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Log
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
        
        # Evaluation
        fb_model.eval()
        with torch.no_grad():
            visualize_rewards(env, None, one_hot_encode_actions, fb_model, device)

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument("--num_pref_samples", type=int, default=100, help="Number of preference samples")
    parser.add_argument("--num_demo_samples", type=int, default=1000, help="Number of demonstration samples")
    parser.add_argument("--num_steps", type=int, default=100, help="Length of each trajectory")
    
    # Policy parameters
    parser.add_argument("--pref_rationality", type=float, default=1.0, help="Rationality for preference generation")
    parser.add_argument("--expert_rationality", type=float, default=5.0, help="Rationality for expert policy")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl_weight", type=float, default=0.0)
    
    # Environment parameters
    parser.add_argument("--grid_size", type=int, default=16)
    parser.add_argument("--n_dct_basis_fns", type=int, default=8)
    parser.add_argument("--reward_type", type=str, default="cliff")
    parser.add_argument("--p_rand", type=float, default=0.0, help="Randomness in transitions (0 for deterministic)")
    
    args = parser.parse_args()
    main(args)