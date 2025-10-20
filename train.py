import argparse
import torch
from torch.utils.data import DataLoader
from virel.envs.dct_grid_env import DCTGridEnv
from virel.data.preference_dataset import PreferenceDataset
from virel.single_fb_mod_model import SingleFeedbackTypeModel
from virel.utils.policies import UniformPolicy
from virel.encoder.reward_encoder import RewardEncoder
from virel.encoder.features import MLPFeatureModule
from virel.log_likelihoods.preference import PreferenceDecoder
from virel.utils.torch import get_device
from virel.losses import elbo_loss

def main(args):

    env = DCTGridEnv(
        grid_size=args.grid_size,
        n_dct_basis_fns=args.n_dct_basis_fns,
        reward_type=args.reward_type,
        p_rand=args.p_rand,
    )

    uniform_policy = UniformPolicy(env.action_space)

    print(uniform_policy(env.reset()))

    device = get_device()
    print(f"Using device: {device}")

    # Define function that performs one-hot encoding of actions
    def one_hot_encode_actions(actions: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(torch.tensor(actions, dtype=torch.long), num_classes=env.action_space.n)

    pref_dataset = PreferenceDataset(
        n_samples=args.num_samples,
        n_steps=args.num_steps,
        rationality=args.rationality,
        env=env,
        policy=uniform_policy,
        device=device,
        act_transform=one_hot_encode_actions,
    )
    
    dataloader = DataLoader(pref_dataset, batch_size=args.batch_size, shuffle=True)

    obs_dim = env.observation_space["observation"].shape[0]
    act_dim = env.action_space.n
    feature_module = MLPFeatureModule(obs_dim, act_dim, [128, 128])

    # Create feedback model
    reward_encoder = RewardEncoder(feature_module)
    preference_decoder = PreferenceDecoder()
    fb_model = SingleFeedbackTypeModel(reward_encoder, preference_decoder)
    fb_model.to(device)

    optimizer = torch.optim.Adam(fb_model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(dataloader):

            # Training
            fb_model.train()

            # Destructure batch
            obs1, acts1, obs2, acts2, targets = batch

            # Concatenate observations and actions along batch dimension
            obs = torch.cat([obs1, obs2], dim=0)
            acts = torch.cat([acts1, acts2], dim=0)

            # Forward pass
            r_samples, preds, loss_dict = fb_model(obs, acts, targets)

            # Backward pass
            loss = elbo_loss(loss_dict["negative_log_likelihood"], loss_dict["kl_divergence"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Log
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--rationality", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--n_dct_basis_fns", type=int, default=8)
    parser.add_argument("--reward_type", type=str, default="sparse")
    parser.add_argument("--p_rand", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)