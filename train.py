import argparse
from torch.utils.data import DataLoader
from virel.envs.dct_grid_env import DCTGridEnv
from virel.data.preference_dataset import PreferenceDataset
from virel.utils.policies import UniformPolicy

def main(args):

    env = DCTGridEnv(
        grid_size=args.grid_size,
        n_dct_basis_fns=args.n_dct_basis_fns,
        reward_type=args.reward_type,
        p_rand=args.p_rand,
    )

    uniform_policy = UniformPolicy(env.action_space)

    print(uniform_policy(env.reset()))


    pref_dataset = PreferenceDataset(
        n_samples=args.num_samples,
        n_steps=args.num_steps,
        rationality=args.rationality,
        env=env,
        policy=uniform_policy,
    )
    
    dataloader = DataLoader(pref_dataset, batch_size=args.batch_size, shuffle=True)

    for batch in dataloader:
        print(batch)
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--rationality", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--n_dct_basis_fns", type=int, default=10)
    parser.add_argument("--reward_type", type=str, default="sparse")
    parser.add_argument("--p_rand", type=float, default=0.1)
    args = parser.parse_args()
    main(args)