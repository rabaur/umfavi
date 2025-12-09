from umfavi.data.preference_dataset import PreferenceDataset
from umfavi.data.demonstration_dataset import DemonstrationDataset
from umfavi.types import FeedbackType
from torch.utils.data import DataLoader

def get_dataset(active_feedback_types, args, env, policies, device, obs_transform, action_transform):
    datasets = {}
    dataloaders = {}
    if FeedbackType.PREFERENCE in active_feedback_types:
        pref_dataset = PreferenceDataset(
            n_samples=args.num_pref_samples,
            n_steps=args.num_steps,
            env=env,
            policy=policies[FeedbackType.PREFERENCE],
            device=device,
            rationality=args.pref_rationality,
            gamma=args.gamma,
            obs_transform=obs_transform,
            act_transform=action_transform,
        )
        datasets[FeedbackType.PREFERENCE] = pref_dataset
        dataloaders[FeedbackType.PREFERENCE] = DataLoader(pref_dataset, batch_size=args.batch_size, shuffle=True)
        print(f"Created preference dataset with {len(pref_dataset)} samples")

    if FeedbackType.DEMONSTRATION in active_feedback_types:
        demo_dataset = DemonstrationDataset(
            num_demonstrations=args.num_demo_samples,
            num_steps=None,
            env=env,
            policy=policies[FeedbackType.DEMONSTRATION],
            device=device,
            rationality=args.demo_rationality,
            gamma=args.gamma,
            td_error_weight=args.td_error_weight,
            obs_transform=obs_transform,
            act_transform=action_transform,
        )
        datasets[FeedbackType.DEMONSTRATION] = demo_dataset
        dataloaders[FeedbackType.DEMONSTRATION] = DataLoader(demo_dataset, batch_size=args.batch_size, shuffle=True)
        print(f"Created demonstration dataset with {len(demo_dataset)} samples")
    return datasets, dataloaders