# UMFAVI - Unified Multi-modal Feedback using Amortized Variational Inference

This package implements a variational inference approach for learning reward functions from multiple types of feedback (preferences, demonstrations, etc.).

## Installation

### Local Development Installation

To install the package locally in editable mode:

```bash
pip install -e .
```

This will install `umfavi` and its dependencies. The `-e` flag means "editable", so changes to the source code will be reflected immediately without reinstalling.

### Installing with Development Dependencies

If you want to install additional development tools (testing, linting, etc.):

```bash
pip install -e ".[dev]"
```

### Installing with Non-Tabular Environment Support

For using expert policies with non-tabular environments (e.g., CartPole, Atari):

```bash
pip install -e ".[nontabular]"
```

This installs `stable-baselines3` for training and using DQN-based expert policies.

## Usage

After installation, you can import the package in your Python code:

```python
from umfavi.envs.dct_grid_env import DCTGridEnv
from umfavi.multi_fb_model import MultiFeedbackTypeModel
from umfavi.encoder.reward_encoder import RewardEncoder
# ... etc
```

## Running the Training Script

To run the training script:

```bash
python train.py --help  # See all available options
python train.py         # Run with default parameters
```

## Project Structure

- `umfavi/` - Main package directory
  - `data/` - Dataset implementations
  - `encoder/` - Reward encoder and feature modules
  - `envs/` - Environment implementations
  - `log_likelihoods/` - Log likelihood decoders for different feedback types
  - `losses.py` - Loss functions
  - `metrics/` - Evaluation metrics (EPIC, regret)
  - `multi_fb_model.py` - Multi-feedback type model
  - `priors.py` - Prior distributions
  - `utils/` - Utility functions
  - `visualization/` - Visualization tools

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy >= 1.20.0
- Gymnasium (or gym)
- wandb

### Optional Dependencies

- `stable-baselines3>=2.0.0` - For non-tabular expert policies (install with `pip install -e ".[nontabular]"`)

See `setup.py` for the complete list of dependencies.

## Expert Policies

The package supports expert policies for both **tabular** and **non-tabular** environments:

### Tabular Environments (GridEnv)

For environments with known transition dynamics, the package computes optimal Q-values analytically:

```python
from umfavi.envs.grid_env.env import GridEnv
from umfavi.utils.policies import create_expert_policy

env = GridEnv(grid_size=10, reward_type="sparse", ...)
policy = create_expert_policy(env, rationality=5.0, gamma=0.9)
```

### Non-Tabular Environments (CartPole, etc.)

For standard Gymnasium environments, the package can use DQN-based expert policies:

```python
import gymnasium as gym
from umfavi.utils.policies import create_expert_policy

env = gym.make("CartPole-v1")
policy = create_expert_policy(
    env, 
    rationality=1.0, 
    gamma=0.99,
    train_if_missing=True  # Auto-train if no pretrained model exists
)
```

The `create_expert_policy()` factory function automatically detects the environment type and creates the appropriate policy.

See `POLICY_USAGE.md` for detailed documentation on expert policies.

