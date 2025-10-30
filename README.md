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

See `setup.py` for the complete list of dependencies.

