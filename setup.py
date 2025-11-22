from setuptools import setup, find_packages

setup(
    name="umfavi",
    version="0.1.0",
    description="Unified Multi-modal Feedback using Amortized Variational Inference",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "gymnasium",  # or gym, depending on your usage
        "wandb",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
        "nontabular": [
            "stable-baselines3>=2.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

