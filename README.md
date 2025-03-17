# PPO in Plain C / CUDA

A lightweight implementation of Proximal Policy Optimization (PPO) in C with CUDA acceleration.
This implementation features a dense neural network architecture that can learn the Pendulum-v1 task in approximately 5 seconds on an NVIDIA RTX 2080.

## Features

- Fast PPO implementation in pure C with optional CUDA acceleration
- Simple dense neural network architecture
- Gymnasium integration for reinforcement learning environments
- Minimal dependencies

## Requirements

- CUDA Toolkit (for GPU acceleration)
- GCC compiler
- Python 3.10 with `scripts/gym_env.py` installed
- Make

## Setup

1. Set up a Python environment with the required packages:
```sh
# Create a conda environment (optional)
conda create -n ppo_env python=3.10
conda activate ppo_env

# Install the required packages
cd scripts
pip install .
```

2. Set the `PPO_PYTHON` environment variable to point to your Python environment:
```sh
# For conda environments
export PPO_PYTHON="$HOME/miniconda3/envs/ppo_env"  # Adjust path as needed

# For system Python
export PPO_PYTHON="/usr/bin/python3"  # Adjust path as needed
```

## Running the Project

### Build and Run

```sh
make
./bin/ppo
```

### Debug

```sh
make debug
./bin/ppo
```

### Clean
```sh
make clean
```

### CPU-Only Version

Pure C version without CUDA available on the `plain_c` branch

