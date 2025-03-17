# PPO in Plain C / CUDA

This project expects a path to a Python 3.10 environment in `$(PPO_PYTHON)` with `gymnasium` and `scripts/gym_env.py` installed.

## Setup

Install the required packages:
```sh
cd scripts
pip install .
```

Set `PPO_PYTHON` to the path of the Python environment. For example, with a conda environment named `ppo_env`:
```sh
export PPO_PYTHON="path/to/miniconda3/envs/ppo_env"
```

## Run

```sh
make
./bin/ppo
```

For the pure C version without CUDA, switch to the `plain_c` branch.

