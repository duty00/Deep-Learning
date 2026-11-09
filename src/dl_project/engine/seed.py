"""Seed control for Python, NumPy, PyTorch (CPU/CUDA) and DataLoader workers.

Fixing seeds makes runs repeatable on the same hardware/software, but cuDNN kernels and AMP can still
introduce small run-to-run differences; that is why key results are repeated over 3 seeds.
"""
import os
import random

import numpy as np


def set_seed(seed, deterministic=False):
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def worker_init_fn(worker_id):
    """Give each DataLoader worker a distinct, reproducible seed derived from the torch base seed."""
    import torch

    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed((seed + worker_id) % 2**32)


def make_generator(seed):
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
