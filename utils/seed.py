import random
import numpy as np
import torch


def set_global_seed(seed: int = 42):
    """
    Set global random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in PyTorch.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


__all__ = ["set_global_seed"]
