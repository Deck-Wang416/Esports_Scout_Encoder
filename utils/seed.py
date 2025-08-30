# utils/seed.py
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42):
    """
    设置 Python、NumPy、PyTorch 的全局随机种子，保证实验可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # PyTorch 确保确定性行为（可能稍慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

__all__ = ["set_global_seed"]
