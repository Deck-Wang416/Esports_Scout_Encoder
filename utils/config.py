# utils/config.py
from types import SimpleNamespace as NS
from pathlib import Path
import yaml
import os

def _to_ns(obj):
    """dict/list 标准化为可点号访问的 SimpleNamespace（递归）"""
    if isinstance(obj, dict):
        return NS(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj

def load_yaml(path: str):
    """读取 YAML，返回原始 dict/list"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_yaml_ns(path: str):
    """读取 YAML，返回 SimpleNamespace（点号访问）"""
    return _to_ns(load_yaml(path))

def load_cfg(path: str):
    """
    读取主配置（如 configs/encoder.yaml），返回 SimpleNamespace。
    预留环境变量覆盖（可选）。
    """
    cfg = load_yaml(path) or {}
    # 简单示例：允许通过环境变量覆盖某些键（可按需扩展）
    # 如：ENCODER_HIDDEN_DIM=512
    env_overrides = {
        ("model", "hidden_dim"): os.getenv("ENCODER_HIDDEN_DIM"),
        ("model", "output_dim"): os.getenv("ENCODER_OUTPUT_DIM"),
    }
    for (k1, k2), v in env_overrides.items():
        if v is not None:
            cfg.setdefault(k1, {})
            cfg[k1][k2] = type(cfg[k1].get(k2, 0))(v)  # 尝试保持原类型
    return _to_ns(cfg)

def load_vocab_dir(vocab_dir: str):
    """
    读取 vocab 目录下若干 YAML（action/location/outcome/impact/weapon），
    返回：
      NS(
        action=NS(tokens=[...]),
        location=NS(tokens=[...]),
        outcome=NS(tokens=[...]),
        impact=NS(tokens=[...]),
        weapon=NS(tokens=[...]),
      )
    若某个文件缺失，也会给一个占位（tokens=["PAD","UNK"]）
    """
    vocab_dir = Path(vocab_dir)
    out = {}
    for name in ["action", "location", "outcome", "impact", "weapon"]:
        p = vocab_dir / f"{name}.yaml"
        if p.exists():
            out[name] = load_yaml_ns(str(p))
            # 兼容只给 tokens 列表的情况
            if not hasattr(out[name], "tokens"):
                # 如果文件内容就是一个列表，包装一下
                if isinstance(out[name], list):
                    out[name] = NS(tokens=out[name])
                else:
                    out[name] = NS(tokens=["PAD", "UNK"])
        else:
            out[name] = NS(tokens=["PAD", "UNK"])
    return _to_ns(out)
