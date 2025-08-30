from types import SimpleNamespace as NS
from pathlib import Path
import yaml
import os


def _to_ns(obj):
    """Recursively convert dict/list to SimpleNamespace for dot access."""
    if isinstance(obj, dict):
        return NS(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load_yaml(path: str):
    """Load YAML and return raw dict/list."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_yaml_ns(path: str):
    """Load YAML and return SimpleNamespace (dot-access)."""
    return _to_ns(load_yaml(path))


def load_cfg(path: str):
    """
    Load main config (e.g., configs/encoder.yaml) and return SimpleNamespace.
    Allows optional overrides from environment variables.
    """
    cfg = load_yaml(path) or {}

    # Example: override some keys via env vars
    env_overrides = {
        ("model", "hidden_dim"): os.getenv("ENCODER_HIDDEN_DIM"),
        ("model", "output_dim"): os.getenv("ENCODER_OUTPUT_DIM"),
    }
    for (k1, k2), v in env_overrides.items():
        if v is not None:
            cfg.setdefault(k1, {})
            # keep original type if possible
            cfg[k1][k2] = type(cfg[k1].get(k2, 0))(v)
    return _to_ns(cfg)


def load_vocab_dir(vocab_dir: str):
    """
    Load vocab files (action/location/outcome/impact/weapon) from a directory.
    Return a namespace:
      NS(
        action=NS(tokens=[...]),
        location=NS(tokens=[...]),
        outcome=NS(tokens=[...]),
        impact=NS(tokens=[...]),
        weapon=NS(tokens=[...]),
      )
    If a file is missing, fallback to ["PAD", "UNK"].
    """
    vocab_dir = Path(vocab_dir)
    out = {}
    for name in ["action", "location", "outcome", "impact", "weapon"]:
        p = vocab_dir / f"{name}.yaml"
        if p.exists():
            out[name] = load_yaml_ns(str(p))
            # if file only has a list, wrap it
            if not hasattr(out[name], "tokens"):
                if isinstance(out[name], list):
                    out[name] = NS(tokens=out[name])
                else:
                    out[name] = NS(tokens=["PAD", "UNK"])
        else:
            out[name] = NS(tokens=["PAD", "UNK"])
    return _to_ns(out)
