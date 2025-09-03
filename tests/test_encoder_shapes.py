import copy
import math
import os
import sys

import pytest
import torch

# Make sure project root is on sys.path when running pytest from repo root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Project imports
from utils.config import load_cfg, load_yaml_ns, load_vocab_dir
from adapters.mock_adapter import MockAdapter
from datasets.player_dataset import PlayerDataset
from dataloading.collate import collate_fn
from models.encoder import Encoder


# ---------- Helpers ----------

def _build_env(T: int = 16):
    """Load cfg, vocab, tag vocab; build vocab sizes and tag size."""
    cfg = load_cfg("configs/encoder.yaml")
    vocab = load_vocab_dir("configs/vocab")
    norm = load_yaml_ns("configs/normalization.yaml")

    def _safe_len(ns_obj, attr):
        try:
            return len(getattr(getattr(ns_obj, attr), "tokens"))
        except Exception:
            return 0

    vocab_sizes = {
        "action":   _safe_len(vocab, "action"),
        "location": _safe_len(vocab, "location"),
        "outcome":  _safe_len(vocab, "outcome"),
        "impact":   _safe_len(vocab, "impact"),
        "weapon":   _safe_len(vocab, "weapon"),
    }

    tag_vocab = load_yaml_ns("configs/tags/tag_vocab.yaml")
    roles_tokens  = getattr(getattr(tag_vocab, "roles", object()), "tokens", []) or []
    traits_tokens = getattr(getattr(tag_vocab, "traits", object()), "tokens", []) or []

    # Merge roles/traits into a global tag vocab: ["PAD","UNK", roles..., traits...] (drop duplicates of PAD/UNK)
    def _drop_pad_unk(tokens):
        return [t for t in tokens if t not in ("PAD", "UNK")]

    global_tag_tokens = ["PAD", "UNK"] + _drop_pad_unk(roles_tokens) + _drop_pad_unk(traits_tokens)
    tag_vocab_size = len(global_tag_tokens)
    token2id = {tok: i for i, tok in enumerate(global_tag_tokens)}

    return cfg, vocab, norm, vocab_sizes, tag_vocab_size, token2id, roles_tokens, traits_tokens


def _make_batch(json_path="data/mock/sample_v01.json", T: int = 16):
    """Build a single batch via MockAdapter → PlayerDataset → collate_fn."""
    cfg, vocab, norm, vocab_sizes, tag_vocab_size, token2id, roles_tokens, traits_tokens = _build_env(T=T)
    adapter = MockAdapter(vocab_cfg=vocab, norm_cfg=norm, T=T, k_multi=3)
    dataset = PlayerDataset([json_path], adapter, split="train", ratio=1.0, seed=42, shuffle_train=True)

    # Just take first B samples and collate into a batch of size B (here B=min(2, len(dataset)) for speed)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=min(2, max(len(dataset), 1)), shuffle=False, num_workers=0, collate_fn=collate_fn)
    batch = next(iter(loader))
    return batch, (cfg, vocab, norm, vocab_sizes, tag_vocab_size, token2id, roles_tokens, traits_tokens)


def _build_model(cfg, vocab_sizes, tag_vocab_size):
    model = Encoder(cfg, vocab_sizes=vocab_sizes, tag_vocab_size=tag_vocab_size)
    model.eval()  # disable dropout for determinism
    return model


# ---------- Tests ----------

def test_encode_behavior_shape_and_dtype():
    batch, env = _make_batch(T=16)
    cfg, vocab, norm, vocab_sizes, tag_vocab_size, token2id, roles_tokens, traits_tokens = env
    model = _build_model(cfg, vocab_sizes, tag_vocab_size)

    with torch.no_grad():
        z = model.encode_behavior(batch)

    B = batch["action_idx"].shape[0]
    assert z.shape == (B, cfg.model.output_dim)
    assert z.dtype == torch.float32
    assert torch.isfinite(z).all(), "z_behavior contains NaN/Inf"


def test_encode_tag_shape_and_fallback():
    _, env = _make_batch(T=16)
    cfg, vocab, norm, vocab_sizes, tag_vocab_size, token2id, roles_tokens, traits_tokens = env
    model = _build_model(cfg, vocab_sizes, tag_vocab_size)

    # Pick a few valid tokens (prefer roles, then traits)
    roles_no_pad  = [t for t in roles_tokens  if t not in ("PAD", "UNK")]
    traits_no_pad = [t for t in traits_tokens if t not in ("PAD", "UNK")]
    tokens = []
    if roles_no_pad:
        tokens.append(roles_no_pad[0])
    if traits_no_pad:
        tokens.append(traits_no_pad[0])
    if not tokens:
        tokens = ["UNK"]

    ids_list = [token2id.get(t, 1) for t in tokens]
    ids_tensor = torch.tensor(ids_list, dtype=torch.long)

    with torch.no_grad():
        z_list = model.encode_tag(ids_list)
        z_tensor = model.encode_tag(ids_tensor)
    assert z_list.shape == z_tensor.shape == (len(ids_list), cfg.model.output_dim)

    # Out-of-range / negative should map to UNK=1 without crash and match UNK output
    unk = 1
    bad_ids = torch.tensor([tag_vocab_size + 5, -3], dtype=torch.long)
    with torch.no_grad():
        z_bad = model.encode_tag(bad_ids)
        z_unk = model.encode_tag(torch.tensor([unk, unk], dtype=torch.long))
    assert torch.allclose(z_bad, z_unk, atol=1e-6)


def test_determinism_behavior():
    batch, env = _make_batch(T=16)
    cfg, vocab, norm, vocab_sizes, tag_vocab_size, *_ = env
    model = _build_model(cfg, vocab_sizes, tag_vocab_size)

    with torch.no_grad():
        z1 = model.encode_behavior(batch)
        z2 = model.encode_behavior(batch)
    assert torch.allclose(z1, z2, atol=1e-7), "encode_behavior is not deterministic in eval()"


def test_mask_consistency_behavior():
    batch, env = _make_batch(T=16)
    cfg, vocab, norm, vocab_sizes, tag_vocab_size, *_ = env
    model = _build_model(cfg, vocab_sizes, tag_vocab_size)

    # Build a "no-padding" variant: zero out padded positions and set mask=1
    mask = batch["mask"].bool()
    batch_no_pad = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    for k, v in batch_no_pad.items():
        if torch.is_tensor(v) and v.ndim >= 2 and v.shape[:2] == mask.shape:
            if v.dtype == torch.long:
                v[~mask] = 0
            elif v.dtype == torch.bool:
                v[~mask] = False
            else:
                v[~mask] = 0.0
    batch_no_pad["mask"] = torch.ones_like(batch["mask"], dtype=torch.bool)

    with torch.no_grad():
        z_mask = model.encode_behavior(batch)
        z_full = model.encode_behavior(batch_no_pad)

    # The two should be very close; allow tiny numerical noise
    diff = (z_mask - z_full).abs().max().item()
    assert diff < 1e-5, f"Mask inconsistency too large: {diff}"


def test_behavior_changes_with_input_perturbation():
    batch, env = _make_batch(T=16)
    cfg, vocab, norm, vocab_sizes, tag_vocab_size, *_ = env
    model = _build_model(cfg, vocab_sizes, tag_vocab_size)

    # Perturb action_idx → UNK for first quarter timesteps
    B, T = batch["action_idx"].shape
    batch_perturbed = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    span = max(1, T // 4)
    batch_perturbed["action_idx"][:, :span] = 1  # UNK

    with torch.no_grad():
        z0 = model.encode_behavior(batch)
        z1 = model.encode_behavior(batch_perturbed)

    # Not identical
    assert not torch.allclose(z0, z1, atol=1e-6), "Behavior vectors unchanged after strong input perturbation"


def test_attn_pooling_output_shape():
    # Switch cfg.pooling.type → "attn" at runtime and ensure shape is correct
    batch, env = _make_batch(T=16)
    cfg, vocab, norm, vocab_sizes, tag_vocab_size, *_ = env
    cfg_attn = copy.deepcopy(cfg)
    cfg_attn.pooling.type = "attn"
    cfg_attn.pooling.attn_dim = 64

    model = _build_model(cfg_attn, vocab_sizes, tag_vocab_size)
    with torch.no_grad():
        z = model.encode_behavior(batch)
    B = batch["action_idx"].shape[0]
    assert z.shape == (B, cfg_attn.model.output_dim)


def test_all_padding_sequence_is_safe():
    # Construct a batch with all padding, ensure no NaN/Inf
    batch, env = _make_batch(T=8)
    cfg, vocab, norm, vocab_sizes, tag_vocab_size, *_ = env
    model = _build_model(cfg, vocab_sizes, tag_vocab_size)

    # Zero all fields; set mask to all False
    batch_all_pad = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if v.dtype == torch.long:
                batch_all_pad[k] = torch.zeros_like(v)
            elif v.dtype == torch.bool:
                batch_all_pad[k] = torch.zeros_like(v, dtype=torch.bool)
            else:
                batch_all_pad[k] = torch.zeros_like(v)
        else:
            batch_all_pad[k] = v
    batch_all_pad["mask"] = torch.zeros_like(batch["mask"], dtype=torch.bool)

    with torch.no_grad():
        z = model.encode_behavior(batch_all_pad)
    assert torch.isfinite(z).all(), "All-padding sequence produced NaN/Inf"
    # Shape check
    B = batch_all_pad["action_idx"].shape[0]
    assert z.shape == (B, cfg.model.output_dim)
