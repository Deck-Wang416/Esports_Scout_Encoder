import argparse
import torch
import yaml
from torch.utils.data import DataLoader

from utils.config import load_cfg, load_yaml_ns, load_vocab_dir
from utils.seed import set_global_seed
from adapters.mock_adapter import MockAdapter
from datasets.player_dataset import PlayerDataset
from dataloading.collate import collate_fn
from models.encoder import Encoder

import warnings
warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage",
    category=UserWarning
)

def has_nan_or_inf(t: torch.Tensor) -> bool:
    return torch.isnan(t).any().item() or torch.isinf(t).any().item()


def main():
    ap = argparse.ArgumentParser(description="Run minimal sanity checks.")
    ap.add_argument("--encoder-cfg", default="configs/encoder.yaml")
    ap.add_argument("--dataloader-cfg", default="configs/dataloader.yaml")
    ap.add_argument("--json", default="data/mock/sample_v01.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_global_seed(args.seed)

    # ---- 1) Load configs ----
    cfg = load_cfg(args.encoder_cfg)
    with open(args.dataloader_cfg, "r", encoding="utf-8") as f:
        dl_cfg = yaml.safe_load(f)

    # Load vocab & normalization (if provided)
    vocab_dir = getattr(getattr(cfg, "paths", object()), "vocab_dir", None) or dl_cfg.get("vocab_dir") or "configs/vocab"
    norm_path = getattr(getattr(cfg, "paths", object()), "normalization", None) or dl_cfg.get("normalization") or "configs/normalization.yaml"

    vocab = load_vocab_dir(vocab_dir)
    try:
        norm = load_yaml_ns(norm_path)
    except FileNotFoundError:
        norm = None
        print(f"[Warn] normalization YAML not found at {norm_path}; continue with norm=None")

    # Compute discrete vocab sizes for Encoder
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

    # Build global tag vocab size from roles/traits (keep single PAD/UNK)
    tag_vocab_path = getattr(getattr(cfg, "paths", object()), "tag_vocab", None) or "configs/tags/tag_vocab.yaml"
    tag_vocab = load_yaml_ns(tag_vocab_path)
    roles_tokens  = getattr(getattr(tag_vocab, "roles", object()), "tokens", []) or []
    traits_tokens = getattr(getattr(tag_vocab, "traits", object()), "tokens", []) or []

    def _drop_pad_unk(tokens):
        return [t for t in tokens if t not in ("PAD", "UNK")]

    global_tag_tokens = ["PAD", "UNK"] + _drop_pad_unk(roles_tokens) + _drop_pad_unk(traits_tokens)
    tag_vocab_size = len(global_tag_tokens)
    token2id = {tok: i for i, tok in enumerate(global_tag_tokens)}

    # ---- 2) Adapter / Dataset / DataLoader ----
    adapter = MockAdapter(
        vocab_cfg=vocab,
        norm_cfg=norm,
        T=dl_cfg.get("T", 32),
        k_multi=3,
    )
    dataset = PlayerDataset([args.json], adapter, split="train", ratio=1.0, seed=args.seed, shuffle_train=True)
    loader = DataLoader(
        dataset,
        batch_size=dl_cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ---- 3) Fetch one batch ----
    batch = next(iter(loader))
    B, T = batch["action_idx"].shape
    print(f"[Info] batch ready: B={B}, T={T}")

    # ---- 4) Build encoder ----
    model = Encoder(cfg, vocab_sizes=vocab_sizes, tag_vocab_size=tag_vocab_size)
    model.eval()

    # ========== Shape checks ==========
    with torch.no_grad():
        z_b = model.encode_behavior(batch)  # [B, D]
        print(f"[Check] encode_behavior -> {tuple(z_b.shape)}")
        assert z_b.ndim == 2 and z_b.shape[0] == B, "Unexpected z_behavior shape"

        # Safe tag_ids from names → ids (avoid out-of-range)
        roles_no_pad  = [t for t in roles_tokens  if t not in ("PAD", "UNK")]
        traits_no_pad = [t for t in traits_tokens if t not in ("PAD", "UNK")]
        sample_tokens = []
        if roles_no_pad:
            sample_tokens.append(roles_no_pad[0])
        if len(roles_no_pad) > 1:
            sample_tokens.append(roles_no_pad[1])
        if traits_no_pad:
            sample_tokens.append(traits_no_pad[0])
        if not sample_tokens:
            sample_tokens = ["UNK"]

        tag_ids = torch.tensor([token2id.get(t, token2id.get("UNK", 1)) for t in sample_tokens], dtype=torch.long)
        z_t = model.encode_tag(tag_ids)  # [N, D]
        print(f"[Check] encode_tag -> {tuple(z_t.shape)}")
        assert z_t.ndim == 2 and z_t.shape[1] == z_b.shape[1], "z_tag should share D with z_behavior"

    # ========== Numeric checks ==========
    assert not has_nan_or_inf(z_b), "NaN/Inf detected in z_behavior"
    assert z_b.abs().sum().item() > 0, "z_behavior appears all-zero"
    mean = z_b.mean().item()
    var = z_b.var(unbiased=False).item()
    print(f"[Check] z_behavior mean={mean:.4f}, var={var:.4f}")

    # ========== Determinism ==========
    with torch.no_grad():
        z_b_again = model.encode_behavior(batch)
    delta = (z_b_again - z_b).abs().max().item()
    print(f"[Check] determinism max|Δ|={delta:.6g}")
    assert delta < 1e-6, "Non-deterministic forward; ensure dropout off / seeds fixed"

    # ========== Input sensitivity (sanity) ==========
    def cosine(a, b, eps=1e-9):
        a = a / (a.norm(dim=1, keepdim=True) + eps)
        b = b / (b.norm(dim=1, keepdim=True) + eps)
        return (a * b).sum(dim=1)

    batch_diff = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    if "action_idx" in batch_diff:
        batch_diff["action_idx"][:, :max(1, T // 4)] = 1  # set to UNK
    with torch.no_grad():
        z_b_diff = model.encode_behavior(batch_diff)
    cos = cosine(z_b, z_b_diff).mean().item()
    print(f"[Check] cosine(z_b, z_b_diff) = {cos:.4f}  (should not be ~1.0)")

    # ========== Mask behavior (strict per-sample trimming) ==========
    # For each sample, trim sequence to its valid length (sum of mask),
    # set mask=1 for the trimmed window, encode, and then compare with the
    # masked forward on the original batch. Results should be close.
    with torch.no_grad():
        z_mask = model.encode_behavior(batch)  # masked forward on full batch

        mask = batch["mask"].bool()  # [B, T]
        valid_len = mask.sum(dim=1).tolist()

        def _slice_sample(bdict, i, L):
            sub = {}
            for k, v in bdict.items():
                if torch.is_tensor(v):
                    # Slice tensors with [B, T, ...] leading dims
                    if v.dim() >= 2 and v.shape[0] == mask.shape[0] and v.shape[1] == mask.shape[1]:
                        sub[k] = v[i:i+1, :L]
                    # Slice tensors with [B, ...] leading dims
                    elif v.dim() >= 1 and v.shape[0] == mask.shape[0]:
                        sub[k] = v[i:i+1]
                    else:
                        sub[k] = v  # leave untouched (e.g., embeddings not in batch)
                else:
                    sub[k] = v  # lists/dicts (e.g., meta)
            # Force mask to all-ones after trimming
            sub["mask"] = torch.ones(1, L, dtype=torch.bool, device=mask.device)
            return sub

        zs = []
        for i, L in enumerate(valid_len):
            L = int(L)
            # If a sample is entirely padding, keep a single pad token to avoid empty tensor
            L_eff = max(L, 1)
            sub = _slice_sample(batch, i, L_eff)
            zs.append(model.encode_behavior(sub))
        z_trim = torch.cat(zs, dim=0)

    diff = (z_mask - z_trim).abs().max().item()
    print(f"[Check] mask consistency max|Δ|={diff:.6g}")

    print("\nSanity checks finished without assertion errors.")


if __name__ == "__main__":
    main()
