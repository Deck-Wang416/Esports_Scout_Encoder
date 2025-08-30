"""
最小链路自测：
- 读取 configs
- 构造 Adapter→Dataset→DataLoader (collate_fn)
- 前向 Encoder: encode_behavior / encode_tag
- 维度、数值、一致性、mask 生效检查
"""

import argparse
import math
import torch
import random
import numpy as np

# === 项目内模块（按你的目录结构）===
from utils.config import load_cfg, load_yaml_ns, load_vocab_dir  # 读取主 cfg、单 YAML、以及 vocab 目录
from utils.seed import set_global_seed            # 统一设置随机种子
from adapters.mock_adapter import MockAdapter     # 读取 mock JSON → Samples
from datasets.player_dataset import PlayerDataset
from dataloading.collate import collate_fn
from models.encoder import Encoder

from torch.utils.data import DataLoader
import yaml

def bool_to_float(t):
    return t.float() if t.dtype == torch.bool else t

def has_nan_or_inf(t):
    return torch.isnan(t).any().item() or torch.isinf(t).any().item()

def masked_mean(x, mask):  # 与文档一致
    m = mask.unsqueeze(-1).float()
    denom = m.sum(dim=1).clamp_min(1.0)
    return (x * m).sum(dim=1) / denom

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-cfg", default="configs/encoder.yaml")
    ap.add_argument("--dataloader-cfg", default="configs/dataloader.yaml")
    ap.add_argument("--json", default="data/mock/sample_v01.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_global_seed(args.seed)

    # 1) 读取配置
    cfg = load_cfg(args.encoder_cfg)              # 应返回 obj/Dict: cfg.model, cfg.inputs, cfg.pooling, cfg.paths...
    with open(args.dataloader_cfg, "r", encoding="utf-8") as f:
        dl_cfg = yaml.safe_load(f)

    # 1.1) 读取 vocab 与 normalization（以规范对象传入 Adapter）
    # 优先从 dataloader.yaml/encoder.yaml 的 paths 中读取；若无则用默认路径
    vocab_dir = getattr(getattr(cfg, 'paths', object()), 'vocab_dir', None) or dl_cfg.get('vocab_dir') or 'configs/vocab'
    norm_path = getattr(getattr(cfg, 'paths', object()), 'normalization', None) or dl_cfg.get('normalization') or 'configs/normalization.yaml'

    vocab = load_vocab_dir(vocab_dir)     # 可点号访问：vocab_cfg.action.tokens 等
    try:
        norm  = load_yaml_ns(norm_path)      # e.g., norm_cfg.timestamp.min/max, norm_cfg.damage_sum.min/max
    except FileNotFoundError:
        norm  = None
        print(f"[Warn] normalization YAML not found at {norm_path}; proceeding with norm_cfg=None")

    # 1.2) 计算离散词表尺寸与 tag 词表尺寸（用于构造 Encoder）
    # 计算 action/location/outcome/impact/weapon 的词表大小
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

    # 读取 tag_vocab.yaml 并合并 roles/traits → 全局词表（PAD/UNK 只保留一份）
    tag_vocab_path = getattr(getattr(cfg, "paths", object()), "tag_vocab", None) or "configs/tags/tag_vocab.yaml"
    tag_vocab = load_yaml_ns(tag_vocab_path)
    roles_tokens  = getattr(getattr(tag_vocab, "roles", object()), "tokens", []) or []
    traits_tokens = getattr(getattr(tag_vocab, "traits", object()), "tokens", []) or []

    def _drop_pad_unk(tokens):
        return [t for t in tokens if t not in ("PAD", "UNK")]

    global_tag_tokens = ["PAD", "UNK"] + _drop_pad_unk(roles_tokens) + _drop_pad_unk(traits_tokens)
    tag_vocab_size = len(global_tag_tokens)
    # 基于合并后的全局词表构造 token→id 映射（PAD=0, UNK=1）
    token2id = {tok: i for i, tok in enumerate(global_tag_tokens)}

    # 2) Adapter / Dataset / DataLoader
    adapter = MockAdapter(
        vocab=vocab,
        norm=norm,
        T=dl_cfg.get("T", 32),
        K_multi=3
    )
    dataset = PlayerDataset([args.json], adapter, split="train", ratio=1.0, seed=args.seed, shuffle_train=True)
    loader = DataLoader(
        dataset,
        batch_size=dl_cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # 3) 取一个 batch
    batch = next(iter(loader))
    B, T = batch["action_idx"].shape
    print(f"[Info] batch ready: B={B}, T={T}")

    # 4) 构建 Encoder
    model = Encoder(cfg, vocab_sizes=vocab_sizes, tag_vocab_size=tag_vocab_size)
    model.eval()

    # ========== 维度检查 ==========
    with torch.no_grad():
        z_b = model.encode_behavior(batch)          # [B, D]
        print(f"[Check] encode_behavior -> {tuple(z_b.shape)}")
        assert z_b.ndim == 2 and z_b.shape[0] == B, "z_behavior 形状不正确"

        # 准备一些 tag_ids 做 encode_tag —— 使用词表名称映射，避免越界
        roles_no_pad  = [t for t in roles_tokens  if t not in ("PAD", "UNK")]
        traits_no_pad = [t for t in traits_tokens if t not in ("PAD", "UNK")]
        sample_tokens = []
        if roles_no_pad:
            sample_tokens.append(roles_no_pad[0])      # 任选一个角色
        if len(roles_no_pad) > 1:
            sample_tokens.append(roles_no_pad[1])      # 再取一个角色（若存在）
        if traits_no_pad:
            sample_tokens.append(traits_no_pad[0])     # 再取一个 trait（若存在）
        if not sample_tokens:
            sample_tokens = ["UNK"]                   # 兜底
        tag_ids = torch.tensor([token2id.get(t, token2id.get("UNK", 1)) for t in sample_tokens], dtype=torch.long)
        z_t = model.encode_tag(tag_ids)             # [N, D]
        print(f"[Check] encode_tag -> {tuple(z_t.shape)}")
        assert z_t.ndim == 2 and z_t.shape[1] == z_b.shape[1], "z_tag 维度应与 z_behavior 对齐"

    # ========== 数值检查 ==========
    assert not has_nan_or_inf(z_b), "z_behavior 存在 NaN/Inf"
    assert z_b.abs().sum().item() > 0, "z_behavior 全为 0？"
    mean = z_b.mean().item()
    var  = z_b.var(unbiased=False).item()
    print(f"[Check] z_behavior mean={mean:.4f}, var={var:.4f}")

    # ========== 一致性检查 ==========
    with torch.no_grad():
        z_b_again = model.encode_behavior(batch)
    delta = (z_b_again - z_b).abs().max().item()
    print(f"[Check] determinism max|Δ|={delta:.6g}")
    assert delta < 1e-6, "同一输入多次前向不一致（检查随机 Dropout 是否关闭、seed 是否固定）"

    # 换个输入（打乱 batch 内顺序或构造一个不同 batch），期望向量有差异
    # 这里简单打乱时间 mask（演示），更严谨的是从 loader 再取一批或构造另一份 mock
    # 但最小链路下，只要 cosine 不全是 1 即可
    def cosine(a, b, eps=1e-9):
        a = a / (a.norm(dim=1, keepdim=True) + eps)
        b = b / (b.norm(dim=1, keepdim=True) + eps)
        return (a * b).sum(dim=1)

    # 人为构造一份“不同输入”：把 action_idx 的一部分换成 UNK（1）
    batch_diff = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    if "action_idx" in batch_diff:
        batch_diff["action_idx"][:, :max(1, T//4)] = 1  # 前 1/4 时间步改成 UNK
    with torch.no_grad():
        z_b_diff = model.encode_behavior(batch_diff)
    cos = cosine(z_b, z_b_diff).mean().item()
    print(f"[Check] cosine(z_b, z_b_diff) = {cos:.4f}  （不应接近 1.0）")

    # ========== Mask 检查 ==========
    # 去掉 padding，再与带 mask 的结果对比（均值池化一致性）
    mask = batch["mask"].bool()          # [B,T]
    x = model.encode_behavior            # 这里我们重用模型的 encode 行为；严格做法是导出 encoder 中的 x 特征
    # 为最小链路，这里用“把 padding 处 action_idx 改成 PAD=0，且强制 mask=1”近似测试：
    batch_no_pad = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    for k, v in batch_no_pad.items():
        if torch.is_tensor(v) and v.ndim >= 2 and v.shape[:2] == mask.shape:
            if v.dtype == torch.long:
                v[~mask] = 0
            else:
                v[~mask] = 0.0
    batch_no_pad["mask"] = torch.ones_like(batch["mask"], dtype=torch.bool)

    with torch.no_grad():
        z_mask = model.encode_behavior(batch)          # 正常 mask
        z_full = model.encode_behavior(batch_no_pad)   # 去掉 padding 的近似版本
    diff = (z_mask - z_full).abs().max().item()
    print(f"[Check] mask consistency max|Δ|={diff:.6g}")

    print("\n✅ Sanity checks finished without assertion errors.")

if __name__ == "__main__":
    main()
