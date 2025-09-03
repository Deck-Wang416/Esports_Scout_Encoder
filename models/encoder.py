# models/encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional import SinCosPosEnc

def _proj_block(in_dim: int, H: int, p: float):
    """Linear -> LayerNorm -> Dropout"""
    return nn.Sequential(
        nn.Linear(in_dim, H),
        nn.LayerNorm(H),
        nn.Dropout(p),
    )

class MeanPool(nn.Module):
    """Mask-aware mean over time: [B,T,H] -> [B,H]."""
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        denom = m.sum(dim=1).clamp_min(1.0)
        return (x * m).sum(dim=1) / denom

class MaxPool(nn.Module):
    """Mask-aware max over time: padding positions are -inf."""
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        neg_inf = torch.finfo(x.dtype).min
        m = mask.unsqueeze(-1)
        x_masked = x.masked_fill(~m, neg_inf)
        return x_masked.max(dim=1).values

class AttnPool(nn.Module):
    """Single-head additive attention pooling with mask: [B,T,H] -> [B,H]."""
    def __init__(self, H: int, attn_dim: int = 128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(H, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x:[B,T,H], mask:[B,T]
        s = self.score(x).squeeze(-1)  # [B,T]
        s = s.masked_fill(~mask, float('-inf'))
        a = F.softmax(s, dim=1).unsqueeze(-1)  # [B,T,1]
        return (a * x).sum(dim=1)

class Encoder(nn.Module):
    """
    Behavior encoder (sequence -> z_behavior) + Tag encoder (ids -> z_tag).
    - Each enabled input channel is projected to H, then concatenated and fused back to H.
    - Sequence modeling via TransformerEncoder.
    - Mask-aware mean pooling -> [B, H] -> head -> [B, D].
    """
    def __init__(self, cfg, vocab_sizes, tag_vocab_size):
        super().__init__()
        self.cfg = cfg
        H, D = cfg.model.hidden_dim, cfg.model.output_dim
        p = cfg.model.dropout

        # == 1) Per-channel encoders (project to H) ==
        use = cfg.inputs

        # Discrete ids -> Embedding(H)
        if use.use_action:
            self.action_emb = nn.Embedding(vocab_sizes["action"], H, padding_idx=getattr(cfg.runtime, "pad_idx", 0))
        if use.use_location:
            self.loc_emb    = nn.Embedding(vocab_sizes["location"], H, padding_idx=getattr(cfg.runtime, "pad_idx", 0))
        if use.use_weapon:
            self.weapon_emb = nn.Embedding(vocab_sizes["weapon"], H, padding_idx=getattr(cfg.runtime, "pad_idx", 0))
        if use.use_team:
            self.team_emb   = nn.Embedding(2, H)  # 0=CT,1=T

        # Multi-hot -> Linear(H)
        if use.use_outcome:
            self.outcome_proj = _proj_block(vocab_sizes["outcome"], H, p)
        if use.use_impact:
            self.impact_proj  = _proj_block(vocab_sizes["impact"],  H, p)

        # Numeric -> Linear(H)
        if use.use_timestamp:
            self.timestamp_proj = _proj_block(1, H, p)

        # damage_* 统一打包成 4 维（sum/mean/max/is_lethal），按开关自动拼装
        self.damage_use = [use.use_damage_sum, use.use_damage_mean, use.use_damage_max, use.use_is_lethal]
        self.num_damage_feats = sum(int(x) for x in self.damage_use)
        if self.num_damage_feats > 0:
            self.damage_proj = _proj_block(self.num_damage_feats, H, p)

        # 统计开启通道数，确定 F
        self.enabled_channels = []
        for name, flag in [
            ("action",   use.use_action),
            ("location", use.use_location),
            ("weapon",   use.use_weapon),
            ("team",     use.use_team),
            ("outcome",  use.use_outcome),
            ("impact",   use.use_impact),
            ("timestamp",use.use_timestamp),
            ("damage",   self.num_damage_feats > 0),
        ]:
            if flag:
                self.enabled_channels.append(name)
        assert len(self.enabled_channels) > 0, "No input channels enabled."

        F = len(self.enabled_channels) * H

        # == 2) Fusion: concat -> Linear(F->H) -> LN -> Dropout ==
        self.fuse = nn.Linear(F, H)
        self.fuse_ln = nn.LayerNorm(H)
        self.fuse_do = nn.Dropout(p)

        # == 3) Positional encoding ==
        self.pos_enc = SinCosPosEnc(H)

        # == 4) Transformer (batch_first=True) ==
        layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=cfg.model.nhead, dropout=p, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.model.num_layers)

        # == 5) Pooling ==
        pool_type = getattr(cfg.pooling, 'type', 'mean').lower()
        if pool_type == 'mean':
            self.pool = MeanPool()
        elif pool_type == 'max':
            self.pool = MaxPool()
        elif pool_type == 'attn':
            attn_dim = getattr(cfg.pooling, 'attn_dim', 128)
            self.pool = AttnPool(H, attn_dim)
        else:
            raise ValueError(f"Unknown pooling.type: {pool_type}")

        # == 6) Heads ==
        self.behavior_head = nn.Linear(H, D)

        # Tag encoder: ids -> H -> D
        self.tag_emb  = nn.Embedding(tag_vocab_size, H)
        self.tag_head = nn.Linear(H, D)
        # meta（可选，用于 encode_tag 兜底越界）
        self.tag_meta = {"unk_idx": getattr(cfg.runtime, "unk_idx", 1)}

    # ---------- Behavior ----------
    def encode_behavior(self, batch):
        """
        batch keys:
          action_idx [B,T] long
          loc_idx    [B,T] long
          team_idx   [B]   long (broadcast to [B,T])
          timestamp_rel [B,T] float
          outcome_multi [B,T,Vo]
          impact_multi  [B,T,Vi]
          weapon_top1_idx [B,T] long
          damage_* [B,T] float
          mask [B,T] bool
        """
        B, T = batch["action_idx"].shape
        mask = batch["mask"].bool()  # [B,T]

        feats = []

        # Discrete channels -> [B,T,H]
        if hasattr(self, "action_emb"):
            feats.append(self.action_emb(batch["action_idx"]))
        if hasattr(self, "loc_emb"):
            feats.append(self.loc_emb(batch["loc_idx"]))
        if hasattr(self, "weapon_emb"):
            feats.append(self.weapon_emb(batch["weapon_top1_idx"]))
        if hasattr(self, "team_emb"):
            team = batch["team_idx"].unsqueeze(1).expand(-1, T)  # [B,T]
            feats.append(self.team_emb(team))

        # Multi-hot -> H
        if hasattr(self, "outcome_proj"):
            feats.append(self.outcome_proj(batch["outcome_multi"]))
        if hasattr(self, "impact_proj"):
            feats.append(self.impact_proj(batch["impact_multi"]))

        # Numeric -> H
        if hasattr(self, "timestamp_proj"):
            feats.append(self.timestamp_proj(batch["timestamp_rel"].unsqueeze(-1)))  # [B,T,1] -> H

        if hasattr(self, "damage_proj") and self.num_damage_feats > 0:
            dmg_tensors = []
            # 顺序固定（与 damage_use 对齐）
            names = ["damage_sum", "damage_mean", "damage_max", "is_lethal"]
            for name, on in zip(names, self.damage_use):
                if on:
                    dmg_tensors.append(batch[name])
            dmg = torch.stack(dmg_tensors, dim=-1)  # [B,T,num_damage_feats]
            feats.append(self.damage_proj(dmg))     # -> [B,T,H]

        # Concatenate all features (including team) for fusion
        x_cat = torch.cat(feats, dim=-1)  # [B,T,F]

        # Compute presence from raw inputs (avoid projection biases)
        eps = 1e-6
        present = torch.zeros((B, T), dtype=torch.bool, device=x_cat.device)
        # Discrete content channels
        if hasattr(self, "action_emb"):
            present |= (batch["action_idx"] > 0)
        if hasattr(self, "loc_emb"):
            present |= (batch["loc_idx"] > 0)
        if hasattr(self, "weapon_emb"):
            present |= (batch["weapon_top1_idx"] > 0)
        # Multi-hot content channels
        if hasattr(self, "outcome_proj"):
            present |= (batch["outcome_multi"].abs().sum(dim=-1) > 0)
        if hasattr(self, "impact_proj"):
            present |= (batch["impact_multi"].abs().sum(dim=-1) > 0)
        # Numeric content channels
        if hasattr(self, "timestamp_proj"):
            present |= (batch["timestamp_rel"].abs() > eps)
        if hasattr(self, "damage_proj") and self.num_damage_feats > 0:
            dmg_present = torch.zeros((B, T), dtype=torch.bool, device=x_cat.device)
            names = ["damage_sum", "damage_mean", "damage_max", "is_lethal"]
            for name, on in zip(names, self.damage_use):
                if on:
                    v = batch[name]
                    if v.dtype.is_floating_point:
                        dmg_present |= (v.abs() > eps)
                    else:
                        dmg_present |= (v != 0)
            present |= dmg_present

        # Effective mask: only positions marked valid by upstream mask AND with any real content
        m_eff = mask & present

        x = self.fuse_do(self.fuse_ln(self.fuse(x_cat)))  # [B,T,H]

        # If all positions are padding, skip Transformer to avoid NestedTensor crash.
        if not m_eff.any():
            z = x.new_zeros((B, x.size(-1)))       # [B,H]
            return self.behavior_head(z)

        x = self.pos_enc(x)
        # Zero-out invalid (padded) positions explicitly to prevent leakage.
        x = x.masked_fill((~m_eff).unsqueeze(-1), 0.0)
        x = self.encoder(x, src_key_padding_mask=~m_eff)

        z = self.pool(x, m_eff)                    # [B,H]
        return self.behavior_head(z)               # [B,D]

    # ---------- Tag ----------
    def encode_tag(self, tag_ids):
        # accepts List[int] or LongTensor
        if isinstance(tag_ids, list):
            tag_ids = torch.tensor(tag_ids, dtype=torch.long, device=self.tag_emb.weight.device)
        else:
            tag_ids = tag_ids.to(dtype=torch.long, device=self.tag_emb.weight.device)

        # out-of-range -> UNK
        unk = self.tag_meta.get("unk_idx", 1)
        vocab_size = self.tag_emb.num_embeddings
        bad = (tag_ids < 0) | (tag_ids >= vocab_size)
        if bad.any():
            tag_ids = tag_ids.clone()
            tag_ids[bad] = unk

        h = self.tag_emb(tag_ids)   # [N,H]
        return self.tag_head(h)     # [N,D]
