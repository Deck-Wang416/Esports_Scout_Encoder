import torch
import torch.nn as nn
from .positional import SinCosPosEnc

class Encoder(nn.Module):
    def __init__(self, cfg, vocab_sizes, tag_vocab_size):
        super().__init__()
        H, D = cfg.model.hidden_dim, cfg.model.output_dim

        # 1) Embedding / Projection → H
        self.action_emb = nn.Embedding(vocab_sizes["action"], H)
        self.loc_emb    = nn.Embedding(vocab_sizes["location"], H)
        self.weapon_emb = nn.Embedding(vocab_sizes["weapon"], H)
        self.team_emb   = nn.Embedding(2, H)

        self.outcome_proj = nn.Linear(vocab_sizes["outcome"], H)
        self.impact_proj  = nn.Linear(vocab_sizes["impact"],  H)

        self.timestamp_proj = nn.Linear(1, H)
        self.damage_proj    = nn.Linear(4, H)

        # 2) Fusion: concat -> Linear to H; then Dropout+LayerNorm
        self.fuse    = nn.Linear(3 * H, H)   # x (H) + mhot (H) + num (H) -> 3H
        self.fuse_ln = nn.LayerNorm(H)
        self.fuse_do = nn.Dropout(cfg.model.dropout)

        # 3) Positional Encoding
        self.pos_enc = SinCosPosEnc(H)

        # 4) Transformer (batch_first=True)
        layer = nn.TransformerEncoderLayer(
            d_model=H,
            nhead=cfg.model.nhead,
            dropout=cfg.model.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.model.num_layers)

        # 5) Heads
        self.behavior_head = nn.Linear(H, D)
        self.tag_emb  = nn.Embedding(tag_vocab_size, H)
        self.tag_head = nn.Linear(H, D)

    def encode_behavior(self, batch):
        B, T = batch["action_idx"].shape

        x = (
            self.action_emb(batch["action_idx"])
            + self.loc_emb(batch["loc_idx"])
            + self.weapon_emb(batch["weapon_top1_idx"])
            + self.team_emb(batch["team_idx"].unsqueeze(1))  # [B,1] -> broadcast to [B,T]
        )

        mhot = (
            self.outcome_proj(batch["outcome_multi"])
            + self.impact_proj(batch["impact_multi"])
        )
        num = (
            self.timestamp_proj(batch["timestamp_rel"].unsqueeze(-1))
            + self.damage_proj(torch.stack(
                [batch["damage_sum"], batch["damage_mean"], batch["damage_max"], batch["is_lethal"]],
                dim=-1
            ))
        )

        x_cat = torch.cat([x, mhot, num], dim=-1)     # [B,T,3H]
        x = self.fuse_do(self.fuse_ln(self.fuse(x_cat)))
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=~batch["mask"].bool())
        z = self._masked_mean(x, batch["mask"])
        return self.behavior_head(z)

    def encode_tag(self, tag_ids):
        # Accept List[int] or LongTensor
        if isinstance(tag_ids, list):
            tag_ids = torch.tensor(tag_ids, dtype=torch.long, device=self.tag_emb.weight.device)
        else:
            tag_ids = tag_ids.to(dtype=torch.long, device=self.tag_emb.weight.device)

        # Out-of-range -> UNK
        unk = getattr(self, "tag_meta", {}).get("unk_idx", 1)
        vocab_size = self.tag_emb.num_embeddings
        bad = (tag_ids < 0) | (tag_ids >= vocab_size)
        if bad.any():
            tag_ids = tag_ids.clone()
            tag_ids[bad] = unk

        h = self.tag_emb(tag_ids)   # [N, H]
        return self.tag_head(h)     # [N, D]

    @staticmethod
    def _masked_mean(x, mask):
        m = mask.unsqueeze(-1).float()
        denom = m.sum(dim=1).clamp_min(1.0)
        return (x * m).sum(dim=1) / denom
