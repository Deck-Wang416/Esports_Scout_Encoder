import torch

def collate_fn(samples):
    """Stack a list[Sample] into a batch dict of torch.Tensors."""
    assert len(samples) > 0, "Empty batch."
    T = len(samples[0].mask)
    # all sequences should share the same T
    assert all(len(s.mask) == T for s in samples), "Inconsistent sequence length T in samples."

    def to_tensor(name, dtype):
        # Each field is a Python list (length T or [T, V]); stack into [B, ...]
        data = [getattr(s, name) for s in samples]
        return torch.tensor(data, dtype=dtype)

    batch = {
        # [B, T] long
        "action_idx":      to_tensor("action_idx",      torch.long),
        "loc_idx":         to_tensor("loc_idx",         torch.long),
        "weapon_top1_idx": to_tensor("weapon_top1_idx", torch.long),

        # [B] long (broadcast later if needed)
        "team_idx": torch.tensor([s.team_idx for s in samples], dtype=torch.long),

        # [B, T] float
        "timestamp_rel":   to_tensor("timestamp_rel",   torch.float),
        "damage_sum":      to_tensor("damage_sum",      torch.float),
        "damage_mean":     to_tensor("damage_mean",     torch.float),
        "damage_max":      to_tensor("damage_max",      torch.float),
        "is_lethal":       to_tensor("is_lethal",       torch.float),

        # [B, T, V] float (multi-hot)
        "outcome_multi":   to_tensor("outcome_multi",   torch.float),
        "impact_multi":    to_tensor("impact_multi",    torch.float),

        # [B, T] bool
        "mask": torch.tensor([s.mask for s in samples], dtype=torch.bool),

        # list of dicts (logging only)
        "meta": [s.meta for s in samples],
    }
    return batch
