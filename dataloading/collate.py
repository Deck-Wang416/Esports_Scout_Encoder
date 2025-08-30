import torch

def collate_fn(samples):
    B = len(samples)
    T = len(samples[0].mask)

    def stack_1d(name, dtype=torch.long):
        data = [getattr(s, name) for s in samples]
        return torch.tensor(data, dtype=dtype)

    def stack_2d(name, dtype=torch.float):
        data = [getattr(s, name) for s in samples]
        return torch.tensor(data, dtype=dtype)

    batch = {
        "action_idx":      stack_1d("action_idx", torch.long),
        "loc_idx":         stack_1d("loc_idx", torch.long),
        "team_idx":        torch.tensor([s.team_idx for s in samples], dtype=torch.long),
        "timestamp_rel":   stack_1d("timestamp_rel", torch.float),
        "outcome_multi":   stack_2d("outcome_multi", torch.float),
        "impact_multi":    stack_2d("impact_multi", torch.float),
        "weapon_top1_idx": stack_1d("weapon_top1_idx", torch.long),
        "damage_sum":      stack_1d("damage_sum", torch.float),
        "damage_mean":     stack_1d("damage_mean", torch.float),
        "damage_max":      stack_1d("damage_max", torch.float),
        "is_lethal":       stack_1d("is_lethal", torch.float),
        "mask":            torch.tensor([s.mask for s in samples], dtype=torch.bool),
        "meta":            [s.meta for s in samples],
    }
    return batch
