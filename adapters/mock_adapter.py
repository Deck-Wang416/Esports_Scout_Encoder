from __future__ import annotations

import json
from typing import List, Dict, Any, Sequence
from .base_adapter import BaseAdapter, Sample


# --- helpers --------------------------------------------------------------

def _tok2id(vocab_tokens: Sequence[str], tok: str) -> int:
    """Map token to id with PAD=0, UNK=1 convention."""
    try:
        return vocab_tokens.index(tok)
    except ValueError:
        return 1  # UNK

def _norm_minmax(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    v = (x - lo) / (hi - lo)
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


# --- adapter --------------------------------------------------------------

class MockAdapter(BaseAdapter):
    """
    Minimal JSON→Sample adapter for the mock demo.
    Expects:
      - self.vocab.action/location/outcome/impact/weapon.tokens
      - self.norm may be None or contain timestamp/damage_* ranges
    """

    def __init__(self, vocab_cfg, norm_cfg, T: int, k_multi: int = 3) -> None:
        super().__init__(vocab_cfg, norm_cfg, T, k_multi)

    # --- BaseAdapter API ---

    def parse_file(self, json_path: str) -> List[Sample]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self.parse_obj(data)

    def parse_obj(self, obj: Dict[str, Any]) -> List[Sample]:
        T = self.T

        # vocab lists (PAD, UNK at 0/1)
        act_vocab = getattr(self.vocab.action, "tokens", [])
        loc_vocab = getattr(self.vocab.location, "tokens", [])
        out_vocab = getattr(self.vocab.outcome, "tokens", [])
        imp_vocab = getattr(self.vocab.impact, "tokens", [])
        wpn_vocab = getattr(self.vocab.weapon, "tokens", [])

        O, I = len(out_vocab), len(imp_vocab)

        # normalization cfg (optional)
        ts_cfg = getattr(self.norm, "timestamp", None) if self.norm is not None else None
        ts_mode = getattr(ts_cfg, "mode", None)
        ts_min  = getattr(ts_cfg, "min", 0.0)
        ts_max  = getattr(ts_cfg, "max", 1.0)

        dmg_sum_cfg = getattr(self.norm, "damage_sum", None) if self.norm is not None else None
        dmg_lo = getattr(dmg_sum_cfg, "min", 0.0)
        dmg_hi = getattr(dmg_sum_cfg, "max", 1e9)

        samples: List[Sample] = []

        for rnd in obj.get("rounds", []):
            for p in rnd.get("players", []):
                traj = sorted(p.get("trajectory", []), key=lambda e: e.get("timestamp", 0.0))

                # single window: truncate/pad to T
                evs = traj[:T]
                L = len(evs)
                pad = T - L
                mask = [1] * L + [0] * pad  # int mask for downstream BoolTensor

                # timestamps
                ts_vals = [float(e.get("timestamp", 0.0)) for e in evs]
                if ts_mode == "minmax":
                    ts = [_norm_minmax(x, ts_min, ts_max) for x in ts_vals]
                else:
                    ts = ts_vals
                ts += [0.0] * pad

                # discrete ids
                action_idx = [_tok2id(act_vocab, str(e.get("action", ""))) for e in evs] + [0] * pad
                loc_idx    = [_tok2id(loc_vocab, str(e.get("location", ""))) for e in evs] + [0] * pad
                team_idx   = 0 if str(p.get("team", "CT")).upper() == "CT" else 1

                # multi-label → cap K, dedupe, then multi-hot
                outcome_multi: List[List[int]] = []
                impact_multi:  List[List[int]] = []

                for e in evs:
                    res = e.get("result", {}) or {}
                    outs = list(dict.fromkeys(res.get("outcome", []) or []))[: self.k_multi]
                    imps = list(dict.fromkeys(res.get("impact",  []) or []))[: self.k_multi]

                    o_vec = [0] * O
                    for tok in outs:
                        idx = _tok2id(out_vocab, str(tok))
                        if 0 <= idx < O:
                            o_vec[idx] = 1

                    i_vec = [0] * I
                    for tok in imps:
                        idx = _tok2id(imp_vocab, str(tok))
                        if 0 <= idx < I:
                            i_vec[idx] = 1

                    outcome_multi.append(o_vec)
                    impact_multi.append(i_vec)

                outcome_multi += [[0] * O] * pad
                impact_multi  += [[0] * I] * pad

                # weapon top1 (mock: take first if present)
                weapon_top1_idx = [
                    _tok2id(wpn_vocab, (e.get("result", {}).get("weapon", [""]) or [""])[0])
                    for e in evs
                ] + [0] * pad

                # ---- damage aggregation (robust extraction) ----
                def _first_num(x):
                    # x can be int/float, list, None, etc.
                    if isinstance(x, (int, float)):
                        return float(x)
                    if isinstance(x, list):
                        return float(x[0]) if x else 0.0
                    return 0.0

                dmg_vals = []
                for e in evs:
                    res = e.get("result", {}) or {}
                    dmg_raw = res.get("damage", 0)       # may be int/float/list/empty
                    v = _first_num(dmg_raw)
                    # optional clamp by normalization cfg (if provided)
                    v = _clip(v, dmg_lo, dmg_hi)
                    dmg_vals.append(v)

                # keep the simple baseline: use the same scalar for sum/mean/max in mock
                dmg_sum   = dmg_vals + [0]*(T-L)
                dmg_mean  = dmg_vals + [0]*(T-L)
                dmg_max   = dmg_vals + [0]*(T-L)
                is_lethal = [1 if v >= 100.0 else 0 for v in dmg_vals] + [0]*(T-L)

                meta = {
                    "match_id": obj.get("match_id"),
                    "player_id": p.get("player_id"),
                    "round_number": rnd.get("round_number"),
                }

                samples.append(Sample(
                    action_idx=action_idx,
                    loc_idx=loc_idx,
                    team_idx=team_idx,
                    timestamp_rel=ts,
                    outcome_multi=outcome_multi,
                    impact_multi=impact_multi,
                    weapon_top1_idx=weapon_top1_idx,
                    damage_sum=dmg_sum,
                    damage_mean=dmg_mean,
                    damage_max=dmg_max,
                    is_lethal=is_lethal,
                    mask=mask,
                    meta=meta,
                ))

        return samples
