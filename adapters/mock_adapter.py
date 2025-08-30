import json
from typing import List
from .base_adapter import BaseAdapter, Sample

# --- helpers ---

def _tok2id(vocab_tokens, tok):
    """Map token to id with PAD=0, UNK=1 convention."""
    try:
        return vocab_tokens.index(tok)
    except ValueError:
        return 1

def _norm_minmax(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    v = (x - lo) / (hi - lo)
    # clamp to [0,1]
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v

def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

class MockAdapter(BaseAdapter):
    def __init__(self, vocab, norm, T: int, K_multi: int = 3):
        """
        vocab: Namespace-like object with fields .action/.location/.outcome/.impact/.weapon, each has .tokens: List[str]
        norm : Namespace-like object with sections like .timestamp.mode/.timestamp.min/.timestamp.max, etc.
        T    : fixed window length
        K_multi: cap for outcome/impact labels per timestep
        """
        self.vocab = vocab
        self.norm = norm
        self.T = T
        self.K_multi = K_multi

    # --- BaseAdapter API ---
    def parse_file(self, json_path: str) -> List[Sample]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self.parse_obj(data)

    def parse_obj(self, obj: dict) -> List[Sample]:
        T = self.T
        # vocab lists
        out_vocab = self.vocab.outcome.tokens
        imp_vocab = self.vocab.impact.tokens
        act_vocab = self.vocab.action.tokens
        loc_vocab = self.vocab.location.tokens
        wpn_vocab = self.vocab.weapon.tokens

        # normalization config (optional; fall back gracefully)
        ts_mode = getattr(getattr(self.norm, "timestamp", {}), "mode", None)
        ts_min  = getattr(getattr(self.norm, "timestamp", {}), "min", 0.0)
        ts_max  = getattr(getattr(self.norm, "timestamp", {}), "max", 1.0)

        dmg_cfg = getattr(self.norm, "damage_sum", None)
        dmg_lo  = getattr(dmg_cfg, "min", 0.0)
        dmg_hi  = getattr(dmg_cfg, "max", 1e9)

        samples = []
        for rnd in obj.get("rounds", []):
            for p in rnd.get("players", []):
                traj = sorted(p.get("trajectory", []), key=lambda e: e.get("timestamp", 0.0))
                # minimal path: single window per player, truncate/pad to T
                evs = traj[:T]
                L = len(evs)
                mask = [True]*L + [False]*(T-L)

                # timestamps → normalized if configured
                ts_vals = [e.get("timestamp", 0.0) for e in evs]
                if ts_mode == "minmax":
                    ts = [_norm_minmax(x, ts_min, ts_max) for x in ts_vals]
                else:
                    ts = ts_vals  # pass-through
                ts += [0.0]*(T-L)

                # discrete ids
                action_idx = [_tok2id(act_vocab, e.get("action", "")) for e in evs] + [0]*(T-L)
                loc_idx    = [_tok2id(loc_vocab, e.get("location", "")) for e in evs] + [0]*(T-L)
                team_idx   = 0 if p.get("team", "CT") == "CT" else 1

                # multi-label outcome/impact → cap K, dedupe, then multi-hot
                O = len(out_vocab)
                I = len(imp_vocab)
                outcome_multi = []
                impact_multi  = []
                for e in evs:
                    res = e.get("result", {})
                    outs = list(dict.fromkeys(res.get("outcome", [])))[: self.K_multi]
                    imps = list(dict.fromkeys(res.get("impact",  [])))[: self.K_multi]

                    o_vec = [0]*O
                    for tok in outs:
                        o_vec[_tok2id(out_vocab, tok)] = 1
                    i_vec = [0]*I
                    for tok in imps:
                        i_vec[_tok2id(imp_vocab, tok)] = 1

                    outcome_multi.append(o_vec)
                    impact_multi.append(i_vec)
                # pad to T
                outcome_multi += [[0]*O]*(T-L)
                impact_multi  += [[0]*I]*(T-L)

                # weapon top1 (mock: take first if present)
                weapon_top1_idx = [
                    _tok2id(wpn_vocab, (e.get("result", {}).get("weapon", [""]) or [""])[0])
                    for e in evs
                ] + [0]*(T-L)

                # damage aggregation (mock: single number list); clamp per normalization if available
                dmg_vals = [
                    (e.get("result", {}).get("damage", [0]) or [0])[0]
                    for e in evs
                ]
                dmg_vals = [_clip(float(v), dmg_lo, dmg_hi) for v in dmg_vals]

                dmg_sum   = dmg_vals + [0]*(T-L)
                dmg_mean  = dmg_vals + [0]*(T-L)
                dmg_max   = dmg_vals + [0]*(T-L)
                is_lethal = [1 if float(v) >= 100.0 else 0 for v in dmg_vals] + [0]*(T-L)

                meta = {
                    "match_id": obj.get("match_id"),
                    "player_id": p.get("player_id"),
                    "round_number": rnd.get("round_number"),
                }

                samples.append(Sample(
                    action_idx, loc_idx, team_idx, ts,
                    outcome_multi, impact_multi, weapon_top1_idx,
                    dmg_sum, dmg_mean, dmg_max, is_lethal, mask, meta
                ))
        return samples
