from __future__ import annotations

import json
from typing import List, Dict, Any, Sequence
from .base_adapter import BaseAdapter, Sample
from collections import Counter, defaultdict


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
        K = self.k_multi

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
        dmg_mode = getattr(dmg_sum_cfg, "mode", "clip_minmax")
        dmg_min = getattr(dmg_sum_cfg, "min", 0.0)
        dmg_max = getattr(dmg_sum_cfg, "max", 1e9)

        samples: List[Sample] = []

        rounds = obj.get("rounds", [])
        for rnd in rounds:
            round_number = rnd.get("round_number", 0)

            players = rnd.get("players", [])
            if isinstance(players, dict):
                players = [players]

            for p in players:
                player_id = p.get("player_id", "unknown")
                team = str(p.get("team", "CT")).upper()
                team_idx = 0 if team == "CT" else 1

                # 过滤有效事件：timestamp 必须存在且非负
                traj = [e for e in p.get("trajectory", []) if "timestamp" in e and isinstance(e["timestamp"], (int, float)) and e["timestamp"] >= 0.0]  # 改动：添加 timestamp 验证

                # 按 timestamp 排序
                traj.sort(key=lambda e: e["timestamp"])

                # 按 T 大小切分非重叠窗口
                for win_start in range(0, len(traj), T):
                    evs = traj[win_start:win_start + T]
                    if not evs:
                        continue

                    L = len(evs)
                    pad = T - L
                    mask = [1] * L + [0] * pad  # 用于后续 BoolTensor

                    ts_vals = [e["timestamp"] for e in evs]
                    min_ts = min(ts_vals) if ts_vals else 0.0
                    ts_rel = [t - min_ts for t in ts_vals]

                    # 归一化
                    if ts_mode == "minmax":
                        ts_norm = [_norm_minmax(t, ts_min, ts_max) for t in ts_rel]
                    else:
                        ts_norm = ts_rel
                    ts_norm += [0.0] * pad

                    # 离散字段索引
                    action_idx = [_tok2id(act_vocab, str(e.get("action", ""))) for e in evs] + [0] * pad
                    loc_idx = [_tok2id(loc_vocab, str(e.get("location", ""))) for e in evs] + [0] * pad

                    # 多标签处理：去重、限制 K 个
                    outcome_multi: List[List[int]] = []
                    impact_multi: List[List[int]] = []

                    for e in evs:
                        res = e.get("result", {}) or {}
                        outs = list(dict.fromkeys(res.get("outcome", []) or []))[:K]
                        imps = list(dict.fromkeys(res.get("impact", []) or []))[:K]

                        o_vec = [0] * O
                        for tok in outs:
                            idx = _tok2id(out_vocab, str(tok))
                            if idx > 1:
                                o_vec[idx] = 1

                        i_vec = [0] * I
                        for tok in imps:
                            idx = _tok2id(imp_vocab, str(tok))
                            if idx > 1:
                                i_vec[idx] = 1

                        outcome_multi.append(o_vec)
                        impact_multi.append(i_vec)

                    outcome_multi += [[0] * O] * pad
                    impact_multi += [[0] * I] * pad

                    # 武器与伤害聚合（每事件）
                    weapon_top1_idx = []
                    dmg_sums = []
                    dmg_means = []
                    dmg_maxs = []
                    is_lethals = []

                    for e in evs:
                        res = e.get("result", {}) or {}
                        weapons = res.get("weapon", [])
                        if not isinstance(weapons, list):
                            weapons = [weapons] if weapons else []

                        damages = res.get("damage", [])
                        if not isinstance(damages, list):
                            damages = [damages] if damages is not None else []

                        damage_vals = []
                        for d in damages:
                            try:
                                damage_vals.append(float(d))
                            except (ValueError, TypeError):
                                pass

                        # 伤害聚合
                        d_sum = sum(damage_vals)
                        d_max = max(damage_vals) if damage_vals else 0.0
                        d_mean = d_sum / len(damage_vals) if damage_vals else 0.0
                        is_l = 1 if d_max >= 100.0 else 0
                        # 最大伤害 >= 100 判断 is_lethal

                        if dmg_mode == "clip_minmax":
                            d_sum = _clip(d_sum, dmg_min, dmg_max)
                            d_max = _clip(d_max, dmg_min, dmg_max)
                            d_mean = _clip(d_mean, dmg_min, dmg_max)

                        dmg_sums.append(d_sum)
                        dmg_means.append(d_mean)
                        dmg_maxs.append(d_max)
                        is_lethals.append(is_l)

                        # 武器 top1：如果有伤害配对则按伤害加权，否则按频次
                        if weapons:
                            if len(weapons) == len(damage_vals):
                                w_dmg = defaultdict(float)
                                for w, d in zip(weapons, damage_vals):
                                    w_dmg[str(w)] += d
                                top_w = max(w_dmg, key=w_dmg.get)
                            else:
                                cnt = Counter(str(w) for w in weapons)
                                top_w = cnt.most_common(1)[0][0] if cnt else ""
                            w_idx = _tok2id(wpn_vocab, top_w)
                        else:
                            w_idx = 1  # 无武器时使用 UNK
                        weapon_top1_idx.append(w_idx)

                    weapon_top1_idx += [0] * pad
                    dmg_sums += [0.0] * pad
                    dmg_means += [0.0] * pad
                    dmg_maxs += [0.0] * pad
                    is_lethals += [0] * pad

                    meta = {
                        "match_id": obj.get("match_id"),
                        "player_id": player_id,
                        "round_number": round_number,
                        "window_start_s": min_ts,
                    }

                    samples.append(Sample(
                        action_idx=action_idx,
                        loc_idx=loc_idx,
                        team_idx=team_idx,
                        timestamp_rel=ts_norm,
                        outcome_multi=outcome_multi,
                        impact_multi=impact_multi,
                        weapon_top1_idx=weapon_top1_idx,
                        damage_sum=dmg_sums,
                        damage_mean=dmg_means,
                        damage_max=dmg_maxs,
                        is_lethal=is_lethals,
                        mask=mask,
                        meta=meta,
                    ))

        return samples
