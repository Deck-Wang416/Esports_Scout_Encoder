from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Sample:
    action_idx: list
    loc_idx: list
    team_idx: int
    timestamp_rel: list
    outcome_multi: list      # [T, |V_outcome|]
    impact_multi: list       # [T, |V_impact|]
    weapon_top1_idx: list
    damage_sum: list
    damage_mean: list
    damage_max: list
    is_lethal: list
    mask: list               # [T] bool/int
    meta: Dict[str, Any]

class BaseAdapter:
    def __init__(self, vocab_cfg: Dict, norm_cfg: Dict, T: int, K_multi: int = 3):
        self.vocab = vocab_cfg
        self.norm  = norm_cfg
        self.T     = T
        self.K     = K_multi

    def parse_file(self, json_path: str) -> List[Sample]:
        raise NotImplementedError

    def parse_obj(self, obj: dict) -> List[Sample]:
        raise NotImplementedError
