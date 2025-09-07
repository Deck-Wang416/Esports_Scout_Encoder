from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Mapping, Optional
from abc import ABC, abstractmethod


@dataclass(frozen=True)
class Sample:
    """Minimal normalized sample for one player window (length T)."""
    action_idx: List[int]                 # [T]
    loc_idx: List[int]                    # [T]
    team_idx: int                         # scalar (broadcast to [T] later if needed)
    timestamp_rel: List[float]            # [T] (optionally normalized)
    outcome_multi: List[List[int]]        # [T, |V_outcome|] multi-hot (0/1)
    impact_multi: List[List[int]]         # [T, |V_impact|] multi-hot (0/1)
    weapon_top1_idx: List[int]            # [T]
    damage_sum: List[float]               # [T]
    damage_mean: List[float]              # [T]
    damage_max: List[float]               # [T]
    is_lethal: List[int]                  # [T] 0/1
    mask: List[int]                       # [T] Boolean mask (1=valid, 0=padding)
    meta: Dict[str, Any]                  # logging only (not used by the model)


class BaseAdapter(ABC):
    """
    Abstract base class for adapters that parse raw JSON into `Sample` objects.

    - Fixed window length: `T`.
    - Multi-label cap per event: `k_multi`.
    - Vocab indexing conventions: PAD=0, UNK=1.

    Implementations must provide `parse_file` and `parse_obj`.
    """

    def __init__(
        self,
        vocab_cfg: Mapping[str, Any],
        norm_cfg: Optional[Mapping[str, Any]],
        T: int,
        k_multi: int = 3,
    ) -> None:
        self.vocab = vocab_cfg            # expects vocab.*.tokens etc.
        self.norm = norm_cfg              # may be None if not provided
        self.T = T                        # fixed window length
        self.k_multi = k_multi            # cap for multi-label per event

    @abstractmethod
    def parse_file(self, json_path: str) -> List[Sample]:
        """Read a JSON file and return a list of Samples."""
        raise NotImplementedError

    @abstractmethod
    def parse_obj(self, obj: Dict[str, Any]) -> List[Sample]:
        """Parse an in-memory JSON object and return a list of Samples."""
        raise NotImplementedError
