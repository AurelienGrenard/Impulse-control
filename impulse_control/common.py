"""Objects shared by the dividend and harvesting experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class OptConfig:
    """Configure the randomized impulse search."""
    n_global: int = 10_000
    n_global_batch: int = 256
    # ``None`` selects the publication rule: test every nonempty coordinate
    # mask of the best randomized vector candidate.
    min_rel_impulse: Optional[float] = None


@dataclass
class TimeGridConfig:
    """Configure the coarse decision grid and fine Euler step."""
    T: float = 2.0
    K: int = 40
    dt_fine: float = 0.01
