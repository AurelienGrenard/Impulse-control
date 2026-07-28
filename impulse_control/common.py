"""Objects shared by the dividend and harvesting experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OptConfig:
    """Configure the randomized impulse search."""
    n_global: int = 10_000
    n_global_batch: int = 256
    min_rel_impulse: float = 0.4


@dataclass
class TimeGridConfig:
    """Configure the coarse decision grid and fine Euler step."""
    T: float = 2.0
    K: int = 40
    dt_fine: float = 0.01
