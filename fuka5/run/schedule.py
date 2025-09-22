"""
fuka5.run.schedule
------------------
Epoch scheduling, seeding, and cadence helpers for the simulator.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import time
import numpy as np


@dataclass
class TimeConfig:
    window_sec: float
    fs: float
    epochs: int
    on_blocks: List[Tuple[int, int]]  # inclusive start, exclusive end

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TimeConfig":
        return TimeConfig(
            window_sec=float(d["window_sec"]),
            fs=float(d["fs"]),
            epochs=int(d["epochs"]),
            on_blocks=[(int(a), int(b)) for a, b in d.get("on_blocks", [])],
        )

    def on_flag(self, epoch: int) -> bool:
        for s, e in self.on_blocks:
            if s <= epoch < e:
                return True
        return False


@dataclass
class Cadence:
    edges_flush_every: int = 5
    metrics_flush_every: int = 5
    volume_every: int = 10
    checkpoint_every: int = 20

    @staticmethod
    def default() -> "Cadence":
        return Cadence()


def seeds_from_int(seed: int | None) -> Dict[str, int]:
    """Derive sub-seeds from a base integer seed."""
    if seed is None:
        seed = int(time.time()) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    return {
        "base": seed,
        "world": int(rng.integers(0, 2**31-1)),
        "graph": int(rng.integers(0, 2**31-1)),
        "sources": int(rng.integers(0, 2**31-1)),
        "substrate": int(rng.integers(0, 2**31-1)),
        "runner": int(rng.integers(0, 2**31-1)),
    }