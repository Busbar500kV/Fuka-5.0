"""
fuka5.core.sources
------------------
Multi-source registry and spatial capacitive coupling to graph nodes.

Responsibilities
* Parse a JSON-like config:
    {
      "sources":[ {"name":"s1","loc":[x,y,z],"tones":[{"f":..,"A":..,"phi_deg":..}, ...]}, ...],
      "coupling": {"base_C": 1e-12, "range": 6.0}
    }
* Provide per-frequency complex source amplitudes.
* Build tiny capacitive taps from each source to nearby graph nodes, scaled by distance
  and local epsilon if desired (kept simple here; epsilon scaling can be folded in graph).
* Expose a compact structure for the physics solver:
    - tones: dict[freq_hz] -> list of SourceDrive
    - taps:  dict[source_name] -> list of (node_id, C_couple)

Notes
- Phases are stored in radians; complex amplitude = A * exp(j*phi).
- The physics layer will superpose contributions from all sources at each frequency.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import numpy as np


@dataclass
class Tone:
    f: float              # Hz
    A: float              # amplitude (arbitrary units)
    phi_rad: float        # radians

@dataclass
class SourceDef:
    name: str
    loc: Tuple[float, float, float]  # voxel coordinates (same space as graph/world)
    tones: List[Tone]

@dataclass
class SourceDrive:
    name: str
    f: float
    amp_complex: complex


class Sources:
    def __init__(self, defs: List[SourceDef], base_C: float = 1e-12, rng: np.random.Generator | None = None):
        self.defs = defs
        self.base_C = float(base_C)
        self.rng = rng or np.random.default_rng(0)

        # Map: freq -> list of SourceDrive
        self.freq_to_drives: Dict[float, List[SourceDrive]] = {}
        for sd in defs:
            for t in sd.tones:
                self.freq_to_drives.setdefault(float(t.f), []).append(
                    SourceDrive(name=sd.name, f=float(t.f), amp_complex=t.A * np.exp(1j * t.phi_rad))
                )

        # Taps are built per graph after nodes are known
        self.taps: Dict[str, List[Tuple[int, float]]] = {}  # name -> [(node_id, C_couple), ...]

    # ------------- Builders -------------

    @staticmethod
    def from_config_dict(d: Dict[str, Any]) -> "Sources":
        defs: List[SourceDef] = []
        for sd in d.get("sources", []):
            tones = [Tone(f=float(t["f"]), A=float(t["A"]), phi_rad=np.deg2rad(float(t.get("phi_deg", 0.0))))
                     for t in sd.get("tones", [])]
            defs.append(SourceDef(name=str(sd["name"]), loc=tuple(sd["loc"]), tones=tones))
        base_C = float(d.get("coupling", {}).get("base_C", 1e-12))
        return Sources(defs, base_C=base_C)

    def build_taps_to_graph(self, node_positions: np.ndarray, coupling_cfg: Dict[str, Any]) -> None:
        """
        Build small capacitive couplings from each source to nearby nodes.
        C_couple = base_C * exp(-(dist / range)^2)
        """
        r = float(coupling_cfg.get("range", 6.0))
        inv_r2 = 1.0 / max(1e-6, r * r)
        taps: Dict[str, List[Tuple[int, float]]] = {}

        for sd in self.defs:
            src = np.array(sd.loc, dtype=np.float32)
            d2 = np.sum((node_positions - src[None, :]) ** 2, axis=1)
            # keep nodes within ~3*r (numerically small beyond)
            keep = d2 < (9.0 * r * r)
            idxs = np.where(keep)[0]
            if idxs.size == 0:
                taps[sd.name] = []
                continue
            Cvals = self.base_C * np.exp(-d2[idxs] * inv_r2)
            taps[sd.name] = [(int(i), float(c)) for i, c in zip(idxs, Cvals)]

        self.taps = taps

    # ------------- Queries -------------

    def frequencies(self) -> List[float]:
        """List of all unique source frequencies."""
        fs = list(self.freq_to_drives.keys())
        fs.sort()
        return fs

    def drives_at(self, f: float) -> List[SourceDrive]:
        """Returns the list of SourceDrive at frequency f."""
        return self.freq_to_drives.get(float(f), [])

    def taps_for(self, name: str) -> List[Tuple[int, float]]:
        """Returns list of (node_id, C_couple) for source `name`."""
        return self.taps.get(name, [])