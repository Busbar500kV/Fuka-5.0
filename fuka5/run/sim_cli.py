from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timezone

import numpy as np

from fuka5.core.world import make_world
from fuka5.core.graph import build_graph
from fuka5.core.sources import make_sources
from fuka5.substrate.updates import run_epoch

from fuka5.io.writers import (
    save_volume_npz,
    ShardWriter,
    write_manifest,
    write_checkpoint,
)
from fuka5.io.manifest import build_manifest, validate_manifest


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def _load_cfg(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _resolve_runs_root(cfg: Dict[str, Any]) -> Path:
    # Env overrides first, then cfg, then default
    base = os.getenv("F5_LOCAL_RUNS_DIR") or cfg.get("runs_dir") or "/home/busbar/fuka-runs"
    prefix = os.getenv("F5_RUNS_PREFIX") or cfg.get("runs_prefix") or "runs"
    root = Path(base) / prefix
    root.mkdir(parents=True, exist_ok=True)
    return root

def _runs_base_only(cfg: Dict[str, Any]) -> Path:
    """Return just the base directory (without runs_prefix)."""
    base = os.getenv("F5_LOCAL_RUNS_DIR") or cfg.get("runs_dir") or "/home/busbar/fuka-runs"
    return Path(base)

def _pick_run_id(run_id: Optional[str]) -> str:
    return run_id or f"FUKA_5_0_{_utc_stamp()}"

def _summarize_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Only keep a small subset to place in manifest
    keys = [
        "grid_shape", "seed", "rho_blur",
        "graph_stride", "graph_knn", "graph_max_dist",
        "edge_decay", "edge_eps_gain",
        "num_sources", "dt", "epochs",
    ]
    return {k: cfg[k] for k in keys if k in cfg}

def _default_cfg() -> Dict[str, Any]:
    return {
        "grid_shape": [64, 64, 64],
        "seed": 1234,
        "rho_blur": 2.0,
        "graph_stride": 6,
        "graph_knn": 6,
        "graph_max_dist": 24.0,
        "edge_decay": 0.12,
        "edge_eps_gain": 0.25,
        "num_sources": 12,
        "dt": 0.05,
        "epochs": 24,
        "flush_every": 4,
        "save_every": 1,   # save an NPZ every epoch (tweakable)
        "manifest_every": 4,
        "checkpoint_every": 4,
        "runs_dir": "${F5_LOCAL_RUNS_DIR}",
        "runs_prefix": "${F5_RUNS_PREFIX}",
    }

# ---------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------

def run_sim(config_path: Optional[str], run_id_arg: Optional[str]) -> str:
    # Load config with defaults
    cfg = _default_cfg()
    cfg.update(_load_cfg(config_path))

    runs_root = _resolve_runs_root(cfg)   # /home/busbar/fuka-runs/runs
    runs_base = _runs_base_only(cfg)      # /home/busbar/fuka-runs
    run_id = _pick_run_id(run_id_arg)
    run_dir = runs_root / run_id
    (run_dir / "volumes").mkdir(parents=True, exist_ok=True)
    (run_dir / "shards").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Build world, graph, sources
    world = make_world(cfg)
    graph = build_graph(cfg, world)
    sources = make_sources(cfg, world)

    # Writers (IMPORTANT: pass base dir, not runs_root)
    gcp_cfg = {
        "runs_dir": str(runs_base),   # base only
        "runs_prefix": "runs",
        "storage": "local",
    }
    edges_writer = ShardWriter(gcp_cfg, run_id, kind="edges", local_dir=str(run_dir), flush_every=int(cfg.get("flush_every", 4)))
    metrics_writer = ShardWriter(gcp_cfg, run_id, kind="metrics", local_dir=str(run_dir), flush_every=int(cfg.get("flush_every", 4)))

    # Manifest skeleton
    versions = {"fuka": "5.0"}
    cadence = {"save_every": cfg.get("save_every", 1)}
    seeds = {"seed": int(cfg.get("seed", 1234))}
    manifest = build_manifest(
        run_id=run_id,
        configs=_summarize_cfg(cfg),
        summaries={"status": "initializing"},
        cadence=cadence,
        versions=versions,
        seeds=seeds,
    )
    validate_manifest(manifest)

    # Write initial manifest at run start
    write_manifest(gcp_cfg, run_id, manifest)

    # Epoch loop
    epochs = int(cfg.get("epochs", 24))
    dt = float(cfg.get("dt", 0.05))
    t = 0.0
    save_every = int(cfg.get("save_every", 1))
    manifest_every = int(cfg.get("manifest_every", 4))
    checkpoint_every = int(cfg.get("checkpoint_every", 4))

    for epoch in range(epochs):
        # One epoch
        out = run_epoch(cfg, world, graph, sources, epoch=epoch, t=t)
        world = out["world"]
        edges_writer.add(out["edge_rows"])
        metrics_writer.add(out["metrics_row"])

        # Save NPZ for UI
        if (epoch % save_every) == 0:
            _ = save_volume_npz(
                gcp_cfg, run_id, str(run_dir), epoch=epoch,
                rho=world["rho"],
                eps=world["eps"],
                outer_mask=world["outer_mask"],
                core_mask=world["core_mask"],
            )

        # Periodic manifest + checkpoint
        if (epoch % manifest_every) == 0:
            manifest["summaries"] = {
                "status": "running",
                "epoch": epoch,
                "time": t,
            }
            write_manifest(gcp_cfg, run_id, manifest)

        if (epoch % checkpoint_every) == 0:
            write_checkpoint(
                gcp_cfg, run_id, str(run_dir),
                name=f"epoch_{epoch:04d}",
                payload={"epoch": epoch, "time": t}
            )

        t += dt

    # Final flush
    edges_writer.maybe_flush(force=True)
    metrics_writer.maybe_flush(force=True)

    manifest["summaries"] = {
        "status": "complete",
        "epochs": epochs,
        "last_time": t,
    }

    # Always write final manifest and checkpoint
    write_manifest(gcp_cfg, run_id, manifest)
    write_checkpoint(gcp_cfg, run_id, str(run_dir), name="final", payload={"epochs": epochs, "time": t})
    return run_id


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fuka 5.0 headless simulator (local-first).")
    ap.add_argument("--config", "-c", type=str, default=None, help="Path to JSON config (optional).")
    ap.add_argument("--run_id", "-r", type=str, default=None, help="Override run id (optional).")
    return ap.parse_args()

def main() -> None:
    args = _parse_args()
    run_id = run_sim(args.config, args.run_id)
    print(run_id)

if __name__ == "__main__":
    main()