from fuka5.io.compat_shim import init_backend_shims
init_backend_shims()

"""
fuka5.run.sim_cli
-----------------
Headless simulator entrypoint.

Responsibilities
- Parse CLI args and load JSON configs with env expansion
- Build World (rho→epsilon,g), Graph (nodes/edges), Sources (taps)
- Initialize substrate state: EdgeState dict, ThermalField, DecoderBank
- Run epoch loop with ON/OFF schedule:
    * physics + local updates via substrate.updates.step_epoch
    * thermal update, rewards/metrics, logging
    * morphogenesis update on rho using local harvested power & mass cost
    * cadence-based shard/volume/checkpoint writes to GCS
- Write manifest at run start (immutable)
"""

from __future__ import annotations
import argparse
import os
import time
import json
from typing import Dict, Any, List, Tuple
import numpy as np

from .. import load_json_with_env, __version__, env_get
from ..core.world import World
from ..core.graph import build_graph_from_world, GraphConfig
from ..core.sources import Sources
from ..substrate.edges import initialize_edge_states
from ..substrate.updates import (
    step_epoch, CapsParams, GateParams, BatteryParams, MaturityParams,
    RewardsParams
)
from ..substrate.thermal import ThermalParams, ThermalField
from ..substrate.decoders import DecoderBank
from ..io.gcs import load_gcp_config, gcs_path
from ..io.writers import ShardWriter, save_volume_npz, write_manifest, write_checkpoint
from .schedule import TimeConfig, Cadence, seeds_from_int


# ---------------------------
# Helpers
# ---------------------------

def _ensure_local_dirs(base: str) -> None:
    for sub in ("shards", "volumes", "checkpoints"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)

def _now_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())

def _world_summary(world: World) -> Dict[str, Any]:
    nx, ny, nz = world.shape
    return {
        "dims": [nx, ny, nz],
        "dx": world.dx,
        "outer_voxels": int(world.outer_mask.sum()),
        "core_voxels": int(world.core_mask.sum()),
    }

def _graph_summary(graph) -> Dict[str, Any]:
    return {"nodes": graph.N, "edges": graph.E}

def _counts_summary(world: World, graph) -> Dict[str, Any]:
    return {
        "outer_voxels": int(world.outer_mask.sum()),
        "core_voxels": int(world.core_mask.sum()),
        "nodes": graph.N,
        "edges": graph.E,
    }

def _voxel_splat(shape, nodes_xyz: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Splat node values into a voxel grid using nearest integer voxel.
    """
    grid = np.zeros(shape, dtype=np.float32)
    assert nodes_xyz.shape[0] == values.shape[0]
    nx, ny, nz = shape
    xs = np.clip(np.rint(nodes_xyz[:, 0]).astype(int), 0, nx - 1)
    ys = np.clip(np.rint(nodes_xyz[:, 1]).astype(int), 0, ny - 1)
    zs = np.clip(np.rint(nodes_xyz[:, 2]).astype(int), 0, nz - 1)
    for x, y, z, val in zip(xs, ys, zs, values):
        grid[x, y, z] += float(val)
    return grid

def _smooth3(vol: np.ndarray, repeats: int = 1) -> np.ndarray:
    """
    Very small 3D smoothing (box filter) to avoid spiky morphogenesis forcing.
    """
    v = vol.astype(np.float32, copy=False)
    for _ in range(max(0, repeats)):
        pad = np.pad(v, ((1,1),(1,1),(1,1)), mode="edge")
        nb = (
            pad[2:,1:-1,1:-1] + pad[:-2,1:-1,1:-1] +
            pad[1:-1,2:,1:-1] + pad[1:-1,:-2,1:-1] +
            pad[1:-1,1:-1,2:] + pad[1:-1,1:-1,:-2]
        )
        v = (v * 0.5 + nb * (0.5 / 6.0)).astype(np.float32, copy=False)
    return v


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Fuka 5.0 simulator")
    ap.add_argument("--run_id", type=str, default=None, help="Run identifier (default: UTC timestamp)")
    ap.add_argument("--world", type=str, required=True, help="Path to world JSON")
    ap.add_argument("--sources", type=str, required=True, help="Path to sources JSON")
    ap.add_argument("--training", type=str, required=True, help="Path to training JSON")
    ap.add_argument("--gcp", type=str, required=True, help="Path to gcp JSON")
    ap.add_argument("--seed", type=int, default=None, help="Base RNG seed")
    ap.add_argument("--local_dir", type=str, default=None, help="Local work dir (default: $F5_DATA_LOCAL/<run_id>)")
    ap.add_argument("--volume_downsample", type=int, default=2, help="Downsample factor for volumes")
    ap.add_argument("--edges_flush_every", type=int, default=5)
    ap.add_argument("--metrics_flush_every", type=int, default=5)
    ap.add_argument("--volume_every", type=int, default=10)
    ap.add_argument("--checkpoint_every", type=int, default=20)
    args = ap.parse_args()

    # Run id & local directories
    run_id = args.run_id or _now_run_id()
    base_local = args.local_dir or os.path.join(env_get("F5_DATA_LOCAL", "/tmp"), run_id)
    _ensure_local_dirs(base_local)

    # Load configs (env-expanded)
    world_cfg_dict = load_json_with_env(args.world)
    sources_cfg_dict = load_json_with_env(args.sources)
    training_cfg_dict = load_json_with_env(args.training)
    gcp_cfg = load_gcp_config(args.gcp)

    # Seeds
    seeds = seeds_from_int(args.seed)
    rng_world = np.random.default_rng(seeds["world"])
    rng_graph = np.random.default_rng(seeds["graph"])
    rng_src   = np.random.default_rng(seeds["sources"])
    rng_sub   = np.random.default_rng(seeds["substrate"])
    rng_run   = np.random.default_rng(seeds["runner"])

    # Build world
    world = World.from_config_dict(world_cfg_dict)

    # Build graph from world (fixed for this run)
    gcfg = GraphConfig()
    graph = build_graph_from_world(
        rho=world.rho, eps=world.eps, g=world.g,
        outer_mask=world.outer_mask, core_mask=world.core_mask,
        dx=world.dx, cfg=gcfg, rng=rng_graph
    )

    # Build sources + taps to graph
    sources = Sources.from_config_dict(sources_cfg_dict)
    sources.build_taps_to_graph(graph.positions(), coupling_cfg=sources_cfg_dict.get("coupling", {}))

    # Substrate initial states
    caps_params  = CapsParams.from_dict(training_cfg_dict["caps"])
    gate_params  = GateParams.from_dict(training_cfg_dict["gates"])
    batt_params  = BatteryParams.from_dict(training_cfg_dict["battery"])
    mat_params   = MaturityParams.from_dict(training_cfg_dict["maturity"])
    therm_params = ThermalParams.from_dict(training_cfg_dict["thermal"])
    rewards_par  = RewardsParams.from_dict(training_cfg_dict["rewards"])
    time_cfg     = TimeConfig.from_dict(training_cfg_dict["time"])

    edge_states = initialize_edge_states(graph, Cmin=caps_params.Cmin, Cmax=caps_params.Cmax, rng=rng_sub)
    therm = ThermalField(graph.N, params=therm_params)
    decoders = DecoderBank(n_edges=graph.E, eta_w=float(training_cfg_dict["decoders"]["eta_w"]), rng=rng_sub)

    # Writers & cadences
    cadence = Cadence(
        edges_flush_every=args.edges_flush_every,
        metrics_flush_every=args.metrics_flush_every,
        volume_every=args.volume_every,
        checkpoint_every=args.checkpoint_every,
    )
    edges_writer   = ShardWriter(gcp_cfg, run_id, kind="edges",   local_dir=base_local, flush_every=cadence.edges_flush_every)
    metrics_writer = ShardWriter(gcp_cfg, run_id, kind="metrics", local_dir=base_local, flush_every=cadence.metrics_flush_every)

    # Manifest
    manifests_cfg = {
        "world_path": args.world,
        "sources_path": args.sources,
        "training_path": args.training,
        "gcp_path": args.gcp,
        "world_cfg": world_cfg_dict,
        "sources_cfg": sources_cfg_dict,
        "training_cfg": training_cfg_dict,
        "gcp_cfg": gcp_cfg,
    }
    summaries = {
        "world": _world_summary(world),
        "graph": _graph_summary(graph),
        "counts": _counts_summary(world, graph),
    }
    manifest = {
        "run_id": run_id,
        "created_utc": int(time.time()),
        "configs": manifests_cfg,
        "summaries": summaries,
        "cadence": {
            "edges_flush_every": cadence.edges_flush_every,
            "metrics_flush_every": cadence.metrics_flush_every,
            "volume_every": cadence.volume_every,
            "checkpoint_every": cadence.checkpoint_every,
        },
        "versions": {"fuka5": __version__},
        "seeds": seeds,
    }
    # upload manifest
    write_manifest(gcp_cfg, run_id, manifest)

    # Epoch loop
    for ep in range(time_cfg.epochs):
        on_flag = time_cfg.on_flag(ep)

        # One epoch of local physics+learning
        edge_rows, metrics_row, Pplus_node = step_epoch(
            epoch=ep, on_flag=on_flag,
            graph=graph,
            edge_states=edge_states,
            sources=sources,
            bands_cfg=training_cfg_dict["bands"],
            time_cfg={"fs": time_cfg.fs, "window_sec": time_cfg.window_sec},
            caps_params=caps_params,
            gate_params=gate_params,
            batt_params=batt_params,
            mat_params=mat_params,
            therm=therm,
            decoders=decoders,
            rewards_params=rewards_par,
            rng=rng_run,
        )

        # Append to writers and maybe flush
        edges_writer.extend(edge_rows)
        edges_writer.maybe_flush(force=False)
        metrics_writer.add(metrics_row)
        metrics_writer.maybe_flush(force=False)

        # Morphogenesis inputs:
        # - harvest field: splat Pplus_node to voxels and smooth
        # - mass cost field: splat local capacity mass per node (sum incident C) to voxels and smooth
        node_pos = graph.positions()  # (N,3) in voxel coords
        harvest_field = _smooth3(_voxel_splat(world.shape, node_pos, Pplus_node), repeats=1)

        # capacity mass per node ~ sum of C for incident edges
        C_per_edge = np.array([edge_states[e.id].C for e in sorted(graph.edges, key=lambda e: e.id)], dtype=np.float32)
        incident_sum = np.zeros(graph.N, dtype=np.float32)
        for e in graph.edges:
            incident_sum[e.u] += float(edge_states[e.id].C)
            incident_sum[e.v] += float(edge_states[e.id].C)
        mass_cost_field = _smooth3(_voxel_splat(world.shape, node_pos, incident_sum), repeats=1)

        # Normalize fields to reasonable dynamic ranges
        if harvest_field.max() > 0:
            harvest_field = harvest_field / (harvest_field.max() + 1e-9)
        if mass_cost_field.max() > 0:
            mass_cost_field = mass_cost_field / (mass_cost_field.max() + 1e-9)

        # Step morphogenesis
        world.step_morphogenesis(harvest_field, mass_cost_field, dt=1.0)

        # Periodic volume snapshots for UI (downsampled)
        if (ep % cadence.volume_every) == 0:
            downs = world.downsample(factor=int(env_get("F5_VOL_DOWNSAMPLE", str(args.volume_downsample))))
            save_volume_npz(
                gcp_cfg, run_id, base_local, epoch=ep,
                rho=downs["rho"], eps=downs["eps"],
                outer_mask=downs["outer_mask"], core_mask=downs["core_mask"]
            )

        # Periodic checkpoint
        if (ep % cadence.checkpoint_every) == 0 or ep == time_cfg.epochs - 1:
            write_checkpoint(gcp_cfg, run_id, base_local, "epoch_last", {"epoch": ep})

    # Final flush
    edges_writer.maybe_flush(force=True)
    metrics_writer.maybe_flush(force=True)
    write_checkpoint(gcp_cfg, run_id, base_local, "done", {"epoch": time_cfg.epochs - 1})

    print(f"[Fuka5] Run complete: {run_id}")
    print("GCS run folder:", gcs_path(gcp_cfg, run_id))


if __name__ == "__main__":
    main()
