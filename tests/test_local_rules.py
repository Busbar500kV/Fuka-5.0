from pathlib import Path
import numpy as np
import pandas as pd
import json
import subprocess
import time

def test_local_run(tmp_path: Path):
    """
    Run a short local simulation and verify expected outputs exist.
    """
    repo = Path(__file__).resolve().parents[1]
    run_script = repo / "fuka5" / "run" / "sim_cli.py"
    cfg = tmp_path / "test_config.json"
    cfg.write_text('{"epochs": 2, "grid_shape": [16,16,16], "graph_stride": 8}')

    run_id = subprocess.check_output(
        ["python3", str(run_script), "--config", str(cfg)]
    ).decode().strip()

    runs_root = Path("/home/busbar/fuka-runs/runs")
    run_dir = runs_root / run_id

    # give filesystem a moment if run is very fast
    time.sleep(1)

    assert run_dir.exists(), f"run dir not found: {run_dir}"
    assert (run_dir / "manifest.json").exists(), "manifest missing"

    vols = sorted((run_dir / "volumes").glob("*.npz"))
    shards = sorted((run_dir / "shards").glob("*.parquet"))
    metrics = sorted((run_dir / "metrics").glob("*.parquet"))

    assert vols, "no npz volumes written"
    assert shards, "no edge parquet written"
    assert metrics, "no metrics parquet written"

    # basic content check
    npz = np.load(vols[0])
    for key in ("rho", "eps", "outer_mask", "core_mask"):
        assert key in npz, f"missing key {key} in npz"

    manifest = json.load(open(run_dir / "manifest.json"))
    assert "run_id" in manifest and manifest["run_id"] == run_id
    assert manifest["summaries"]["status"] in ("running", "complete")