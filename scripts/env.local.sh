#!/bin/bash
# Fuka 5.0 — Local environment setup (Linux laptop)
# Source or run this before launching the sim.

set -euo pipefail

# ---- Required env for local runs (UI reads from this path) ----
export F5_STORAGE="local"
export F5_LOCAL_RUNS_DIR="/home/busbar/fuka-runs"
export F5_RUNS_PREFIX="runs"

# ---- Ensure directory exists ----
mkdir -p "${F5_LOCAL_RUNS_DIR}/${F5_RUNS_PREFIX}"

echo "[FUKA] Local env set."
echo "       STORAGE=${F5_STORAGE}"
echo "       RUNS_DIR=${F5_LOCAL_RUNS_DIR}/${F5_RUNS_PREFIX}"