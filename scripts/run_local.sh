#!/bin/bash
# Fuka 5.0 – Local simulation launcher (for Blink/Tailscale)
set -euo pipefail

REPO_DIR="$(dirname "$(dirname "$0")")"
VENV_DIR="/opt/fuka-venv"
PYTHON_BIN="$VENV_DIR/bin/python"
RUN_SCRIPT="$REPO_DIR/fuka5/run/sim_cli.py"
LOG_DIR="/var/log/fuka"
mkdir -p "$LOG_DIR"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ID="FUKA_5_0_${STAMP}"
CFG_PATH="${REPO_DIR}/configs/local.default.json"

echo "[FUKA 5.0] Starting simulation (run id $RUN_ID)..."
sudo -E "$PYTHON_BIN" "$RUN_SCRIPT" --config "$CFG_PATH" --run_id "$RUN_ID" \
  >> "$LOG_DIR/sim_${RUN_ID}.log" 2>&1

echo "[FUKA 5.0] Simulation complete (run id $RUN_ID)"
echo "Log file: $LOG_DIR/sim_${RUN_ID}.log"