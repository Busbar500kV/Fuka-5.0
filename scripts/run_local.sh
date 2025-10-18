#!/bin/bash
# Fuka 5.0 – Local simulation launcher (single-command, loads env from repo)
set -euo pipefail

# Resolve repo root and load local env (creates runs dir, sets env vars)
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "${REPO_DIR}/scripts/env.local.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR}/scripts/env.local.sh"
else
  echo "[FUKA] ERROR: scripts/env.local.sh not found. Did you add it to GitHub?" >&2
  exit 1
fi

# Python/venv
VENV_DIR="/opt/fuka-venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
RUN_SCRIPT="${REPO_DIR}/fuka5/run/sim_cli.py"
LOG_DIR="/var/log/fuka"
mkdir -p "$LOG_DIR"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ID="FUKA_5_0_${STAMP}"
CFG_PATH="${REPO_DIR}/configs/local.default.json"

echo "[FUKA] Starting local simulation"
echo "       RUN_ID=${RUN_ID}"
echo "       CFG=${CFG_PATH}"
echo "       PY=${PYTHON_BIN}"

sudo -E "${PYTHON_BIN}" "${RUN_SCRIPT}" --config "${CFG_PATH}" --run_id "${RUN_ID}" \
  >> "${LOG_DIR}/sim_${RUN_ID}.log" 2>&1

echo "[FUKA] Done. Log: ${LOG_DIR}/sim_${RUN_ID}.log"