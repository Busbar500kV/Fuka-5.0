#!/bin/bash
# Fuka 5.0 — Local UI launcher (auto-port version)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Load local env
if [[ -f "${REPO_DIR}/scripts/env.local.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR}/scripts/env.local.sh"
else
  echo "[FUKA] ERROR: scripts/env.local.sh not found." >&2
  exit 1
fi

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

APP="${REPO_DIR}/app/streamlit_app.py"
VENV_DIR="/opt/fuka-venv"
PY="${VENV_DIR}/bin/python"

LOG_DIR="${REPO_DIR}/.logs"
mkdir -p "${LOG_DIR}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG_FILE="${LOG_DIR}/ui_${STAMP}.log"

# ---- Find first available port ≥ 8501 ----
PORT=8501
while ss -lnt | awk '{print $4}' | grep -q ":${PORT}$"; do
  PORT=$((PORT+1))
done

echo "[FUKA] Launching UI"
echo "    APP=${APP}"
echo "    RUNS_DIR=${F5_LOCAL_RUNS_DIR}/${F5_RUNS_PREFIX}"
echo "    PYTHONPATH=${PYTHONPATH}"
echo "    PORT=${PORT}"
echo "    LOG_FILE=${LOG_FILE}"
echo "Open: http://localhost:${PORT}"

exec "${PY}" -m streamlit run "${APP}" --server.port "${PORT}" --server.address 0.0.0.0 2>&1 | tee -a "${LOG_FILE}"