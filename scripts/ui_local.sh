#!/bin/bash
# Fuka 5.0 — Local UI launcher (uses same env as sim)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Load local env (sets RUNS_DIR used by the UI)
if [[ -f "${REPO_DIR}/scripts/env.local.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR}/scripts/env.local.sh"
else
  echo "[FUKA] ERROR: scripts/env.local.sh not found." >&2
  exit 1
fi

# Ensure UI can import repo modules
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Streamlit app entry (no UI code changes)
APP="${REPO_DIR}/app/streamlit_app.py"
VENV_DIR="/opt/fuka-venv"
PY="${VENV_DIR}/bin/python"

echo "[FUKA] Launching UI"
echo "       APP=${APP}"
echo "       RUNS_DIR=${F5_LOCAL_RUNS_DIR}/${F5_RUNS_PREFIX}"
echo "       PYTHONPATH=${PYTHONPATH}"

exec "${PY}" -m streamlit run "${APP}" --server.port 8501 --server.address 0.0.0.0