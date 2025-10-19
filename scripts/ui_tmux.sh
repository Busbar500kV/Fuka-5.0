#!/bin/bash
# Fuka 5.0 — Streamlit UI via tmux (persists after terminal closes)
# Usage:
#   bash scripts/ui_tmux.sh start     # start (or restart) UI in session "fuka-ui"
#   bash scripts/ui_tmux.sh stop      # stop the session if running
#   bash scripts/ui_tmux.sh status    # show session status
#   bash scripts/ui_tmux.sh attach    # attach to the session (view live logs)
#   bash scripts/ui_tmux.sh port 8503 # change port used next start
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="fuka-ui"
PORT_FILE="${REPO_DIR}/.ui_port"
PORT="$(cat "${PORT_FILE}" 2>/dev/null || echo 8501)"
PY="/opt/fuka-venv/bin/python"
APP="${REPO_DIR}/app/streamlit_app.py"

# Load env (same as other launchers)
if [[ -f "${REPO_DIR}/scripts/env.local.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR}/scripts/env.local.sh"
else
  export F5_STORAGE="local"
  export F5_LOCAL_RUNS_DIR="/home/busbar/fuka-runs"
  export F5_RUNS_PREFIX="runs"
fi
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

cmd="${1:-status}"
case "${cmd}" in
  start)
    # Ensure tmux exists
    if ! command -v tmux >/dev/null 2>&1; then
      echo "[FUKA] tmux not found. Install with: sudo apt-get update && sudo apt-get install -y tmux" >&2
      exit 1
    fi
    # Kill any existing session
    tmux has-session -t "${SESSION}" 2>/dev/null && tmux kill-session -t "${SESSION}" || true
    # Start new detached session
    tmux new -s "${SESSION}" -d \
      "cd '${REPO_DIR}' && exec ${PY} -m streamlit run '${APP}' --server.address 0.0.0.0 --server.port ${PORT}"
    echo "[FUKA] UI started in tmux session '${SESSION}' on port ${PORT}."
    echo "       Open: http://localhost:${PORT}"
    echo "       Attach logs: tmux attach -t ${SESSION}  (Ctrl-b then d to detach)"
    ;;
  stop)
    tmux has-session -t "${SESSION}" 2>/dev/null && tmux kill-session -t "${SESSION}" || true
    echo "[FUKA] UI stopped (session '${SESSION}' removed)."
    ;;
  status)
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      echo "[FUKA] UI is running in tmux session '${SESSION}' on port ${PORT}."
    else
      echo "[FUKA] UI is not running."
    fi
    ;;
  attach)
    tmux attach -t "${SESSION}"
    ;;
  port)
    newp="${2:-}"
    if [[ -z "${newp}" ]]; then
      echo "Usage: bash scripts/ui_tmux.sh port <PORT>" >&2
      exit 1
    fi
    echo "${newp}" > "${PORT_FILE}"
    echo "[FUKA] Port updated to ${newp}. Next 'start' will use it."
    ;;
  *)
    echo "Usage: bash scripts/ui_tmux.sh {start|stop|status|attach|port <PORT>}"
    exit 1
    ;;
esac