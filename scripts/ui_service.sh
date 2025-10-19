#!/bin/bash
# Fuka 5.0 — Streamlit UI as a user-level systemd service (survives terminal close)
# Usage:
#   bash scripts/ui_service.sh install|start|stop|restart|status|logs|port <PORT>

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_DIR="${HOME}/.config/systemd/user"
UNIT_NAME="fuka-ui.service"
UNIT_PATH="${UNIT_DIR}/${UNIT_NAME}"
VENV_PY="/opt/fuka-venv/bin/python"
APP_PATH="${REPO_DIR}/app/streamlit_app.py"
PORT_FILE="${REPO_DIR}/.ui_port"
DEFAULT_PORT="8501"

mkdir -p "${UNIT_DIR}"

port="$(cat "${PORT_FILE}" 2>/dev/null || echo "${DEFAULT_PORT}")"

load_env() {
  if [[ -f "${REPO_DIR}/scripts/env.local.sh" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_DIR}/scripts/env.local.sh"
  else
    export F5_STORAGE="local"
    export F5_LOCAL_RUNS_DIR="/home/busbar/fuka-runs"
    export F5_RUNS_PREFIX="runs"
  end
  export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
}

write_unit() {
  load_env
  cat > "${UNIT_PATH}" <<EOF
[Unit]
Description=Fuka 5.0 Streamlit UI (user service)

[Service]
Type=simple
WorkingDirectory=${REPO_DIR}
Environment=F5_STORAGE=${F5_STORAGE}
Environment=F5_LOCAL_RUNS_DIR=${F5_LOCAL_RUNS_DIR}
Environment=F5_RUNS_PREFIX=${F5_RUNS_PREFIX}
Environment=PYTHONPATH=${REPO_DIR}
ExecStart=${VENV_PY} -m streamlit run ${APP_PATH} --server.port ${port} --server.address 0.0.0.0
Restart=always
RestartSec=2
# Log to journal (view with: journalctl --user -u ${UNIT_NAME} -f)
StandardOutput=journal
StandardError=inherit

[Install]
WantedBy=default.target
EOF
}

cmd="${1:-help}"
case "${cmd}" in
  install)
    write_unit
    systemctl --user daemon-reload
    systemctl --user enable "${UNIT_NAME}"
    systemctl --user start "${UNIT_NAME}"
    echo "[FUKA] UI service installed and started on port ${port}."
    echo "       View: http://localhost:${port}"
    echo "       Logs: journalctl --user -u ${UNIT_NAME} -f"
    ;;
  start)
    systemctl --user start "${UNIT_NAME}"
    echo "[FUKA] Started. http://localhost:${port}"
    ;;
  stop)
    systemctl --user stop "${UNIT_NAME}" || true
    echo "[FUKA] Stopped."
    ;;
  restart)
    systemctl --user restart "${UNIT_NAME}"
    echo "[FUKA] Restarted. http://localhost:${port}"
    ;;
  status)
    systemctl --user status "${UNIT_NAME}" --no-pager
    ;;
  logs)
    journalctl --user -u "${UNIT_NAME}" -f -n 100
    ;;
  port)
    new_port="${2:-}"
    if [[ -z "${new_port}" ]]; then
      echo "Usage: bash scripts/ui_service.sh port <PORT>"
      exit 1
    fi>
    echo "${new_port}" > "${PORT_FILE}"
    port="${new_port}"
    write_unit
    systemctl --user daemon-reload
    systemctl --user restart "${UNIT_NAME}"
    echo "[FUKA] Port changed to ${port}. http://localhost:${port}"
    ;;
  help|*)
    echo "Usage:"
    echo "  bash scripts/ui_service.sh install|start|stop|restart|status|logs|port <PORT>"
    ;;
esac