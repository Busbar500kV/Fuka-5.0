#!/bin/bash
# Fuka 5.0 — One-time local setup (creates venv, installs deps)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="/opt/fuka-venv"
PY="${VENV_DIR}/bin/python"

echo "[FUKA] Creating venv at ${VENV_DIR} (sudo needed if /opt)"
sudo mkdir -p "$(dirname "$VENV_DIR")"
sudo python3 -m venv "$VENV_DIR"

echo "[FUKA] Upgrading pip"
sudo "$PY" -m pip install --upgrade pip

echo "[FUKA] Installing requirements"
sudo "$PY" -m pip install -r "${REPO_DIR}/requirements.txt"

echo "[FUKA] Setup complete."
echo "Next run:"
echo "  sudo bash ${REPO_DIR}/scripts/run_local.sh"