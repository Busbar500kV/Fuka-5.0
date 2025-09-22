#!/usr/bin/env bash
set -euo pipefail

# Fuka 5.0 VM one-time setup.
# - Installs system deps, Python venv, gcloud SDK
# - Installs Python requirements
# - Ensures GCS bucket exists
#
# Prereqs:
#   1) You edited scripts/env.sh (exporting F5_* vars)
#   2) On GCE, attach a Service Account with Storage permissions (recommended)
#      OR run `gcloud auth application-default login` interactively.

# Load environment variables
if [[ -f "$(dirname "$0")/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$(dirname "$0")/env.sh"
else
  echo "[ERROR] scripts/env.sh not found. Copy from .env.example and edit."
  exit 1
fi

echo "[SETUP] Project: ${F5_GCP_PROJECT_ID}"
echo "[SETUP] Bucket : ${F5_GCS_BUCKET}"
echo "[SETUP] Repo   : ${F5_REPO_DIR}"
echo "[SETUP] Data   : ${F5_DATA_LOCAL}"

# 1) System packages
echo "[SETUP] Updating apt and installing base packages..."
sudo apt-get update -y
sudo apt-get install -y \
  python3.11 python3.11-venv python3.11-dev \
  build-essential git curl

# 2) Google Cloud SDK (if not present)
if ! command -v gcloud >/dev/null 2>&1; then
  echo "[SETUP] Installing Google Cloud SDK..."
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | \
    sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
  sudo apt-get update -y && sudo apt-get install -y google-cloud-sdk
else
  echo "[SETUP] gcloud already installed."
fi

# 3) Create working directories
sudo mkdir -p "${F5_REPO_DIR}" "${F5_DATA_LOCAL}"
sudo chown -R "$USER":"$USER" "${F5_REPO_DIR}" "${F5_DATA_LOCAL}"

# 4) Python venv + deps
if [[ ! -x "${F5_PY}" ]]; then
  echo "[SETUP] Creating Python venv at $(dirname "${F5_PY}") ..."
  python3.11 -m venv "$(dirname "${F5_PY}")"
fi

echo "[SETUP] Upgrading pip and installing requirements..."
"${F5_PY}" -m pip install --upgrade pip
"${F5_PY}" -m pip install -r "$(dirname "$0")/../requirements.txt"

# 5) gcloud defaults (optional but recommended)
echo "[SETUP] Configuring gcloud defaults..."
gcloud config set project "${F5_GCP_PROJECT_ID}" || true
gcloud config set compute/region "${F5_GCP_REGION}" || true
gcloud config set compute/zone "${F5_GCP_ZONE}" || true

# 6) Application Default Credentials (ADC)
# If the VM has an attached Service Account with Storage access, this is not needed.
if ! "${F5_PY}" -c 'import google.auth; print("adc-ok")' >/dev/null 2>&1; then
  echo "[INFO] If you need ADC locally, run: gcloud auth application-default login"
fi

# 7) Ensure GCS bucket exists
echo "[SETUP] Ensuring bucket exists: ${F5_GCS_BUCKET}"
if ! gsutil ls "${F5_GCS_BUCKET}" >/dev/null 2>&1; then
  echo "[SETUP] Bucket not found. Creating..."
  gsutil mb -p "${F5_GCP_PROJECT_ID}" -l "${F5_GCP_REGION}" "${F5_GCS_BUCKET}"
else
  echo "[SETUP] Bucket already exists."
fi

echo "[SETUP] Done. Next steps:"
echo "  1) Run a simulation:   bash scripts/run_sim.sh"
echo "  2) Launch the UI:      ${F5_PY} -m streamlit run app/streamlit_app.py --server.port ${F5_STREAMLIT_PORT}"