#!/usr/bin/env bash
# Source this before running scripts.

# ====== Edit these ======
export F5_GCP_PROJECT_ID="your-project-id"
export F5_GCS_BUCKET="gs://your-fuka5-bucket"
export F5_RUNS_PREFIX="runs"

export F5_GCP_REGION="us-central1"
export F5_GCP_ZONE="us-central1-a"

# Where the repo will live on the VM
export F5_REPO_DIR="/srv/fuka5"

# Local data/cache directory on the VM
export F5_DATA_LOCAL="/srv/fuka5_data"

# Python venv interpreter path
export F5_PY="/srv/fuka5_venv/bin/python"

# Streamlit UI port
export F5_STREAMLIT_PORT="8501"

# Optional: downsample factor for volume snapshots the UI will load
export F5_VOL_DOWNSAMPLE="2"