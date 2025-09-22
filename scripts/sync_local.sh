#!/usr/bin/env bash
set -euo pipefail

# Sync a run folder from GCS to local disk for offline inspection.
# Usage:
#   RUN_ID=20250921-120000 bash scripts/sync_local.sh
#
# Requires:
#   - scripts/env.sh configured
#   - gsutil authenticated (ADC or user creds)

# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"

if [[ -z "${RUN_ID:-}" ]]; then
  echo "[ERROR] Please set RUN_ID env var. Example:"
  echo "  RUN_ID=$(date -u +%Y%m%d-%H%M%S) bash scripts/sync_local.sh"
  exit 1
fi

LOCAL_BASE="${LOCAL_BASE:-./_runs}"
LOCAL_DIR="${LOCAL_BASE}/${RUN_ID}"
mkdir -p "${LOCAL_DIR}"

# Build GCS prefix
BUCKET_NO_SCHEME="${F5_GCS_BUCKET#gs://}"
GCS_PREFIX="gs://${BUCKET_NO_SCHEME}/${F5_RUNS_PREFIX}/${RUN_ID}"

echo "[SYNC] From: ${GCS_PREFIX}"
echo "[SYNC]   To: ${LOCAL_DIR}"

gsutil -m rsync -r "${GCS_PREFIX}" "${LOCAL_DIR}"

echo "[SYNC] Complete."