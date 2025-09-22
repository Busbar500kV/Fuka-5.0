#!/usr/bin/env bash
set -euo pipefail

# Launch a Fuka 5.0 run and stream artifacts to GCS.

# Load env
# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"

WORLD_CFG="${WORLD_CFG:-configs/world.default.json}"
SOURCES_CFG="${SOURCES_CFG:-configs/sources.example.json}"
TRAINING_CFG="${TRAINING_CFG:-configs/training.default.json}"
GCP_CFG="${GCP_CFG:-configs/gcp.default.json}"

LOCAL_DIR="${LOCAL_DIR:-${F5_DATA_LOCAL}/${RUN_ID}}"
mkdir -p "${LOCAL_DIR}"

echo "[RUN] Run ID: ${RUN_ID}"
echo "[RUN] Local dir: ${LOCAL_DIR}"
echo "[RUN] World:   ${WORLD_CFG}"
echo "[RUN] Sources: ${SOURCES_CFG}"
echo "[RUN] Training:${TRAINING_CFG}"
echo "[RUN] GCP:     ${GCP_CFG}"

"${F5_PY}" -m fuka5.run.sim_cli \
  --run_id "${RUN_ID}" \
  --world "${WORLD_CFG}" \
  --sources "${SOURCES_CFG}" \
  --training "${TRAINING_CFG}" \
  --gcp "${GCP_CFG}" \
  --local_dir "${LOCAL_DIR}" \
  --volume_downsample "${F5_VOL_DOWNSAMPLE:-2}" \
  --edges_flush_every 5 \
  --metrics_flush_every 5 \
  --volume_every 10 \
  --checkpoint_every 20

echo "[RUN] done."