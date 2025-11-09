#!/usr/bin/env bash
set -e

KEEP_RUN_ID="FUKA_5_0_20251019T002759Z"
RUNS_DIR="/home/busbar/fuka-runs/runs"

echo "[FUKA] Cleaning local runs except ${KEEP_RUN_ID}"
shopt -s nullglob
for d in "${RUNS_DIR}"/*; do
  base="$(basename "$d")"
  if [[ "$base" != "$KEEP_RUN_ID" ]]; then
    echo "  Removing $base"
    rm -rf "$d"
  else
    echo "  Preserving $base"
  fi
done
echo "[FUKA] Done."