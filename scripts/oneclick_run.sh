#!/usr/bin/env bash
set -euo pipefail

# -------- Config (change if you like) --------
PORT="${PORT:-8501}"                 # Streamlit port
KEEP_RUNS="${KEEP_RUNS:-5}"          # keep newest N runs, purge older
LOG_RETENTION_DAYS="${LOG_RETENTION_DAYS:-14}"
STORAGE="${F5_STORAGE:-local}"
RUNS_DIR="${F5_LOCAL_RUNS_DIR:-/home/$USER/fuka-runs}"
RUNS_PREFIX="${F5_RUNS_PREFIX:-runs}"

WORLD_CFG="${WORLD_CFG:-configs/world.default.json}"
SOURCES_CFG="${SOURCES_CFG:-configs/sources.example.json}"
TRAINING_CFG="${TRAINING_CFG:-configs/training.default.json}"
GCP_CFG="${GCP_CFG:-configs/local.default.json}"

# -------- Env bootstrap --------
export F5_STORAGE="$STORAGE"
export F5_LOCAL_RUNS_DIR="$RUNS_DIR"
export F5_RUNS_PREFIX="$RUNS_PREFIX"

mkdir -p "$RUNS_DIR/$RUNS_PREFIX" logs

# Ensure venv exists (idempotent)
if [[ ! -d .venv ]]; then
  echo "[oneclick] creating .venv…"
  sudo apt-get update -y >/dev/null 2>&1 || true
  sudo apt-get install -y python3.12-venv >/dev/null 2>&1 || true
  python3 -m venv .venv
fi
source .venv/bin/activate

# Ensure deps present (fast if already installed)
python -m pip -q install --upgrade pip >/dev/null
python -m pip -q install -r <(cat <<REQ
numpy==1.26.4
pandas==2.2.2
scipy==1.14.1
fastparquet==2024.5.0
pyarrow==17.0.0
plotly==5.24.1
streamlit==1.38.0
watchdog==4.0.2
REQ
) >/dev/null || true

# -------- Start Streamlit if not running --------
if ! pgrep -fa "streamlit run .*app/streamlit_app.py" >/dev/null 2>&1; then
  echo "[oneclick] starting streamlit on :$PORT"
  nohup streamlit run app/streamlit_app.py \
      --server.headless true \
      --server.address 0.0.0.0 \
      --server.port "$PORT" \
      >> "logs/streamlit.log" 2>&1 &
  disown
else
  echo "[oneclick] streamlit already running"
fi

# -------- Clean old runs (keep newest N) --------
runs_base="$RUNS_DIR/$RUNS_PREFIX"
if [[ -d "$runs_base" ]]; then
  mapfile -t runs < <(ls -1t "$runs_base" | grep -E '^run-' || true)
  if (( ${#runs[@]} > KEEP_RUNS )); then
    echo "[oneclick] pruning older runs (keeping $KEEP_RUNS newest)"
    for old in "${runs[@]:$KEEP_RUNS}"; do
      echo "  - removing $runs_base/$old"
      rm -rf -- "$runs_base/$old"
    done
  fi
fi

# -------- Rotate old logs --------
find logs -type f -mtime +${LOG_RETENTION_DAYS} -print -delete 2>/dev/null || true

# -------- Start a new simulation (detached) --------
RUN_ID="run-$(date +%Y%m%d-%H%M%S)"
echo "[oneclick] starting simulation RUN_ID=$RUN_ID"
echo "[oneclick] artifacts -> $RUNS_DIR/$RUNS_PREFIX/$RUN_ID"
nohup python -u -m fuka5.run.sim_cli \
  --run_id "$RUN_ID" \
  --world "$WORLD_CFG" \
  --sources "$SOURCES_CFG" \
  --training "$TRAINING_CFG" \
  --gcp "$GCP_CFG" \
  > "logs/${RUN_ID}.log" 2>&1 &
echo $! > "logs/${RUN_ID}.pid"
disown

echo "[oneclick] done. Tail logs with:"
echo "  tail -f logs/${RUN_ID}.log"
echo "[oneclick] UI: http://\$(tailscale ip -4 | head -1):${PORT}"
