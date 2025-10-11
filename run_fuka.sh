#!/usr/bin/env bash
# Fuka one-command simulation launcher
set -Eeuo pipefail

# -------- repo context --------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
VENV="$REPO_ROOT/.venv"
LOG_DIR="$REPO_ROOT/logs"

# -------- defaults (override with flags/env) --------
WORLD="${WORLD:-configs/world.default.json}"
SOURCES="${SOURCES:-configs/sources.example.json}"
TRAINING="${TRAINING:-configs/training.default.json}"
GCP_CFG="${GCP_CFG:-configs/local.default.json}"

export F5_STORAGE="${F5_STORAGE:-local}"
export F5_LOCAL_RUNS_DIR="${F5_LOCAL_RUNS_DIR:-/home/$USER/fuka-runs}"
export F5_RUNS_PREFIX="${F5_RUNS_PREFIX:-runs}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

RUN_ID=""
STEPS=""           # positional arg (optional)
SKIP_INSTALL=0

usage() {
  cat <<USAGE
Usage: $(basename "$0") [steps] [--run-id ID] [--world PATH] [--sources PATH] [--training PATH] [--gcp PATH] [--no-install]

Positional:
  steps              Optional. If provided, a temporary copy of WORLD is used with "steps" overridden.

Options:
  --run-id ID        Use a specific run id (default: run-YYYYmmdd-HHMMSS)
  --world PATH       Path to world config (default: $WORLD)
  --sources PATH     Path to sources config (default: $SOURCES)
  --training PATH    Path to training config (default: $TRAINING)
  --gcp PATH         Path to storage config (default: $GCP_CFG)
  --no-install       Skip (re)installing Python dependencies

Env overrides:
  F5_STORAGE=local
  F5_LOCAL_RUNS_DIR=/home/$USER/fuka-runs
  F5_RUNS_PREFIX=runs

Examples:
  $(basename "$0")              # use defaults from configs/*.json
  $(basename "$0") 2000         # run with 2000 steps (temp world copy)
  $(basename "$0") --run-id run-x --no-install
USAGE
}

# -------- parse args --------
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  STEPS="$1"; shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)         RUN_ID="${2:-}"; shift 2;;
    --run-id=*)       RUN_ID="${1#*=}"; shift;;
    --world)          WORLD="${2:-}"; shift 2;;
    --world=*)        WORLD="${1#*=}"; shift;;
    --sources)        SOURCES="${2:-}"; shift 2;;
    --sources=*)      SOURCES="${1#*=}"; shift;;
    --training)       TRAINING="${2:-}"; shift 2;;
    --training=*)     TRAINING="${1#*=}"; shift;;
    --gcp)            GCP_CFG="${2:-}"; shift 2;;
    --gcp=*)          GCP_CFG="${1#*=}"; shift;;
    --no-install)     SKIP_INSTALL=1; shift;;
    -h|--help)        usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

mkdir -p "$LOG_DIR" "$F5_LOCAL_RUNS_DIR/$F5_RUNS_PREFIX"

# -------- venv --------
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[setup] creating venv at $VENV"
  if ! python3 -m venv "$VENV" 2>/dev/null; then
    echo "[setup] python3-venv missing. On Ubuntu: sudo apt-get install -y python3.12-venv" >&2
    exit 1
  fi
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -V
pip -V

# -------- deps --------
if [[ "$SKIP_INSTALL" -eq 0 ]]; then
  echo "[deps] upgrading pip/setuptools/wheel"
  python -m pip install -U pip setuptools wheel >/dev/null
  if [[ -f requirements.txt ]]; then
    echo "[deps] installing from requirements.txt"
    python -m pip install -r requirements.txt
  else
    echo "[deps] installing core stack"
    python -m pip install numpy pandas pyarrow plotly streamlit
  fi
fi

# -------- optional: override steps by writing a temp world --------
TMP_WORLD="$WORLD"
if [[ -n "$STEPS" ]]; then
  if [[ ! -f "$WORLD" ]]; then
    echo "[error] WORLD not found: $WORLD" >&2; exit 1
  fi
  TMP_WORLD="$(mktemp -p "$REPO_ROOT" world.override.XXXX.json)"
  echo "[world] overriding steps=$STEPS -> $TMP_WORLD"
  python - "$WORLD" "$TMP_WORLD" "$STEPS" <<'PY'
import json, sys
src, dst, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src) as f:
    cfg = json.load(f)
cfg["steps"] = steps
with open(dst, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"wrote {dst}")
PY
fi

# -------- RUN_ID & paths --------
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="run-$(date +%Y%m%d-%H%M%S)"
fi
LOG_FILE="$LOG_DIR/$RUN_ID.log"
ART_DIR="$F5_LOCAL_RUNS_DIR/$F5_RUNS_PREFIX/$RUN_ID"

echo "[run] Storage=$F5_STORAGE  runs_dir=$F5_LOCAL_RUNS_DIR  prefix=$F5_RUNS_PREFIX"
echo "[run] RUN_ID=$RUN_ID"
echo "[run] world=$TMP_WORLD"
echo "[run] sources=$SOURCES"
echo "[run] training=$TRAINING"
echo "[run] gcp_cfg=$GCP_CFG"
echo "[run] Log: $LOG_FILE"
echo "[run] Artifacts (expected): $ART_DIR"

# -------- go --------
set -x
python -u -m fuka5.run.sim_cli \
  --run_id "$RUN_ID" \
  --world "$TMP_WORLD" \
  --sources "$SOURCES" \
  --training "$TRAINING" \
  --gcp "$GCP_CFG" \
  2>&1 | tee "$LOG_FILE"
set +x

echo
echo "=== DONE ==="
echo "Log:        $LOG_FILE"
echo "Artifacts:  $ART_DIR"
echo "Tail log:   tail -f \"$LOG_FILE\""
