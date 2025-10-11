#!/usr/bin/env bash
# Fuka 5.0 one-shot simulation runner for "busbar"
# - Creates/uses .venv
# - Ensures deps installed
# - Uses local storage backend (no GCS)
# - Logs to logs/<RUN_ID>.log
set -Eeuo pipefail

# -------- config (defaults, can be overridden by env or flags) ----------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/.venv"
LOG_DIR="$REPO_ROOT/logs"
export F5_STORAGE="${F5_STORAGE:-local}"
export F5_LOCAL_RUNS_DIR="${F5_LOCAL_RUNS_DIR:-/home/$USER/fuka-runs}"
export F5_RUNS_PREFIX="${F5_RUNS_PREFIX:-runs}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

RUN_ID=""
SKIP_INSTALL=0
FORWARD_ARGS=()

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--run-id ID] [--dir PATH] [--prefix NAME] [--no-install] [--] [extra sim args...]

Options:
  --run-id ID       Use a specific run id (default: run-YYYYmmdd-HHMMSS)
  --dir PATH        Override F5_LOCAL_RUNS_DIR (default: $F5_LOCAL_RUNS_DIR)
  --prefix NAME     Override F5_RUNS_PREFIX (default: $F5_RUNS_PREFIX)
  --no-install      Do not (re)install Python dependencies
  --                Pass all following args to the simulator

Env you can set instead of flags:
  F5_STORAGE=local
  F5_LOCAL_RUNS_DIR=/home/$USER/fuka-runs
  F5_RUNS_PREFIX=runs

Examples:
  $(basename "$0")
  $(basename "$0") --run-id run-test -- --steps 1000
  F5_LOCAL_RUNS_DIR=/srv/fuka/runs $(basename "$0")
USAGE
}

# -------- parse args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)       RUN_ID="${2:-}"; shift 2;;
    --run-id=*)     RUN_ID="${1#*=}"; shift;;
    --dir)          export F5_LOCAL_RUNS_DIR="${2:-}"; shift 2;;
    --dir=*)        export F5_LOCAL_RUNS_DIR="${1#*=}"; shift;;
    --prefix)       export F5_RUNS_PREFIX="${2:-}"; shift 2;;
    --prefix=*)     export F5_RUNS_PREFIX="${1#*=}"; shift;;
    --no-install)   SKIP_INSTALL=1; shift;;
    -h|--help)      usage; exit 0;;
    --)             shift; FORWARD_ARGS+=("$@"); break;;
    *)              FORWARD_ARGS+=("$1"); shift;;
  esac
done

# -------- ensure directories ----------
mkdir -p "$LOG_DIR" "$F5_LOCAL_RUNS_DIR/$F5_RUNS_PREFIX"

# -------- ensure venv ----------
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

# -------- install deps (unless skipped) ----------
if [[ "$SKIP_INSTALL" -eq 0 ]]; then
  echo "[deps] upgrading pip/setuptools/wheel"
  python -m pip install -U pip setuptools wheel >/dev/null
  if [[ -f "$REPO_ROOT/requirements.txt" ]]; then
    echo "[deps] installing from requirements.txt"
    python -m pip install -r "$REPO_ROOT/requirements.txt"
  else
    echo "[deps] requirements.txt not found; installing common stack"
    python -m pip install numpy pandas pyarrow plotly streamlit
  fi
fi

# -------- verify local backend shim present (non-fatal if missing) ----------
if [[ ! -f "$REPO_ROOT/.shim_injected.txt" ]]; then
  echo "[warn] .shim_injected.txt not found; ensure local backend patches are applied." >&2
fi

# -------- determine RUN_ID ----------
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="run-$(date +%Y%m%d-%H%M%S)"
fi

LOG="$LOG_DIR/$RUN_ID.log"
ART_DIR="$F5_LOCAL_RUNS_DIR/$F5_RUNS_PREFIX/$RUN_ID"

echo "[run] Storage=$F5_STORAGE  runs_dir=$F5_LOCAL_RUNS_DIR  prefix=$F5_RUNS_PREFIX"
echo "[run] RUN_ID=$RUN_ID"
echo "[run] Log: $LOG"
echo "[run] Artifacts (expected): $ART_DIR"
echo "[run] Forward args: ${FORWARD_ARGS[*]:-<none>}"

# -------- execute simulation ----------
set -x
python -u -m fuka5.run.sim_cli --run-id "$RUN_ID" "${FORWARD_ARGS[@]}" 2>&1 | tee "$LOG"
set +x

echo
echo "=== DONE ==="
echo "Log:        $LOG"
echo "Artifacts:  $ART_DIR"
echo "Tail log:   tail -f \"$LOG\""
