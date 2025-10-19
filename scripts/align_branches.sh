#!/bin/bash
# Align main (sim fixes) with busbar-local (latest UI) into a new branch.
# Keeps sim from main: fuka5/, scripts/, configs/local.default.json
# Takes UI from busbar-local: app/
# Usage: bash scripts/align_branches.sh

set -euo pipefail

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "[ALIGN] ERROR: Run this inside the cloned repo." >&2
  exit 1
}

echo "[ALIGN] Fetching remotes..."
git fetch origin --prune

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ALIGN_BRANCH="align-${STAMP}"

echo "[ALIGN] Creating new branch from origin/main => ${ALIGN_BRANCH}"
git checkout -B "${ALIGN_BRANCH}" origin/main

echo "[ALIGN] Merging origin/busbar-local (no commit yet)…"
git merge --no-commit --no-ff origin/busbar-local || true

echo "[ALIGN] Resolving directory preferences..."
# Keep **sim** from main (ours)
git checkout --ours fuka5 || true
git checkout --ours scripts || true
git checkout --ours configs/local.default.json || true

# Take **UI** from busbar-local (theirs)
git checkout --theirs app || true

# Stage everything
git add -A

# Any remaining conflicts?
if ! git diff --check --quiet; then
  echo
  echo "[ALIGN] There are remaining textual conflicts. Resolve them, then run:"
  echo "        git add -A && git commit -m 'Align main(sim) + busbar-local(UI) [${STAMP}]'"
  exit 2
fi

echo "[ALIGN] Committing aligned merge..."
git commit -m "Align main(sim) + busbar-local(UI) [${STAMP}]"

echo "[ALIGN] Pushing new branch to origin/${ALIGN_BRANCH}..."
git push -u origin "${ALIGN_BRANCH}"

echo
echo "[ALIGN] ✅ Done."
echo "[ALIGN] New branch: ${ALIGN_BRANCH}"
echo "  - Sim (fuka5/, scripts/, configs/local.default.json) from **main**"
echo "  - UI (app/) from **busbar-local**"
echo
echo "[ALIGN] Next:"
echo "  git checkout ${ALIGN_BRANCH}"
echo "  # then run your usual sim/UI commands"