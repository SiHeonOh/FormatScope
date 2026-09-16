#!/usr/bin/env bash
# FormatScope: reproduce the committed FP32 + PTQ accuracy table on this machine.
#
#   bash scripts/reproduce_ptq.sh              # 60 epochs, the committed setting
#   EPOCHS=5 bash scripts/reproduce_ptq.sh     # quick end-to-end smoke
#   TOL=3.0 bash scripts/reproduce_ptq.sh      # loosen the per-row tolerance
#
# Trains the FP32 ResNet-8 baseline, runs PTQ for all six formats, and diffs
# the result against results/accuracy.csv.
#
# Both models/train.py and quant/eval.py APPEND to results/accuracy.csv. To
# keep the committed table clean, this snapshots it first, splits the new rows
# into results/accuracy_$FORMATSCOPE_MACHINE.csv, and restores the original --
# so a reproduction run leaves the tracked file untouched.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export FORMATSCOPE_MACHINE="${FORMATSCOPE_MACHINE:-desktop-wsl}"
export EPOCHS="${EPOCHS:-60}"
TOL="${TOL:-2.0}"

CSV="results/accuracy.csv"
OUT="results/accuracy_${FORMATSCOPE_MACHINE}.csv"
SNAPSHOT="$(mktemp)"
trap 'rm -f "$SNAPSHOT"' EXIT

step() { printf '\n==> %s\n' "$*"; }

[ -d .venv ] && { set +u; source .venv/bin/activate; set -u; }

step "Preflight"
python - <<'PY'
import sys, torch
print("torch", torch.__version__)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", torch.cuda.get_device_name(0) if dev == "cuda" else "cpu")
if dev == "cpu":
    print("\nwarning: no CUDA -- a 60-epoch run will take hours on CPU.", file=sys.stderr)
PY

cp "$CSV" "$SNAPSHOT"
BASE_LINES=$(wc -l < "$SNAPSHOT")

step "FP32 baseline ($EPOCHS epochs, seed 0)"
time python models/train.py

step "PTQ, all six formats"
time python quant/eval.py

step "Splitting out this run's rows -> $OUT"
{ head -n 1 "$SNAPSHOT"; tail -n +$((BASE_LINES + 1)) "$CSV"; } > "$OUT"
cp "$SNAPSHOT" "$CSV"          # restore the committed table
echo "wrote $OUT; $CSV restored to its committed contents"

step "Comparison against the committed table (tolerance +/-$TOL points)"
python scripts/compare_accuracy.py --baseline "$CSV" --candidate "$OUT" --tol "$TOL"
