#!/usr/bin/env bash
# FormatScope: overnight multi-seed accuracy sweep. Fire-and-forget (risk #13).
#
#   mkdir -p logs && nohup bash scripts/seed_sweep.sh > logs/seed_sweep_$(date +%F_%H%M).log 2>&1 &
#   SEEDS="1 2 3" bash scripts/seed_sweep.sh        # a specific set
#   EPOCHS=5 SEEDS="1" bash scripts/seed_sweep.sh   # quick end-to-end check
#
# For each seed: train the FP32 baseline (models/train.py), then PTQ for all six
# formats (quant/eval.py). One seed is ~3 min on a 4070 Ti with the loader
# workers on, ~15 min without.
#
# Rows land in results/accuracy_seeds.csv, NOT results/accuracy.csv. The tool
# takes the last row per (format, stage) from accuracy.csv, so sweep rows there
# would silently replace the seed-0 numbers every figure is built from.
# accuracy.csv is snapshotted first and restored after every seed and on exit.
#
# Safe to re-run: seeds already present in accuracy_seeds.csv are skipped.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
[ -d .venv ] && { set +u; source .venv/bin/activate; set -u; }

export EPOCHS="${EPOCHS:-60}"
export FORMATSCOPE_WORKERS="${FORMATSCOPE_WORKERS:-8}"
SEEDS="${SEEDS:-$(seq 1 20 | tr '\n' ' ')}"

CSV="results/accuracy.csv"
OUT="results/accuracy_seeds.csv"
SNAP="$(mktemp)"
trap 'cp "$SNAP" "$CSV"; rm -f "$SNAP"' EXIT
cp "$CSV" "$SNAP"
[ -s "$OUT" ] || head -n 1 "$CSV" > "$OUT"

echo "seeds: $SEEDS | epochs: $EPOCHS | loader workers: $FORMATSCOPE_WORKERS"
python - <<'PY'
import torch
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu (this will be slow)")
PY

for s in $SEEDS; do
  if grep -q ",${s},models/checkpoints/fp32_seed${s}.pt," "$OUT"; then
    echo "seed $s already in $OUT, skipping"
    continue
  fi
  echo; echo "==> seed $s  start $(date '+%F %T')"
  base=$(wc -l < "$CSV")
  SEED="$s" python models/train.py
  SEED="$s" python quant/eval.py
  tail -n +$((base + 1)) "$CSV" >> "$OUT"     # move this seed's rows out
  cp "$SNAP" "$CSV"                            # and put the committed table back
  echo "==> seed $s  done  $(date '+%F %T')"
done

echo; echo "sweep complete: $(( $(wc -l < "$OUT") - 1 )) rows in $OUT"
