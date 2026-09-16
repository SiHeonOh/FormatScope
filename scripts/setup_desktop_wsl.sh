#!/usr/bin/env bash
# FormatScope: one-shot desktop bring-up inside WSL2 Ubuntu -- GPU lane + lane S.
#
#   bash scripts/setup_desktop_wsl.sh            # seed from the synced iCloud folder
#   FORMATSCOPE_SRC=/mnt/c/path/to/FORGE bash ...   # seed from a different folder
#   FORMATSCOPE_SRC=git bash ...                    # clone from origin instead
#
# The Mac's ~/Documents syncs to C:\Users\<you>\iCloudDrive\Documents via
# Desktop & Documents, so the lane-S work reaches this machine without being
# pushed. Falls back to a clone if that folder isn't there.
#
# Copies the project onto the Linux filesystem, builds the venv with a CUDA
# PyTorch, runs scripts/setup_lane_s_wsl.sh (Yosys + sky130 + OpenSTA), and
# runs the test suite. Safe to re-run: finished steps are skipped.
#
# Why it copies: the iCloud folder is reachable at /mnt/c but the Windows
# filesystem is ~10x slower under WSL, and two machines syncing one .git
# concurrently corrupts it. ~/FormatScope is a normal clone with the same
# origin, so `git pull` / `git push` still work.
set -euo pipefail

# The WSL username need not match the Windows one, so find the iCloud copy by
# globbing every Windows profile rather than guessing a name.
find_icloud() {
  local p
  for p in /mnt/c/Users/*/iCloudDrive/Documents/Projects/FORGE; do
    [ -d "$p" ] && { printf '%s' "$p"; return 0; }
  done
  return 1
}
SRC="${FORMATSCOPE_SRC:-}"
[ -n "$SRC" ] || SRC="$(find_icloud || echo git)"
REPO="${FORMATSCOPE_REPO_DIR:-$HOME/FormatScope}"
REPO_URL="${FORMATSCOPE_REPO_URL:-https://github.com/SiHeonOh/FormatScope.git}"
export FORMATSCOPE_MACHINE="${FORMATSCOPE_MACHINE:-desktop-wsl}"

step() { printf '\n==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }

step "Sanity"
grep -qi microsoft /proc/version || warn "this does not look like WSL"
[ "$(id -u)" -ne 0 ] || { echo "run as your normal user, not root" >&2; exit 1; }

step "System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq \
  build-essential git curl ca-certificates rsync \
  python3 python3-venv python3-dev python3-pip \
  tcl-dev swig bison flex libeigen3-dev zlib1g-dev cmake automake autoconf libtool

step "Project source -> $REPO"
if [ -d "$REPO/.git" ]; then
  echo "$REPO already exists; leaving it in place."
elif [ "$SRC" = "git" ]; then
  git clone "$REPO_URL" "$REPO"
elif [ -d "$SRC" ]; then
  echo "seeding from $SRC"
  # iCloud on Windows can keep files online-only; a dataless file reads as
  # empty rather than failing, so check a known file has real bytes first.
  if [ ! -s "$SRC/quant/formats.py" ]; then
    echo "$SRC/quant/formats.py is empty or missing -- the iCloud folder may not be downloaded." >&2
    echo "In Windows Explorer: right-click the FORGE folder -> 'Always keep on this device', wait for it to finish, then re-run." >&2
    exit 1
  fi
  rsync -a --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' \
           --exclude 'data' --exclude 'synth/out' --exclude 'sta/out' \
           "$SRC/" "$REPO/"
else
  echo "no source found at $SRC (set FORMATSCOPE_SRC, or use FORMATSCOPE_SRC=git)" >&2
  exit 1
fi
cd "$REPO"
git rev-parse --short HEAD 2>/dev/null || warn "not a git repo; history and push are unavailable"

step "Python venv + CUDA PyTorch"
[ -d "$REPO/.venv" ] || python3 -m venv "$REPO/.venv"
set +u; source "$REPO/.venv/bin/activate"; set -u
python -m pip install --quiet --upgrade pip
# On Linux the default PyPI torch wheel is the CUDA 12.x build, which is what
# the 4070 Ti needs; the Windows-side driver supplies the rest through /dev/dxg.
python -m pip install --quiet -e ".[train,test]"

step "CUDA check"
python - <<'PY'
import sys, torch
print("torch", torch.__version__, "| cuda build:", torch.version.cuda)
if not torch.cuda.is_available():
    print("\nFAIL: torch.cuda.is_available() is False.", file=sys.stderr)
    print("Update the Windows NVIDIA driver, run 'wsl --shutdown' in PowerShell,"
          " reopen Ubuntu, and re-run this script.", file=sys.stderr)
    sys.exit(1)
print("device:", torch.cuda.get_device_name(0))
PY

step "Lane S (Yosys, sky130 liberty, OpenSTA) via setup_lane_s_wsl.sh"
bash "$REPO/scripts/setup_lane_s_wsl.sh"

step "Test suite"
set +u; source "$HOME/tools/oss-cad-suite/environment"; set -u
source "$REPO/.venv/bin/activate"
make test-ci

step "Ready"
cat <<EOF

Environment is up at $REPO (machine name: $FORMATSCOPE_MACHINE).

Reproduce the committed accuracy numbers on the GPU:

    cd $REPO && bash scripts/reproduce_ptq.sh

That trains the FP32 ResNet-8 baseline for 60 epochs (~10-15 min on a 4070 Ti),
runs PTQ for all six formats, and diffs the result against results/accuracy.csv.
EOF
