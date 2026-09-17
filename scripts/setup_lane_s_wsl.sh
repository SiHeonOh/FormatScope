#!/usr/bin/env bash
# Lane S environment on WSL2 Ubuntu (build-plan.md S3.2-S3.4, S3.7a/d).
#
#   bash scripts/setup_lane_s_wsl.sh [open_pdks-hash]
#
# Installs OSS CAD Suite 2026-09-09, the sky130 HD liberty through ciel, and
# OpenSTA (Docker image if Docker works, native build otherwise), records the
# versions in versions.lock, adds this machine to synth/libs.toml, then runs the
# smoke synthesis and smoke STA. Safe to re-run: finished steps are skipped.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$HOME/tools"
SUITE_DATE="2026-09-09"
SUITE_TGZ="oss-cad-suite-linux-x64-${SUITE_DATE//-/}.tgz"
SUITE_URL="https://github.com/YosysHQ/oss-cad-suite-build/releases/download/${SUITE_DATE}/${SUITE_TGZ}"
export PDK_ROOT="${PDK_ROOT:-$HOME/.ciel}"
SKY130_LIB="$PDK_ROOT/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
export FORMATSCOPE_MACHINE="${FORMATSCOPE_MACHINE:-desktop-wsl}"
PDK_HASH="${1:-}"

step() { printf '\n==> %s\n' "$*"; }
profile_line() { grep -qxF "$1" "$HOME/.bashrc" 2>/dev/null || echo "$1" >> "$HOME/.bashrc"; }
lock() {  # lock key value: record once in versions.lock
  touch "$REPO/versions.lock"
  grep -q "^$1=" "$REPO/versions.lock" || echo "$1=$2" >> "$REPO/versions.lock"
}

case "$REPO" in
  /mnt/*) echo "warning: $REPO is on the Windows filesystem; clone into ~/FormatScope instead (S3.2)" >&2 ;;
esac
mkdir -p "$TOOLS"

step "OSS CAD Suite $SUITE_DATE"
if [ ! -x "$TOOLS/oss-cad-suite/bin/yosys" ]; then
  curl -fL -o "$TOOLS/$SUITE_TGZ" "$SUITE_URL"
  tar xzf "$TOOLS/$SUITE_TGZ" -C "$TOOLS"
  rm -f "$TOOLS/$SUITE_TGZ"
fi
set +u; source "$TOOLS/oss-cad-suite/environment"; set -u
profile_line "source \$HOME/tools/oss-cad-suite/environment"
yosys -V
iverilog -V 2>&1 | sed -n 1p
lock oss-cad-suite "$SUITE_DATE"
lock yosys "$(yosys -V | awk '{print $2}')"

step "Python venv and ciel"
if [ ! -d "$REPO/.venv" ]; then
  python3 -m venv "$REPO/.venv"
fi
set +u; source "$REPO/.venv/bin/activate"; set -u
python -m pip install --quiet --upgrade pip ciel

step "sky130 HD liberty (ciel)"
if [ ! -f "$SKY130_LIB" ]; then
  if [ -z "$PDK_HASH" ]; then
    PDK_HASH="$(ciel ls-remote --pdk-family sky130 | grep -oE '[0-9a-f]{40}' | sed -n 1p)"
  fi
  [ -n "$PDK_HASH" ] || { echo "could not read a sky130 hash from ciel ls-remote; pass one as \$1" >&2; exit 1; }
  echo "enabling sky130 open_pdks $PDK_HASH"
  ciel enable --pdk-family sky130 "$PDK_HASH"
fi
ls "$SKY130_LIB"
profile_line "export PDK_ROOT=$PDK_ROOT"
profile_line "export FORMATSCOPE_MACHINE=$FORMATSCOPE_MACHINE"
if [ -n "$PDK_HASH" ]; then lock sky130-open_pdks "$PDK_HASH"; fi

step "OpenSTA"
if command -v sta >/dev/null 2>&1; then
  echo "using $(command -v sta)"
  if [ -d "$TOOLS/OpenSTA/.git" ]; then
    lock opensta "native:$(git -C "$TOOLS/OpenSTA" rev-parse --short HEAD)"
  else
    lock opensta "path:$(sta -version)"
  fi
elif command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  if ! docker image inspect opensta >/dev/null 2>&1; then
    [ -d "$TOOLS/OpenSTA" ] || git clone https://github.com/parallaxsw/OpenSTA "$TOOLS/OpenSTA"
    (cd "$TOOLS/OpenSTA" && docker build --file Dockerfile.ubuntu24.04 --tag opensta .)
  fi
  # Repo and PDK are mounted at identical paths so script paths resolve in the container.
  STA_CMD="docker run --rm -i -v $REPO:$REPO -v $PDK_ROOT:$PDK_ROOT -w $REPO opensta"
  export FORMATSCOPE_STA="$STA_CMD"
  profile_line "export FORMATSCOPE_STA='$STA_CMD'"
  lock opensta "docker:$(git -C "$TOOLS/OpenSTA" rev-parse --short HEAD 2>/dev/null || echo unknown)"
else
  sudo apt-get update
  sudo apt-get install -y build-essential cmake tcl-dev swig bison flex libeigen3-dev zlib1g-dev git automake autoconf libtool
  if [ ! -f "$TOOLS/cudd/lib/libcudd.a" ]; then
    [ -d "$TOOLS/cudd-src" ] || git clone https://github.com/ivmai/cudd "$TOOLS/cudd-src"
    # A clone gives the generated autotools files fresh mtimes, so make tries to
    # regenerate them with aclocal-1.14, which Ubuntu 24.04 does not ship.
    # Touch them in dependency order so they read as up to date.
    (cd "$TOOLS/cudd-src" && touch aclocal.m4 && sleep 1 && touch Makefile.in configure \
      && sleep 1 && find . -name config.h.in -exec touch {} + \
      && ./configure --prefix="$TOOLS/cudd" && make -j"$(nproc)" && make install)
  fi
  [ -d "$TOOLS/OpenSTA" ] || git clone https://github.com/parallaxsw/OpenSTA "$TOOLS/OpenSTA"
  (cd "$TOOLS/OpenSTA" && cmake -B build -DCUDD_DIR="$TOOLS/cudd" && cmake --build build -j"$(nproc)")
  mkdir -p "$HOME/.local/bin"
  ln -sf "$TOOLS/OpenSTA/build/sta" "$HOME/.local/bin/sta"
  export PATH="$HOME/.local/bin:$PATH"
  profile_line 'export PATH="$HOME/.local/bin:$PATH"'
  lock opensta "native:$(git -C "$TOOLS/OpenSTA" rev-parse --short HEAD)"
fi

step "synth/libs.toml [$FORMATSCOPE_MACHINE]"
if ! grep -qxF "[$FORMATSCOPE_MACHINE]" "$REPO/synth/libs.toml"; then
  printf '\n[%s]\nsky130hd = "%s"\n' "$FORMATSCOPE_MACHINE" "$SKY130_LIB" >> "$REPO/synth/libs.toml"
fi
grep -A2 -xF "[$FORMATSCOPE_MACHINE]" "$REPO/synth/libs.toml"

step "Smoke tests (S3.7a synthesis, S3.7d OpenSTA)"
cd "$REPO"
status=0
python synth/run_synth.py --smoke || status=1
if [ "$status" -eq 0 ]; then
  python sta/run_sta.py --smoke || status=1
fi
if [ "$status" -eq 0 ]; then
  echo "PASS: lane S environment ready. Open a new shell (or source ~/.bashrc) before the next run."
else
  echo "FAIL: see the logs in synth/out and sta/out" >&2
  exit 1
fi
