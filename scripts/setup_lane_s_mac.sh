#!/usr/bin/env bash
# Lane S environment on an Apple Silicon Mac (build-plan.md S3.2-S3.4, S3.7a/d).
#
#   bash scripts/setup_lane_s_mac.sh
#
# Installs the OSS CAD Suite, the sky130 HD liberty through ciel, and a native
# OpenSTA build, all at the versions pinned in versions.lock; adds this machine
# to synth/libs.toml; then runs the smoke synthesis and smoke STA. Nothing goes
# under /usr/local: tools live in ~/tools, the PDK in ~/.ciel, Homebrew supplies
# the build dependencies. Safe to re-run: finished steps are skipped.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$HOME/tools"
LOCK="$REPO/versions.lock"
export PDK_ROOT="${PDK_ROOT:-$HOME/.ciel}"
SKY130_LIB="$PDK_ROOT/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
export FORMATSCOPE_MACHINE="${FORMATSCOPE_MACHINE:-$(scutil --get LocalHostName 2>/dev/null || hostname -s)}"

step() { printf '\n==> %s\n' "$*"; }
profile_line() { grep -qxF "$1" "$HOME/.zprofile" 2>/dev/null || echo "$1" >> "$HOME/.zprofile"; }
pinned() { grep "^$1=" "$LOCK" | cut -d= -f2-; }
# `brew --prefix <formula>` prints a path whether or not the formula is
# installed, so every dependency is checked with `brew list` instead.
brew_need() { brew list --versions "$1" >/dev/null 2>&1 || brew install -q "$1"; }

[ "$(uname -s)-$(uname -m)" = "Darwin-arm64" ] || { echo "this script is for Apple Silicon macOS" >&2; exit 1; }
command -v brew >/dev/null || { echo "Homebrew is required: https://brew.sh" >&2; exit 1; }
# Resolve the system Python before the OSS CAD Suite goes on PATH: the suite
# bundles its own python3, which has no pip, and it would shadow this one.
SYS_PY="$(command -v python3)"
[ -f "$LOCK" ] || { echo "$LOCK missing: the lane S versions are pinned there" >&2; exit 1; }
SUITE_DATE="$(pinned oss-cad-suite)"; PDK_HASH="$(pinned sky130-open_pdks)"
STA_COMMIT="$(pinned opensta | sed -n 's/^native://p')"
[ -n "$SUITE_DATE" ] && [ -n "$PDK_HASH" ] && [ -n "$STA_COMMIT" ] || {
  echo "versions.lock needs oss-cad-suite=, sky130-open_pdks=, opensta=native:<commit>" >&2; exit 1; }
mkdir -p "$TOOLS"

step "OSS CAD Suite $SUITE_DATE (Yosys, ABC, Icarus)"
SUITE_TGZ="oss-cad-suite-darwin-arm64-${SUITE_DATE//-/}.tgz"
if [ ! -x "$TOOLS/oss-cad-suite/bin/yosys" ]; then
  curl -fL -o "$TOOLS/$SUITE_TGZ" "https://github.com/YosysHQ/oss-cad-suite-build/releases/download/${SUITE_DATE}/${SUITE_TGZ}"
  tar xzf "$TOOLS/$SUITE_TGZ" -C "$TOOLS" && rm -f "$TOOLS/$SUITE_TGZ"
  xattr -dr com.apple.quarantine "$TOOLS/oss-cad-suite" 2>/dev/null || true
fi
set +u; source "$TOOLS/oss-cad-suite/environment"; set -u
profile_line "source \$HOME/tools/oss-cad-suite/environment"
# sed, not head: head closes the pipe early and the writer dies with SIGPIPE,
# which pipefail turns into a script exit.
yosys -V | sed -n 1p; iverilog -V 2>&1 | sed -n 1p

step "Python venv and ciel"
[ -x "$REPO/.venv/bin/pip" ] || { rm -rf "$REPO/.venv"; "$SYS_PY" -m venv "$REPO/.venv"; }
set +u; source "$REPO/.venv/bin/activate"; set -u
python -m pip install --quiet --upgrade pip ciel
python -m pip install --quiet -e "$REPO[test]"

step "sky130 HD liberty (ciel, open_pdks $PDK_HASH)"
if [ ! -f "$SKY130_LIB" ]; then
  ciel enable --pdk-family sky130 --include-libraries sky130_fd_sc_hd "$PDK_HASH" \
    || ciel enable --pdk-family sky130 "$PDK_HASH"
fi
ls "$SKY130_LIB"
profile_line "export PDK_ROOT=$PDK_ROOT"
profile_line "export FORMATSCOPE_MACHINE=$FORMATSCOPE_MACHINE"

step "OpenSTA $STA_COMMIT (native build)"
STA_BIN="$TOOLS/OpenSTA/build/sta"
if [ ! -x "$STA_BIN" ]; then
  for f in cmake swig bison flex eigen tcl-tk@8; do brew_need "$f"; done
  export PATH="$(brew --prefix bison)/bin:$(brew --prefix flex)/bin:$PATH"
  if [ ! -f "$TOOLS/cudd/lib/libcudd.a" ]; then
    [ -d "$TOOLS/cudd-src" ] || git clone -q https://github.com/ivmai/cudd "$TOOLS/cudd-src"
    # A fresh clone gives the generated autotools files new mtimes, so make
    # would try to regenerate them; touch them in dependency order instead.
    (cd "$TOOLS/cudd-src" && touch aclocal.m4 && sleep 1 && touch Makefile.in configure \
      && sleep 1 && find . -name config.h.in -exec touch {} + \
      && ./configure --prefix="$TOOLS/cudd" >/dev/null && make -j"$(sysctl -n hw.ncpu)" >/dev/null && make install >/dev/null)
  fi
  [ -d "$TOOLS/OpenSTA/.git" ] || git clone -q https://github.com/parallaxsw/OpenSTA "$TOOLS/OpenSTA"
  (cd "$TOOLS/OpenSTA" && git checkout -q "$STA_COMMIT" && rm -rf build \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release \
         -DCUDD_DIR="$TOOLS/cudd" \
         -DEigen3_DIR="$(brew --prefix eigen)/share/eigen3/cmake" \
         -DBISON_EXECUTABLE="$(brew --prefix bison)/bin/bison" \
         -DFLEX_EXECUTABLE="$(brew --prefix flex)/bin/flex" \
         -DCMAKE_CXX_FLAGS="-I$(brew --prefix flex)/include" \
         -DTCL_LIBRARY="$(brew --prefix tcl-tk@8)/lib/libtcl8.6.dylib" \
         -DTCL_HEADER="$(brew --prefix tcl-tk@8)/include/tcl-tk/tcl.h" \
         -DCMAKE_PREFIX_PATH="$(brew --prefix flex);$(brew --prefix tcl-tk@8)" >/dev/null \
    && cmake --build build -j"$(sysctl -n hw.ncpu)" >/dev/null)
fi
mkdir -p "$HOME/.local/bin" && ln -sf "$STA_BIN" "$HOME/.local/bin/sta"
export PATH="$HOME/.local/bin:$PATH"
profile_line 'export PATH="$HOME/.local/bin:$PATH"'
echo "OpenSTA $(sta -version 2>&1 | sed -n 1p)"

step "synth/libs.toml [$FORMATSCOPE_MACHINE]"
if ! grep -qxF "[$FORMATSCOPE_MACHINE]" "$REPO/synth/libs.toml"; then
  printf '\n[%s]\nsky130hd = "%s"\n' "$FORMATSCOPE_MACHINE" '$HOME/.ciel/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib' >> "$REPO/synth/libs.toml"
fi
grep -A1 -xF "[$FORMATSCOPE_MACHINE]" "$REPO/synth/libs.toml"

step "Smoke tests (S3.7a synthesis, S3.7d OpenSTA)"
cd "$REPO"
python synth/run_synth.py --smoke && python sta/run_sta.py --smoke \
  && echo "PASS: lane S environment ready. Open a new shell (or source ~/.zprofile) before the next run." \
  || { echo "FAIL: see the logs in synth/out and sta/out" >&2; exit 1; }
