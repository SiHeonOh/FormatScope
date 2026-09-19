#!/usr/bin/env bash
# Lane R/V environment on WSL2 Ubuntu (build-plan.md S3.1, S3.2, S3.7b).
#
#   bash scripts/setup_rtl_wsl.sh
#
# Installs system Python and Icarus Verilog (the same apt build CI uses), OSS CAD
# Suite 2026-09-09 (Yosys, Verilator, GTKWave), clones the repo onto the Linux
# filesystem, builds a venv with cocotb 2.x + pytest + numpy + CPU torch, then
# runs the cocotb hello smoke test on Icarus. Safe to re-run: finished steps are skipped.
set -euo pipefail

REPO="${FORMATSCOPE_REPO_DIR:-$HOME/FormatScope}"
REPO_URL="${FORMATSCOPE_REPO_URL:-https://github.com/SiHeonOh/FormatScope.git}"
TOOLS="$HOME/tools"
SUITE_DATE="2026-09-09"
SUITE_TGZ="oss-cad-suite-linux-x64-${SUITE_DATE//-/}.tgz"
SUITE_URL="https://github.com/YosysHQ/oss-cad-suite-build/releases/download/${SUITE_DATE}/${SUITE_TGZ}"
GCM="/mnt/c/Program Files/Git/mingw64/bin/git-credential-manager.exe"

step() { printf '\n==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
profile_line() { grep -qxF "$1" "$HOME/.bashrc" 2>/dev/null || echo "$1" >> "$HOME/.bashrc"; }

step "Sanity"
grep -qi microsoft /proc/version || warn "this does not look like WSL"
[ "$(id -u)" -ne 0 ] || { echo "run as your normal user, not root" >&2; exit 1; }

step "System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq \
  build-essential git curl ca-certificates make \
  python3 python3-venv python3-dev python3-pip iverilog

step "Git identity and credentials (shared with Windows)"
if [ -z "$(git config --global credential.helper || true)" ] && [ -x "$GCM" ]; then
  git config --global credential.helper "${GCM// /\\ }"
fi
for key in user.name user.email; do
  if [ -z "$(git config --global "$key" || true)" ] && command -v git.exe >/dev/null 2>&1; then
    val="$(git.exe config --global "$key" 2>/dev/null | tr -d '\r' || true)"
    [ -n "$val" ] && git config --global "$key" "$val"
  fi
done
echo "user.name=$(git config --global user.name || echo '<unset>')  user.email=$(git config --global user.email || echo '<unset>')"

step "Repository -> $REPO"
if [ -d "$REPO/.git" ]; then
  echo "already cloned; leaving it in place"
else
  git clone "$REPO_URL" "$REPO"
fi

step "OSS CAD Suite $SUITE_DATE"
mkdir -p "$TOOLS"
if [ ! -x "$TOOLS/oss-cad-suite/bin/iverilog" ]; then
  curl -fL -o "$TOOLS/$SUITE_TGZ" "$SUITE_URL"
  tar xzf "$TOOLS/$SUITE_TGZ" -C "$TOOLS"
  rm -f "$TOOLS/$SUITE_TGZ"
fi
# Append the suite to PATH instead of sourcing its environment script: that
# script puts the suite's own python and Icarus first, which breaks venv
# creation and makes cocotb embed the wrong interpreter. Ubuntu's python and
# iverilog stay first; yosys, verilator, and gtkwave come from the suite.
sed -i '\|^source \$HOME/tools/oss-cad-suite/environment$|d' "$HOME/.bashrc"
export PATH="$PATH:$TOOLS/oss-cad-suite/bin"
profile_line 'export PATH="$PATH:$HOME/tools/oss-cad-suite/bin"'

step "Python venv"
# Pin Ubuntu's python explicitly; the suite's bundled python cannot bootstrap
# pip into a venv. A venv without pip is a leftover from a failed run, so
# rebuild it.
if [ ! -x "$REPO/.venv/bin/pip" ]; then
  rm -rf "$REPO/.venv"
  /usr/bin/python3 -m venv "$REPO/.venv"
fi
set +u; source "$REPO/.venv/bin/activate"; set -u
python -m pip install --quiet --upgrade pip
python -m pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install --quiet "cocotb~=2.0" pytest numpy pandas matplotlib
python -c "import torch, numpy"

step "Smoke test (S3.7b): cocotb hello on Icarus"
SMOKE="$(mktemp -d)"
cat > "$SMOKE/hello.v" <<'EOF'
module hello (
  input  wire       clk,
  input  wire       rst_n,
  input  wire [7:0] a,
  input  wire [7:0] b,
  output reg  [8:0] y
);
  always @(posedge clk) begin
    if (!rst_n) y <= 9'd0;
    else        y <= a + b;
  end
endmodule
EOF
cat > "$SMOKE/test_hello.py" <<'EOF'
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge
from cocotb_tools.runner import get_runner

HERE = Path(__file__).resolve().parent


@cocotb.test()
async def hello_add(dut):
    Clock(dut.clk, 10, unit="ns").start()
    dut.rst_n.value = 0
    dut.a.value = 0
    dut.b.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    dut.a.value = 200
    dut.b.value = 100
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    assert int(dut.y.value) == 300, f"y = {int(dut.y.value)}"


def test_hello():
    runner = get_runner("icarus")
    runner.build(sources=[HERE / "hello.v"], hdl_toplevel="hello",
                 build_dir=HERE / "sim_build", build_args=["-g2012"], always=True,
                 timescale=("1ns", "1ps"))
    runner.test(hdl_toplevel="hello", test_module="test_hello", test_dir=HERE)
EOF
(cd "$SMOKE" && python -m pytest -q test_hello.py) || { echo "smoke test failed; files kept in $SMOKE" >&2; exit 1; }
rm -rf "$SMOKE"

step "Versions (record these in versions.lock)"
echo "oss-cad-suite=$SUITE_DATE"
yosys -V
iverilog -V 2>&1 | sed -n 1p
verilator --version
python --version
echo "cocotb $(cocotb-config --version)"

cat <<EOF

PASS: RTL environment ready. For each new shell:

    cd $REPO && source .venv/bin/activate

(~/.bashrc appends OSS CAD Suite to PATH, so Ubuntu's python and iverilog
stay first. Open a new shell once for that to take effect.)
EOF
