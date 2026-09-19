# FormatScope Build Plan

**Purdue Chips & AI Hackathon, build window Sep 7–19, 2026. This plan started Thu Sep 10, was re-planned on Tue Sep 15 (Section 10), and ends with the video submitted on Sat Sep 19.**

Team Hidden Bit: Si Heon Oh (RTL + cocotb), Seungmin Nam (quantization + training), Rakshita Gupta (synthesis, OpenSTA, the `formatscope` tool, and the GPU machines). Claude drafts code in all three lanes; the lane owner reviews, runs, and commits it.

---

## 0. How to use this document

This is the single to-do list for the whole project. Read Sections 0–2 once (what we are building and why). Then live in your own lane section (4, 5, 6, 7, or 8) and the day-by-day calendar (Section 10). Every step has a **Do**, a **Done when**, and where it matters a **Watch out**. If a step's "Done when" cannot be shown on a screen, it is not done.

Rules the whole team agreed on:

1. **The submitted proposal governs.** The submitted proposal PDF is the spec the judges read. The playbook PDF is a verified tool-command reference only; where they disagree, the proposal wins. Both are internal documents and are kept outside this repository.
2. **No RTL merges to `main` without its cocotb test green.** No exceptions, including the last night.
3. **Results are committed, code is reproducible.** Every number in the README regenerates from `make` targets. CSVs in `results/` are committed; netlists, checkpoints, and simulator builds are not.
4. **Owners commit their own lane.** Claude writes first drafts on a branch; the owner reads every line, runs it, and commits under their own name. Commit messages are plain imperative sentences with no AI attribution of any kind.
5. **The deliverable is the full proposal scope.** Five verified DP32 units with sky130 netlists and area/timing reports at three delay targets and two alignment-window settings; six quantized configurations each measured after PTQ and after QAT; the stage breakdown; the fidelity check; the ASAP7 rerun for H3; the tool, the plots, and the video. The floor is a contingency, not a target: if the schedule slips we cut in the proposal's own order (ASAP7 rerun first, then the third delay target, then the fine-tuning pass on the MX formats), and we never fall below INT4, INT8, and FP8 verified end to end at two matched delay targets with PTQ accuracy for all five formats.
6. **Daily 15-minute sync** at a fixed time. Each owner says: done yesterday, doing today, blocked on. Rakshita's hours are limited, so her lane is drafted to be run, not written, by her.

---

## 1. What we are building (the spec, restated precisely)

### 1.1 The one-paragraph version

Choosing a number format for an AI accelerator trades model accuracy against silicon cost, and today only the accuracy half is reproducible. FormatScope measures both halves on the same model with open tools: a ResNet-8 on CIFAR-10 quantized to INT4, INT8, FP8-E4M3, MXINT8, and MXFP4, and one 32-lane dot-product unit (DP32) per format synthesized to the SkyWater 130 nm cell library at three delay targets and timed with OpenSTA. A command-line tool joins the two and plots accuracy against area at each delay target, then recommends the smallest unit inside an accuracy margin or the most accurate unit under an area budget.

### 1.2 Deliverables (from the proposal, verbatim in spirit)

| # | Deliverable | Lane | Rubric line it serves |
|---|-------------|------|-----------------------|
| D1 | Public repository with the `formatscope` tool and one-command regeneration | Tool (Rakshita) | Implementation quality 20% |
| D2 | Five verified DP32 designs: RTL, sky130 netlists, area and timing reports at each delay target, committed under `results/netlists/` and `results/reports/` at the `v1.0` tag | RTL (Si Heon), Synth (Rakshita) | Technical merit 30%, Results 20% |
| D3 | Six quantized ResNet-8 configurations with measured accuracy, after PTQ and after a 5-epoch QAT | Quant (Seungmin) | Results 20%, Relevance 20% |
| D4 | Accuracy-vs-area plots at each delay target, plus the per-stage area breakdown | Tool (Rakshita) | Results 20% |
| D5 | 3–5 minute demo video: live synthesis, passing test suite, the frontier | Everyone | Video clarity 10% |

The organizers require only the video. The repository must be public by the time the video is submitted because the video and proposal both promise it.

### 1.3 The five formats, exactly

All definitions live in one file, `quant/formats.py`, and the RTL decoders are checked against that file exhaustively. If you are unsure what a code means, that file is the answer.

| Format ID | Bits | Encoding | Values | Notes |
|-----------|------|----------|--------|-------|
| `int4` | 4 | two's complement | codes −7…+7 (−8 unused) | symmetric, so the quantizer is sign-symmetric |
| `int8` | 8 | two's complement | codes −127…+127 (−128 unused) | same reason |
| `fp8e4m3` | 8 | sign, 4-bit exponent (bias 7), 3-bit mantissa | max finite ±448; subnormals at E=0 are ±2⁻⁶·(M/8); the single code S.1111.111 is NaN; **no infinities** | OCP FP8 spec. Overflow policy: saturate to ±448 |
| `mxint8` | 8 per element + 8-bit shared scale | element = INT8 two's complement with an implicit 2⁻⁶ scale; block of 32 elements shares one E8M0 scale | element range −2…+1.984375; block value = 2^(X−127) · e/64 | OCP MX v1.0. E8M0: unsigned 8-bit exponent, bias 127, code 0xFF = NaN, no zero, no infinity |
| `mxfp4` | 4 per element + 8-bit shared scale | element = E2M1 (sign, 2-bit exponent bias 1, 1-bit mantissa) | element values ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}; block of 32 shares one E8M0 scale | OCP MX v1.0 |

Shared-scale rule for the MX formats: `shared_exp = floor(log2(max_i |V_i|)) − emax_elem`, where `emax_elem` is the exponent of the largest normal element value (2 for E2M1, 0 for INT8 — confirm against the OCP MX v1.0 spec table on Day 0 and record it in `docs/decisions.md`). Elements are divided by 2^shared_exp, rounded to nearest-even, and clamped to the element max. Edge cases: an all-zero block gets scale code 127 (2⁰) and zero elements; a reduction dimension that is not a multiple of 32 is zero-padded to 32 (padding contributes 0 to the dot product).

A sixth, accuracy-only configuration, `int4_b32` (INT4 elements with block-32 scales), separates how much of MXFP4's advantage comes from block scaling and how much from the E2M1 encoding. It has no hardware unit.

### 1.4 The accuracy pipeline (what "accuracy" means here)

- **Model:** ResNet-8 on CIFAR-10, expected 85–88% top-1 in FP32. Whatever the FP32 number is, it is the ceiling we report against.
- **What gets quantized:** weights *and* activations of every convolution and linear layer, first and last layers included, fake-quantized to the format the hardware consumes.
- **Scales:** integer and FP8 formats use symmetric per-output-channel weight scales and per-tensor activation scales, calibrated on 512 training images. The per-channel scale is applied during requantization *after* the accumulator, a stage every format needs and that we exclude from the hardware comparison uniformly. MX block scales come from each block's largest magnitude.
- **Rounding:** round-to-nearest-even with saturation, everywhere.
- **Two numbers per configuration:** after post-training quantization (PTQ), and again after a five-epoch quantization-aware fine-tune (QAT) with the straight-through estimator.
- **Numerical fidelity check:** PyTorch fake-quantization accumulates in FP32, but the fused FP8 and MX hardware rounds once per 32 products. For INT4 and INT8 the two are exactly equal. For FP8 and the MX formats they are not, so we push one real layer's dot products through both paths and report the measured difference instead of assuming it is negligible.

### 1.5 The silicon pipeline (what "area" and "delay" mean here)

- **One DP32 per format:** 32 multipliers, an adder tree, an accumulator register. The 32-lane boundary is deliberate: it puts the microscaling formats' shared-scale logic inside the comparison, where a single-MAC study would hide it, and the accumulator is part of the format's cost.
- **INT4 / INT8:** exact products summed in an integer tree into an INT32 register.
- **FP8-E4M3, fused stage:** each lane produces an 8-bit significand product and a 5-bit exponent sum; the 32 products and the current accumulator (33 terms) are aligned to their maximum exponent in a window with a sticky bit, summed in one integer tree, normalized and rounded once (RNE) into an FP32-format register. The alignment window width is an RTL parameter and we sweep two settings rather than defend one.
- **MXINT8:** reuses the INT8 datapath for the exact 21-bit block sum, adds the combined E8M0 exponent (the elements' implicit 2⁻⁶ folds into the exponent), and enters the same fused stage with two terms instead of 33.
- **MXFP4:** E2M1 products are multiples of 0.25 no larger than 36, so they fit exactly in 9 bits; the block sum is an exact 14-bit integer tree before the same two-term fused stage.
- **Synthesis:** one Yosys script (`synth → dfflibmap → abc`) per unit against `sky130_fd_sc_hd` at the tt / 25 °C / 1.80 V corner, at three delay targets: unconstrained plus two clock periods shared by all units. OpenSTA times each netlist. We report **area at matched delay**, because an adder tree can always be traded for speed.
- **Stage breakdown:** a hierarchical synthesis run reports area per stage (multiply, align, tree, normalize/accumulate) so the plot can say *why* a format costs what it costs.
- **Claims and limits:** pre-layout, cell-level numbers with no interconnect. ABC's constraint-driven mapping varies between runs, so the claim is the *ranking* of formats, not absolute µm², and we report the spread across runs. Power is out of scope.

### 1.6 Hypotheses (reported either way)

- **H1.** At matched delay the FP8-E4M3 unit is larger than INT8, but by less than Qualcomm's 183% single-MAC gate estimate, because a fused DP32 normalizes and rounds once per 32 products. The stage breakdown shows where the penalty sits.
- **H2.** Block scaling is the cheap lever: MXINT8 costs a small increment over INT8, and MXFP4 costs little more than INT4 while closing most of the accuracy gap to INT8 after fine-tuning.
- **H3.** The format ranking is unchanged on ASAP7, an open predictive 7 nm PDK the same scripts target as a second library. (First thing cut if behind.)

### 1.7 Judging criteria and how each is earned

| Criterion | Weight | What earns it in our submission |
|-----------|--------|--------------------------------|
| Technical merit and innovation | 30% | Fused DP32 design per format, matched-delay comparison, stage breakdown, MX formats on an open frontier for the first time |
| Relevance to AI and chips | 20% | Theme 1 (quantization) × Theme 3 (compute datapath) connected on one design |
| Results and evidence | 20% | Committed CSVs, bit-exact test logs, the frontier at three delay targets, H1/H2 answered with numbers |
| Implementation quality | 20% | One-command reproduction, pinned versions, exhaustive tests, clean repo, readable RTL |
| Demo video clarity | 10% | Hook → live tool → chart → proof → findings → limits, in under five minutes |

---

## 2. Repository layout and conventions

Local checkout: `/Users/rakki/Documents/Projects/FORGE` on Rakshita's Mac, cloned by the others. Remote: `https://github.com/SiHeonOh/FormatScope` (private now, flipped to public on submission day).

```
FormatScope/
├── README.md                 # judge-facing; written last, in the order given in §9.1
├── LICENSE                   # decision D16 (recommend MIT)
├── pyproject.toml            # package metadata + pinned Python deps
├── versions.lock             # exact OSS CAD Suite date, sky130 PDK hash, pip freeze
├── Makefile                  # env, train, quant, test, synth, sta, plot, demo
├── .gitignore                # checkpoints/, synth/out/, sta/out/, sim_build/, *.vcd, *.fst, __pycache__/, .venv/
├── .github/workflows/ci.yml  # pytest: exhaustive decoders + short DP32 tests on every PR
├── docs/
│   ├── build-plan.md         # this document
│   ├── decisions.md          # decision log D1…Dn with the date and who decided
│   ├── fused-stage.md        # bit-width derivation for the FP8/MX fused stage (§5.4)
│   └── references.md         # OCP FP8, OCP MX v1.0, arXiv 2303.17951, 2209.05433, 2310.10537, 2303.02347
├── models/
│   ├── resnet8.py            # the network (decision D9)
│   ├── data.py               # CIFAR-10 loaders, augmentation, calibration subset (512 images, fixed seed)
│   ├── train.py              # FP32 training → models/checkpoints/fp32.pt
│   └── checkpoints/          # gitignored
├── quant/
│   ├── formats.py            # SINGLE SOURCE OF TRUTH: encode/decode/quantize for every format
│   ├── fakequant.py          # QConv2d / QLinear wrappers, BN folding, STE
│   ├── calibrate.py          # activation ranges from 512 images
│   ├── eval.py               # PTQ accuracy → results/accuracy.csv
│   ├── qat.py                # 5-epoch fine-tune → results/accuracy.csv (stage=qat)
│   ├── fidelity.py           # one layer through PyTorch path vs DP32 reference → results/fidelity.csv
│   └── tables/               # exported decode tables (JSON) consumed by tb/
├── rtl/
│   ├── common/
│   │   ├── adder_tree.v      # generate-based balanced tree, parameterized N and W
│   │   ├── lzc.v             # leading-zero counter
│   │   ├── fused_stage.v     # align (window + sticky) → tree → normalize → RNE → FP32 pack; N_TERMS param
│   │   ├── decode_fp8e4m3.v
│   │   ├── decode_e2m1.v
│   │   └── decode_e8m0.v
│   ├── int8/dp32_int8.v
│   ├── int4/dp32_int4.v
│   ├── fp8e4m3/dp32_fp8e4m3.v
│   ├── mxint8/dp32_mxint8.v
│   └── mxfp4/dp32_mxfp4.v
├── tb/
│   ├── conftest.py           # runner helpers (cocotb 2.0 Python runner, SIM=icarus)
│   ├── refs.py               # NumPy bit-exact reference models mirroring the RTL
│   ├── test_decode.py        # exhaustive decoder tests (256 / 16 / 256 / 16 codes)
│   ├── test_dp32_int.py      # INT4/INT8: 10,000 seeded vectors + corners
│   ├── test_dp32_fp8.py      # FP8 fused: 10,000 seeded vectors + corners
│   └── test_dp32_mx.py       # MXINT8/MXFP4: 10,000 seeded vectors + corners
├── synth/
│   ├── synth_flat.ys.template   # area runs; {LIB} {FMT} {UNIT} {ALIGN_W} {DTARGET} substituted
│   ├── synth_hier.ys.template   # stage-breakdown runs (no -flatten)
│   ├── run_synth.py             # renders, runs yosys, parses stat → results/area.csv, results/breakdown.csv
│   ├── libs.toml                # absolute paths to sky130 and ASAP7 liberty files per machine
│   └── out/                     # gitignored netlists + logs
├── sta/
│   ├── sta.tcl.template
│   ├── run_sta.py               # runs OpenSTA per netlist → results/timing.csv
│   └── out/                     # gitignored
├── tool/formatscope/
│   ├── cli.py                   # run / plot / recommend / table / demo
│   ├── data.py                  # loads and joins the CSVs
│   ├── pareto.py                # frontier + knee
│   ├── recommend.py
│   └── plots.py                 # matplotlib only; PNG + SVG
├── results/                     # COMMITTED
│   ├── accuracy.csv  area.csv  timing.csv  breakdown.csv  fidelity.csv
│   ├── figures/                 # frontier_<lib>_<target>.png/svg, breakdown.png, results_table.png
│   ├── netlists/                # final sky130 (and ASAP7) netlists per unit × target × window, copied by `make freeze`
│   └── reports/                 # the matching yosys stat and OpenSTA report_checks outputs
└── video/
    ├── script.md                # shot list with timings (§9.3)
    └── qa.md                    # showcase Q&A answers (§9.5)
```

**Conventions**

- Format IDs are the fixed strings `int4, int8, fp8e4m3, mxint8, mxfp4, int4_b32`. Unit ID is `dp32`. Library IDs are `sky130hd` and `asap7`. Delay-target IDs are `unc`, `t1`, `t2`.
- Seeds are 0 everywhere unless a column says otherwise. Every CSV row that depends on a random choice carries a `seed` column.
- CSV schemas are in Appendix A. A CSV is append-only through the scripts; hand edits are forbidden.
- Verilog-2005 only (Yosys reads it natively). Flatten 2-D ports into 1-D vectors (`a_flat[32*W-1:0]`). No latches: every `always @*` assigns every output on every path. Wrap signed operands in `$signed()`. Register all outputs. Build trees with `generate`.
- Python 3.11, `numpy`, `torch`, `torchvision`, `cocotb~=2.0`, `pytest`, `matplotlib`, `pandas`. Pin with `pip freeze > versions.lock` on Day 0.
- Branch names: `sh/<topic>`, `sn/<topic>`, `rg/<topic>`. Pull requests into `main`; the CI must be green; one other person skims RTL PRs.
- Makefile targets, thin so judges reproduce trivially: `make env`, `make train`, `make quant`, `make test`, `make synth`, `make sta`, `make plot`, `make demo`, and `make freeze` (copies the final netlists and reports from the gitignored working directories into `results/` for commit).

---

## 3. Environment setup (Day 0, everyone)

Each member follows the branch for their machine. Record the exact versions in `versions.lock` the first time it works.

### 3.1 Common (all machines)

**Do**
```bash
git clone https://github.com/SiHeonOh/FormatScope.git && cd FormatScope
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install "cocotb~=2.0" pytest numpy pandas matplotlib
pip install torch torchvision          # CPU build is fine on the Macs; CUDA build on the GPU machines
cocotb-config --version                # expect 2.0.x
```
**Done when** `cocotb-config --version` prints 2.0.x and `python -c "import torch, numpy"` is silent.

### 3.2 OSS CAD Suite (Yosys, ABC, Icarus Verilog, Verilator, GTKWave, SMT solvers)

The suite does **not** include OpenSTA. Pin one dated release for the whole team; the current one is 2026-09-09.

| Machine | Asset | Notes |
|---------|-------|-------|
| Apple Silicon Mac (Si Heon, Seungmin, Rakshita's M5) | `oss-cad-suite-darwin-arm64-20260909.tgz` (520 MB) | macOS may quarantine the binaries; run `xattr -dr com.apple.quarantine oss-cad-suite` after extracting |
| Windows laptop or desktop (Seungmin; Rakshita's 4070 Ti / 5080) | Use **WSL2 Ubuntu** and the `oss-cad-suite-linux-x64-20260909.tgz` (742 MB) inside it | Keep the repo on the Linux filesystem (`~/FormatScope`), never under `/mnt/c/…`; small-file I/O there is dramatically slower and cripples the sim/synth loop |
| Linux | `oss-cad-suite-linux-x64-20260909.tgz` | — |

**Do**
```bash
# download from https://github.com/YosysHQ/oss-cad-suite-build/releases/tag/2026-09-09
tar xzf oss-cad-suite-<platform>-20260909.tgz -C ~/tools
source ~/tools/oss-cad-suite/environment     # puts yosys, iverilog, verilator, gtkwave on PATH
yosys -V ; iverilog -V | head -1 ; verilator --version
```
**Done when** all three print versions. Add the `source` line to your shell profile. Write `oss-cad-suite=2026-09-09` into `versions.lock`.

### 3.3 The sky130 cell library (one file)

We need exactly one liberty file: `sky130_fd_sc_hd__tt_025C_1v80.lib` (high-density library, typical corner, 25 °C, 1.80 V). Smallest verified route is `ciel` (formerly `volare`), the sky130 PDK version manager, which downloads only what is asked for.

**Do**
```bash
pip install ciel
export PDK_ROOT=$HOME/.ciel
ciel ls-remote --pdk-family sky130          # pick the newest listed open_pdks hash
ciel enable --pdk-family sky130 <hash>
ls $PDK_ROOT/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
```
Write the absolute path into `synth/libs.toml` under your machine's name. Every script reads the path from that file; nothing hard-codes it. Record the hash in `versions.lock`.

**Done when** the `ls` above prints the file.

### 3.4 OpenSTA (now on the critical path — Rakshita's Day 0 spike, 45-minute timebox)

The proposal reports area *at matched delay*, so OpenSTA is core, not stretch. It is not in the OSS CAD Suite. Three verified routes, in order of least effort:

1. **WSL2 Ubuntu on the 4070 Ti desktop** (recommended primary). The desktop already needs WSL2 for CUDA training, and OpenSTA's own `Dockerfile.ubuntu24.04` builds cleanly there: `git clone https://github.com/parallaxsw/OpenSTA && cd OpenSTA && docker build --file Dockerfile.ubuntu24.04 --tag opensta .`, then `docker run -i -v $HOME:/data opensta` with the netlist and liberty under `$HOME`. Alternative without Docker: install the listed apt dependencies (cmake ≥ 3.24, tcl 8.6, swig, bison, flex, eigen, zlib, cudd) and `cmake -B build && cmake --build build`.
2. **Source build on the M5 Mac with Homebrew.** The OpenSTA repo ships a `Brewfile`; Apple's Tcl, flex, and bison are incompatible, so export `PATH="$(brew --prefix bison)/bin:$(brew --prefix flex)/bin:$PATH"` before configuring. Slower to get right, but then STA runs where synthesis runs.
3. **Docker on the Mac with `--platform=linux/amd64`.** Works through emulation; slow but our netlists are tiny.

Pick the first that produces a `report_checks` on the smoke-test adder (§3.7 d). All STA runs can happen on one machine; the netlists are small text files pushed through git or copied over.

**Done when** `sta` opens, `read_liberty` of the sky130 file succeeds, and `report_checks` on the adder netlist prints a `Startpoint … Endpoint … slack (MET|VIOLATED)` block. Record the route and commit in `versions.lock`.

### 3.5 ASAP7 (H3 only; set up Thu Sep 17 night, rerun Fri Sep 18)

ASAP7 is BSD-3 licensed and public at `https://github.com/The-OpenROAD-Project/asap7`. The 7.5-track library lives under `asap7sc7p5t_28/LIB/NLDM/`, split into cell groups (AO, INVBUF, OA, SEQ, SIMPLE) and compressed; the flow needs all groups' TT RVT libs read together. An alternative is the merged library shipped under OpenROAD-flow-scripts `flow/platforms/asap7/lib/`. Which one to use is decided on Sep 17 by whoever runs it, and recorded in `versions.lock`. Delay targets must be rescaled for a 7 nm library (their unconstrained critical paths will be several times shorter than sky130).

### 3.6 GPU machines (Rakshita's 4070 Ti desktop and 5080 laptop; both can run 24/7)

Seungmin writes the training and quantization code; Rakshita launches runs. To keep Rakshita's hours small, every run is fire-and-forget: a single `make` target that logs to a file, checkpoints every epoch, appends its CSV row on completion, and never needs babysitting.

**Do** (on each GPU machine, inside WSL2 Ubuntu)
```bash
# CUDA-enabled torch inside the venv
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # pick the cu* that matches the driver
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Data lands once in `~/data/cifar10` (set `FORMATSCOPE_DATA` in the shell profile). Runs are launched with `nohup make train > logs/train_$(date +%F_%H%M).log 2>&1 &`.

**Done when** the CUDA check prints `True` and the GPU name, and a 1-epoch `make train EPOCHS=1` completes on the GPU.

### 3.7 Day 0 smoke tests (all four must pass before anyone writes real code)

**(a) 4-bit adder area** — Rakshita. `rtl/smoke/smoke_add.v` is `module smoke_add(input [3:0] a, b, output [4:0] y); assign y = a + b; endmodule`. Script:
```tcl
read_liberty -lib {LIB}
read_verilog rtl/smoke/smoke_add.v
hierarchy -check -top smoke_add
synth -top smoke_add -flatten
abc -liberty {LIB}
opt_clean
stat -liberty {LIB}
```
Look for `Chip area for module '\smoke_add': <positive float>`. Zero or a vanished module means the design was optimized away.

**(b) cocotb hello on Icarus** — Si Heon. Minimal DUT plus `tb/test_hello.py` using `from cocotb_tools.runner import get_runner; runner = get_runner("icarus")`. Look for `** TEST … PASS **` and pytest `1 passed`. This also confirms the cocotb 2.0 idioms (`dut.sig.value = x`; `int(dut.sig.value)`; `Clock(dut.clk, 10, unit="ns").start()`; `runner.build(sources=[...], hdl_toplevel=..., parameters={...}, build_args=["-g2012"])`).

**(c) CIFAR-10 one-epoch sanity** — Seungmin writes, Rakshita runs on the desktop (or CPU on any laptop, ~5 min). Look for test top-1 above roughly 40% after one epoch, well above the 10% chance line, proving data and loss are wired.

**(d) OpenSTA on the adder netlist** — Rakshita. `read_liberty`, `read_verilog synth/out/smoke_add_netlist.v`, `link_design smoke_add`, `create_clock -name clk -period 10 [get_ports clk]` (add a clock port to the smoke design or use a virtual clock), `report_checks`. Look for a MET/VIOLATED path.

### 3.8 CI (Rakshita, drafted by Claude; 30 minutes on Day 0 or Day 1)

`.github/workflows/ci.yml` on `ubuntu-latest`: `apt-get install iverilog`, `pip install -e . cocotb pytest numpy`, then `pytest tb -q -m "not slow"`. The exhaustive decoder tests and a 300-vector DP32 subset run on every PR in a few minutes; the full 10,000-vector runs are local (`make test`). Synthesis is not run in CI (decision D15). The README shows the green badge.

---

## 4. Lane M — Quantization and training (owner Seungmin; runs launched by Rakshita; Claude drafts)

Order matters: `formats.py` first, because the RTL lane needs its decode tables on Day 1.

### 4.1 `quant/formats.py` — the single source of truth (Day 0–1)

**Do.** One module with, for every format ID, four pure NumPy functions: `encode(values, scale) → codes`, `decode(codes, scale) → values`, `quantize(values) → (codes, scale)` (choose the scale, encode), and `dequantize(codes, scale)`. Plus `decode_table(fmt)` that enumerates every code (256 for INT8 and FP8, 16 for INT4 and E2M1, 256 for E8M0) and dumps `quant/tables/<fmt>.json` as `{code: exact_value_or_"nan"}`. Exact rules:

- **INT8/INT4 symmetric:** `s = max|x| / qmax` (qmax = 127 or 7), `q = clamp(rne(x / s), −qmax, +qmax)`, `x̂ = q · s`. Per-output-channel `s` for weights (one per output channel), per-tensor `s` for activations.
- **FP8-E4M3:** `s` maps `max|x|` just under 448 (decision D11); `x/s` is rounded to the nearest representable E4M3 value (round-to-nearest-even on the 3-bit mantissa, subnormals at E=0, saturate to ±448). Generate the 256-entry table programmatically; NaN maps to code S.1111.111 and is never produced by the quantizer.
- **MX (block 32):** block along the reduction dimension (input channel × kernel position, flattened, for convolutions; input features for linear layers; zero-pad to a multiple of 32). `shared_exp` per the §1.3 rule; elements = `rne(x / 2^shared_exp)` in the element format (INT8 with the implicit 2⁻⁶, or E2M1), clamped to the element max preserving sign. E8M0 code = `shared_exp + 127`, clamped to 0…254.
- **`int4_b32`:** INT4 elements with a block-32 power-of-two scale, same blocking rule.

**Done when** `pytest quant/tests` passes: (i) `decode(encode(v)) == rne_to_format(v)` on 10⁵ random values per format, (ii) every table value matches an independent formula, (iii) FP8 table contains exactly 448 as its max, 2⁻⁹ as its min subnormal, and one NaN code, (iv) the E2M1 table equals ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}.

**Watch out.** The MX block axis. Blocking along the wrong axis (output channels instead of the reduction dimension) still "works" and silently produces wrong accuracy. Unit-test a known 64-element vector and assert which elements share a scale. E8M0 bias off-by-one makes every scale 2× off; test `shared_exp = 0 → code 127`.

### 4.2 `models/resnet8.py`, `data.py`, `train.py` — FP32 baseline (Day 0 write, Day 0 overnight run)

**Do.** ResNet-8 per decision D9: 3×3 stem conv (16 channels) → three stages of one basic block each (16, 32, 64 channels; stride 2 into stages 2 and 3; parameter-free zero-padding shortcuts at the transitions) → global average pool → linear (10). That is eight weighted layers, all of which are quantized. Batch norm after every conv. Recipe: SGD, momentum 0.9, weight decay 5e-4, batch 128, cosine-annealed learning rate from 0.1, 60 epochs (≈ 15 min on the 4070 Ti), random crop (pad 4) + horizontal flip + per-channel normalization. Seed 0. Save `models/checkpoints/fp32.pt` every epoch (last and best). `make train EPOCHS=1` must work for the smoke test.

**Done when** the log shows a test top-1 in or above the 85–88% band and `fp32.pt` exists. Append the row `fp32,none,<top1>,…` to `results/accuracy.csv`.

**Watch out.** If the number lands at 80% or below, check augmentation and the learning-rate schedule before touching architecture. Do not chase 92%; the ceiling only has to be stable and reported.

### 4.3 `quant/fakequant.py` — wrappers, BN folding, STE (Day 1)

**Do.** `QConv2d` and `QLinear` replace `nn.Conv2d` / `nn.Linear` by module replacement (not hooks). Each wrapper: (i) folds the following batch norm into the weight and bias first (decision D9; the hardware consumes folded weights), (ii) quant-dequantizes the folded weight once per forward with per-output-channel scales (per-block for MX), (iii) quant-dequantizes the input activation at runtime with the calibrated per-tensor scale (per-block for MX), (iv) runs the ordinary float conv/linear on the dequantized tensors. A `format` argument selects the `formats.py` functions; a `first_last` flag exists but defaults to quantizing everything, for honesty. For QAT, the quantizer's backward is the straight-through estimator (identity inside the clamp range, zero outside).

**Done when** a model converted with `format="int8"` and evaluated without calibration runs end to end, and `format=None` reproduces the FP32 accuracy exactly (proves BN folding is correct).

### 4.4 `quant/calibrate.py` — activation ranges (Day 1)

**Do.** Run 512 fixed training images (seeded subset, same for every format) through the folded FP32 model with observers on every wrapper input; per-tensor scale from the 99.99th percentile of |x| (decision D10; `max` is the alternative and is reported as a second row if time allows). MX formats need no activation calibration (block scales are computed on the fly), but the calibration images are still used for the FP8/INT per-tensor scales.

**Done when** `results/accuracy.csv` rows carry `calib_images=512` and `calib_stat=p99.99`.

### 4.5 `quant/eval.py` — PTQ for all six configurations (Day 1 for INT8, Day 2 for the rest)

**Do.** For each of `int8, int4, fp8e4m3, mxint8, mxfp4, int4_b32`: load `fp32.pt`, fold, wrap, calibrate, evaluate on the 10,000-image test set, append one row with `stage=ptq`. One `make quant` runs all six in a few minutes on a GPU (or ~30 min on CPU).

**Done when** six `ptq` rows exist and pass the sanity bounds below.

Expected PTQ top-1 relative to FP32 (sanity bounds, not targets):

| Configuration | Expected | Basis |
|---------------|----------|-------|
| INT8 | ≈ lossless | ResNet-20/CIFAR-10 literature: uniform INT8 has virtually no loss (arXiv 2512.14090) |
| INT4 (naive per-tensor or per-channel PTQ) | large drop, possibly tens of points | ResNet-20 naive INT4 73.45% vs 91.36% FP32 (arXiv 2303.02347, Table 1) |
| FP8-E4M3 | ≈ lossless | arXiv 2209.05433 |
| MXINT8 | ≈ lossless | arXiv 2310.10537 |
| MXFP4 | moderate loss | E2M1 coarseness |
| INT4-b32 | between INT4 and MXFP4 | the ablation's whole point |

**Watch out.** Accuracy collapsing toward 10% (chance) means a scale or rounding bug, never a format cost. If naive INT4 is *not* visibly below INT8, the quantizer is broken; fix it before proceeding rather than explaining it away.

### 4.6 `quant/qat.py` — five-epoch quantization-aware fine-tune (Day 3–6)

**Do.** Start from `fp32.pt`, wrap with STE quantizers, keep BN folded and frozen, SGD momentum 0.9, learning rate 0.01 cosine-annealed to 0 over 5 epochs, batch 128, same augmentation, seed 0 (decision D12). One run per configuration, all six (≈ 2 min each on the 4070 Ti, so one `make qat` covers them). Append rows with `stage=qat`. Run order, so the most informative rows land first: INT4, MXFP4, INT8, FP8, MXINT8, INT4-b32.

**Done when** `qat` rows exist for all six configurations, and the MX rows show whether MXFP4 closes most of its gap to INT8 (H2).

### 4.7 `quant/fidelity.py` — PyTorch path vs DP32 path on real data (Day 4–6)

**Do.** Pick one layer (recommended: the first conv of stage 2, 16→32 channels, reduction length 16×9 = 144 = 4.5 blocks, padded to 5). Take one batch of real activations and the folded weights, quantize both to the format's codes, then compute every output dot product two ways: (a) the PyTorch path — dequantize and accumulate in FP32; (b) the DP32 path — the NumPy reference in `tb/refs.py` that mirrors the RTL exactly (block of 32, one fused rounding per block into the FP32-format accumulator, sequential over blocks). Report per format: max and mean absolute and relative difference over all outputs, and the change in the layer's output when the whole layer runs through path (b). For INT4 and INT8 assert exact equality (both are exact integer sums; ResNet-8's largest reduction, 64×9 terms at 127², fits in FP32's 24-bit mantissa).

**Done when** `results/fidelity.csv` has one row per format and INT rows show difference 0.

### 4.8 Ablation and write-up (Day 7)

**Do.** A short `docs/results.md` paragraph per hypothesis with the numbers, and the INT4 vs INT4-b32 vs MXFP4 comparison in one sentence. Seungmin also produces the results-table PNG for the video (`formatscope table --png`).

---

## 5. Lane R — RTL (owner Si Heon; Claude drafts; Si Heon reviews, simulates, commits)

Implementation order: INT8 → INT4 → fused stage + FP8 → MXINT8 → MXFP4. The fused stage is the single hardest piece of the project; its NumPy reference (§6.2) is written *before* its RTL, and the RTL is checked against that reference, not against intuition.

### 5.1 Common interface and rules

Every unit exposes the same ports so the synthesis and test scripts are identical across formats:

```verilog
module dp32_<fmt> #(parameter ALIGN_W = 24) (   // ALIGN_W used by fp8e4m3 / mxint8 / mxfp4 only
  input  wire         clk,
  input  wire         rst_n,
  input  wire         en,        // accumulate this cycle
  input  wire         clear,     // zero the accumulator
  input  wire [32*W-1:0] a_flat, // 32 elements, W bits each (W = 4 or 8), element 0 in the LSBs
  input  wire [32*W-1:0] b_flat,
  input  wire [7:0]   scale_a,   // E8M0 shared scale; MX units only, tie to 8'd127 elsewhere
  input  wire [7:0]   scale_b,
  output reg  [31:0]  acc_out,   // INT32 (int4/int8) or FP32-format (fp8e4m3/mx)
  output reg          flag_nan   // sticky: set when a NaN code or 0xFF scale was seen
);
```

Pipeline choice, identical across all five units (decision D14): purely combinational multiply → align → tree → normalize/round feeding the single accumulator register. `clear` zeroes the accumulator; `en` gates accumulation. Registered outputs only. Each unit instantiates its stages as *separate named modules* (`mul_stage`, `align_stage`, `tree_stage`, `normacc_stage`) so the hierarchical synthesis run can report area per stage.

Coding rules: Verilog-2005; `localparam` for widths; `generate` for the tree; `$signed()` around every signed operand; every `always @*` assigns every output; `hierarchy -check` must pass with no unconnected-output warnings.

### 5.2 Decoders (`rtl/common/decode_*.v`)

- `decode_fp8e4m3`: in `[7:0]`, out `sign`, `exp[4:0]` (unbiased exponent + 6, so subnormals and normals share one scale: normal `E−7+6 = E−1`, subnormal `0`), `sig[3:0]` (hidden bit + 3 mantissa bits; subnormal has hidden 0), `is_nan`. Purely combinational.
- `decode_e2m1`: in `[3:0]`, out `sign`, `mag[3:0]` = element value × 2 ∈ {0,1,2,3,4,6,8,12}. No NaN.
- `decode_e8m0`: in `[7:0]`, out `exp[7:0]` (raw), `is_nan` (= 0xFF).
- INT decoders are identity; they exist as modules so `test_decode.py` treats every format the same way.

**Done when** `test_decode.py` (§6.3) passes exhaustively for every decoder.

### 5.3 INT8 and INT4 DP32 (Day 1–2)

**Do.** 32 signed products (`$signed(a[i]) * $signed(b[i])`): 16-bit for INT8, 8-bit for INT4. Balanced adder tree (5 levels), sign-extending one bit per level: 21-bit sum for INT8 (max 32 × 127² = 516,128 < 2²⁰), 13-bit for INT4. Sign-extend into the INT32 accumulator; overflow is two's-complement wraparound (decision D13; it never triggers for ResNet-8, but it is stated in the README so a judge does not read it as an omission). `dp32_int4` is `dp32_int8` with `W = 4`, so write the tree once with parameters.

**Done when** `test_dp32_int.py` passes 10,000 vectors + corners for both widths, and Rakshita's flat synthesis prints a positive area for both.

### 5.4 The fused stage (`rtl/common/fused_stage.v`) — write `docs/fused-stage.md` first (Day 2–3)

This module is shared by FP8 (33 terms) and both MX units (2 terms), parameterized by `N_TERMS`, term significand width `SIG_W`, and `ALIGN_W`.

Input: `N_TERMS` terms, each `(sign, exp, sig)` with a common exponent scale, plus the current FP32-format accumulator unpacked into the same form (sign, exponent, 24-bit significand with hidden bit).

**Do**, in order, and put every width in `docs/fused-stage.md` with a one-line justification:

1. **Max exponent** across all terms (a tree of comparators; 33 inputs for FP8).
2. **Align:** shift each term's significand right by `(max_exp − exp_i)` into a window of `ALIGN_W` bits plus 3 guard/round/sticky bits; every bit shifted past the window ORs into that term's sticky bit. Shifts larger than the window width produce all-zero plus sticky.
3. **Negate** aligned terms with `sign = 1` into two's complement (window + carry bits).
4. **Sum** in one balanced integer tree with `ceil(log2(N_TERMS + 1))` extra carry bits (6 for 33 terms, 1 for 2 terms).
5. **Sign-magnitude:** take the absolute value of the sum and its sign.
6. **Normalize:** leading-zero count (`lzc.v`) on the magnitude; shift left by the count; exponent = `max_exp + carry_bits − lzc`.
7. **Round once, RNE:** using the guard bit, the round bit, and the OR of all sticky bits (term stickies OR bits shifted out during normalization). Handle the mantissa-overflow-on-round case (renormalize by one).
8. **Pack** into FP32 format: sign, biased 8-bit exponent, 23-bit fraction. Range analysis for the README: FP8 products lie in [2⁻¹⁸, 448²] and 32 of them summed over any realistic number of accumulations stay far inside FP32's normal range, so the register never needs subnormals or infinities; assert this in the tests instead of building the hardware for it.

Sweep: `ALIGN_W ∈ {24, 32}` (decision D6; 24 is FP32's own significand precision, 32 shows what extra precision costs). The default in `results/` is 24; the sweep rows carry `align_w=32`.

**Done when** `docs/fused-stage.md` exists with every width, and `tb/refs.py` implements the identical steps in NumPy with explicit integer shifts (so any RTL mismatch is an RTL bug, or a documented reference bug, never ambiguity).

### 5.5 FP8-E4M3 DP32 (Day 3–4)

**Do.** Per lane: decode both codes; significand product `sig_a × sig_b` (4b × 4b → 8-bit, max 225); exponent sum `exp_a + exp_b` (5-bit, in the +6 shifted scale from §5.2, so the term value is `product × 2^(exp_sum − 12 − 6)` — write the exact constant in `fused-stage.md`); sign = XOR. NaN policy (decision D3): a NaN input makes that lane's term zero and sets `flag_nan`; inference data never contains NaN, so this never fires but is documented and tested. Feed the 32 terms plus the accumulator into `fused_stage` with `N_TERMS = 33`.

**Done when** `test_dp32_fp8.py` passes 10,000 vectors + corners (all-max, all-min, alternating signs, subnormal inputs, NaN inputs, exact cancellation, accumulator much larger than the products, products much larger than the accumulator) for `ALIGN_W = 24` and `32`.

### 5.6 MXINT8 DP32 (Day 5)

**Do.** Instantiate the INT8 multiply and tree stages unchanged to get the exact 21-bit block sum. Block exponent: `scale_a + scale_b − 254 − 12` (two E8M0 biases and the two implicit 2⁻⁶ element scales), computed in enough bits to be signed. Feed `(sign, exponent, |block_sum|)` and the accumulator into `fused_stage` with `N_TERMS = 2` and `SIG_W = 21`. E8M0 NaN policy (decision D4): if either scale is 0xFF, the block term is zero and `flag_nan` is set. Clamp the exponent difference so the align shifter never wraps.

**Done when** `test_dp32_mx.py::test_mxint8` passes 10,000 vectors + corners (extreme scales 0 and 254, both scales NaN, all-zero block, block sum exactly at ±2²⁰).

### 5.7 MXFP4 DP32 (Day 6)

**Do.** Decode each E2M1 code to `mag ∈ {0,1,2,3,4,6,8,12}` (value × 2) and a sign; the product of two mags is at most 144, so a signed product fits in 9 bits (a 16-entry-by-16-entry table or a 4b×4b multiplier; the table is smaller and is what the proposal describes). The 32 products sum into a 14-bit tree (|sum| ≤ 4608 < 2¹³). Block exponent: `scale_a + scale_b − 254 − 2` (the ×2 scaling of each mag). Then `fused_stage` with `N_TERMS = 2`, `SIG_W = 14`.

**Done when** `test_dp32_mx.py::test_mxfp4` passes 10,000 vectors + corners.

### 5.8 Freeze (Day 7)

Tag `rtl-v1` on `main` when all five units are green; Rakshita re-runs every synthesis from that tag so netlists and RTL match.

---

## 6. Lane V — Verification harness (owner Si Heon with Claude; runs in CI)

### 6.1 Runner pattern (cocotb 2.0, Icarus, pytest)

```python
# tb/conftest.py (helper used by every test file)
import os
from pathlib import Path
from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parent.parent

def run(sources, toplevel, test_module, parameters=None):
    runner = get_runner(os.getenv("SIM", "icarus"))
    runner.build(sources=[ROOT / s for s in sources], hdl_toplevel=toplevel,
                 parameters=parameters or {}, build_args=["-g2012"], always=True)
    runner.test(hdl_toplevel=toplevel, test_module=test_module)
```

Test body idioms (2.0-correct; the 1.x forms raise errors): `dut.a_flat.value = packed_a`; `int(dut.acc_out.value)` or `dut.acc_out.value.to_signed()`; `Clock(dut.clk, 10, unit="ns").start()` with no `start_soon` wrapper; reset for two edges, then drive and wait one rising edge per accumulation.

### 6.2 `tb/refs.py` — NumPy references (written before the matching RTL)

One function per unit: `dp32_int(a_codes, b_codes, acc, w)`, `dp32_fp8(a_codes, b_codes, acc_bits, align_w)`, `dp32_mxint8(...)`, `dp32_mxfp4(...)`. They use `np.int64` intermediates and explicit shifts, mirroring §5.4 step by step, and return the exact 32-bit register value. They import decoders from `quant/formats.py` so the two lanes cannot drift. Also provide `pack(codes, w)` to flatten 32 elements little-endian into one integer for `a_flat`.

### 6.3 `tb/test_decode.py` — exhaustive decoders

For each decoder, loop over every code (256 or 16), drive it, read the RTL's `(sign, exp, sig, is_nan)` and compare with the `quant/tables/<fmt>.json` value. Runtime is under a second each. The video shows the line `256/256 FP8 codes match`.

### 6.4 `tb/test_dp32_*.py` — random plus directed vectors

10,000 seeded random vectors per unit (seed 0; the CI subset runs 300 with `-m "not slow"`), plus the directed corners listed in §5. Each vector drives `a_flat`, `b_flat`, the scales, and a random prior accumulator (via `clear` then a preload sequence, or a `preload` port used only in simulation — decide in D14 notes), waits one edge, and asserts exact equality with the reference. On failure, print the vector in hex and the expected/actual values; save a VCD only when `FORMATSCOPE_VCD=1` to keep runs fast.

### 6.5 Debugging playbook

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| negatives wrong, positives right | `$signed` missing on one operand | wrap both; add the alternating-sign corner |
| off by exactly one LSB on some vectors | RNE tie handling or sticky lost during normalization | compare guard/round/sticky between ref and RTL on that vector |
| result 2× or ½× | E8M0 bias or the ×2 / 2⁻⁶ constants | recheck the exponent constant in `fused-stage.md` |
| Yosys "inferring latch" | an `always @*` path leaves an output unassigned | assign defaults at the top of the block |
| `stat` shows ~0 cells | outputs optimized away | outputs must be registered and used; run `hierarchy -check` |
| cocotb `ValueError` on assignment | value out of range for the vector width | check `pack()` and widths |

---

## 7. Lane S — Synthesis, timing, and the stage breakdown (owner Rakshita; Claude drafts everything so Rakshita's time goes to running and reading results)

### 7.1 Flat synthesis for area (`synth/synth_flat.ys.template`)

```tcl
read_liberty -lib {LIB}
read_verilog -sv rtl/common/adder_tree.v rtl/common/lzc.v rtl/common/fused_stage.v rtl/common/decode_fp8e4m3.v rtl/common/decode_e2m1.v rtl/common/decode_e8m0.v rtl/{FMT}/{UNIT}_{FMT}.v
chparam -set ALIGN_W {ALIGN_W} {UNIT}_{FMT}
hierarchy -check -top {UNIT}_{FMT}
synth -top {UNIT}_{FMT} -flatten
dfflibmap -liberty {LIB}
abc -liberty {LIB} {DTARGET}
opt_clean
tee -o synth/out/{LIB_ID}_{FMT}_{TARGET}_a{ALIGN_W}_r{RUN}_stat.txt stat -liberty {LIB}
write_verilog -noattr synth/out/{LIB_ID}_{FMT}_{TARGET}_a{ALIGN_W}_r{RUN}_netlist.v
```

`{DTARGET}` is empty for the unconstrained run and `-D <picoseconds>` for the two shared delay targets. `dfflibmap` maps the flip-flops, `abc -liberty` maps the combinational logic, `write_verilog -noattr` emits the clean netlist OpenSTA reads. Fairness rule: identical script, library, accumulator width, and rounding mode across all five formats; only `{FMT}` and `{DTARGET}` change.

`synth/run_synth.py` renders the template for every (library, format, target, align_w, run), runs Yosys, and parses `Chip area for module '\<top>': <float>` and `Number of cells:` into `results/area.csv`. It refuses to write a row whose area is zero.

**Done when** `make synth FMT=int8` produces an `area.csv` row with a positive area and a netlist file (Day 1), and `make synth` runs all five (Day 6).

### 7.2 Choosing the two shared delay targets (Thu Sep 17, needs INT4/INT8/FP8 unconstrained numbers)

**Do.** Run every available unit unconstrained, time each with OpenSTA (§7.3), and record the critical-path delay. Then set (decision D7, fixed by Thu Sep 17 14:00 in the revised calendar): `T1` = the slowest unit's unconstrained delay rounded up to the next 100 ps (every unit can meet it), and `T2` = roughly 0.7 × T1 (a target that forces ABC to spend area for speed). Both targets are shared by all units, written once into `synth/targets.toml`, and never changed after that without re-running everything. Units that miss `T2` are still reported, with their achieved delay, and the README says so.

### 7.3 OpenSTA timing (`sta/sta.tcl.template`, `sta/run_sta.py`)

```tcl
read_liberty {LIB}
read_verilog synth/out/{NETLIST}
link_design {UNIT}_{FMT}
create_clock -name clk -period {PERIOD_NS} [get_ports clk]
set_input_delay  -clock clk 0 [all_inputs]
set_output_delay -clock clk 0 [all_outputs]
report_checks -path_delay max -fields {slew cap input nets fanout} -digits 3
report_wns
report_tns
```

`run_sta.py` parses the `data arrival time` of the worst path (that is the combinational delay from the input ports to the accumulator register, independent of the chosen period) plus WNS, and writes `results/timing.csv`. The README states plainly: these are pre-layout numbers with no wire parasitics — indicative, not sign-off.

**Done when** `make sta` produces a `timing.csv` row for every netlist in `synth/out/`.

### 7.4 Spread across runs (Day 6–7)

The proposal promises the spread across "seeds". Yosys's ABC pass has no seed switch and is expected to be deterministic for a fixed input, so define the spread as **perturbation runs** (decision D8): for each (unit, target), five runs at `-D` = target × {0.98, 0.99, 1.00, 1.01, 1.02}, reporting min / median / max area. First, on Day 2, confirm determinism by running the INT8 unconstrained synthesis three times and diffing the areas; if they differ, that variance is reported as-is and the perturbation runs come on top.

### 7.5 Stage breakdown (`synth/synth_hier.ys.template`, Day 4–6)

Same script without `-flatten`, and with `stat -liberty {LIB}` run after `abc` so it reports each module's area separately (`mul_stage`, `align_stage`, `tree_stage`, `normacc_stage`, and the decoders). `run_synth.py --hier` writes `results/breakdown.csv` with one row per (format, stage). Hierarchical synthesis loses cross-boundary optimization, so the breakdown total will not equal the flat area; report both and say why.

### 7.6 ASAP7 rerun (H3, scheduled Fri Sep 18)

In scope, scheduled, and the first thing the proposal names to cut only if the schedule has slipped. Add the ASAP7 liberty path(s) to `synth/libs.toml`, run the same templates with `LIB_ID=asap7`, rescale `-D` targets after an unconstrained ASAP7 pass, and time with OpenSTA. The deliverable is the ASAP7 netlists and reports, one extra frontier figure, and one sentence: does the ranking hold?

### 7.7 Sanity checks that gate every result row

- Expected area ordering, unconstrained: INT4 < INT8 ≈ MXINT8 < FP8-E4M3 (the direction H1 predicts). A violation is a fairness bug until proven otherwise; the first suspect is the accumulator or window width differing between formats.
- `stat` showing zero cells or a tiny area means outputs were optimized away.
- `read_liberty` errors mean `libs.toml` has the wrong path for this machine.
- ABC `-D` is never used for the unconstrained run; using it there makes areas incomparable.

---

## 8. Lane T — The `formatscope` tool (owner Rakshita; Claude drafts; Rakshita reviews and runs)

### 8.1 Commands

```
formatscope run       [--from-results]   # runs quant eval + synth + sta, or just reloads the CSVs
formatscope plot      [--lib sky130hd] [--target t1] [--stage ptq|qat] [--logx]
formatscope recommend --min-acc 0.87 [--target t1]        # smallest-area unit meeting the target
formatscope recommend --area-budget 30000 [--target t1]   # best accuracy under the cap (µm²)
formatscope table     [--png]                              # the results table, markdown or PNG
formatscope demo                                           # one INT8 synthesis on camera + a recommend
```

### 8.2 Data model (`tool/formatscope/data.py`)

Join `accuracy.csv` (format, stage, top1) with `area.csv` (format, lib, target, align_w, run, area_um2) and `timing.csv` (format, lib, target, align_w, run, delay_ps) on `format`. Default view: `lib=sky130hd`, `align_w=24`, median over runs, `stage=ptq` with `qat` as hollow markers. The `int4_b32` configuration has no hardware row and appears only in the table.

### 8.3 Frontier and recommendation (`pareto.py`, `recommend.py`)

Pareto front per (lib, target, stage): sort by area ascending, keep a point iff no other point has both ≤ area and ≥ accuracy. Knee = the point maximizing accuracy per µm² (documented; the second-difference alternative is a flag). `--min-acc X` returns the min-area format whose accuracy ≥ X (default margin 1.0 point below FP32 when `--min-acc` is omitted, decision D17); `--area-budget Y` returns the max-accuracy format with area ≤ Y. Unit tests on a tiny synthetic CSV cover both.

### 8.4 Plots (`plots.py`, matplotlib only)

- Frontier: x = DP32 area (µm², linear; `--logx` optional), y = top-1; one marker per format, consistent colors across all figures; the front drawn as a step line; delay target in the title; PTQ filled, QAT hollow. One figure per delay target, plus a 1×3 panel.
- Breakdown: stacked bars of stage area per format.
- Window sweep: FP8 and MX area at `align_w` 24 vs 32.
- Export PNG and SVG at 300 dpi, fonts ≥ 14 pt, no gridline clutter.

**Done when** `make plot` regenerates every file in `results/figures/` from the CSVs alone.

### 8.5 `formatscope demo`

Runs the INT8 flat synthesis (must finish in under 60 s on camera; the smoke test measured it), prints the area line, then prints the `recommend --min-acc` answer. Rehearsed on Sep 18.

---

## 9. Integration, README, video, submission

### 9.1 README order (judge-readable; matters for the 20% implementation-quality and 10% video lines)

1. One-paragraph pitch.
2. The frontier figure at the middle delay target.
3. "Reproduce in six commands": `make env`, `make train`, `make quant`, `make test`, `make synth sta`, `make plot`.
4. Results table (accuracy PTQ/QAT × area × delay per format, three targets).
5. H1 / H2 / H3 with numbers.
6. Limitations: pre-layout, no interconnect, sky130 absolute values do not transfer (ratios do), per-tensor activation scaling, ABC variance and how it is reported, power out of scope.
7. Decisions (link to `docs/decisions.md`) and references.

### 9.2 Reproduction check (Sep 18)

Si Heon on a Mac and Seungmin on Windows each do a fresh clone and run the six commands. Anything that needs a hand edit becomes a bug fixed that day.

### 9.3 Video shot list (3–5 minutes, 1080p screen capture, terminal font ≥ 18 pt, ~140 words per minute, captions on)

| Time | Shot | Content |
|------|------|---------|
| 0:00–0:25 | Hook | "Which number format buys the most accuracy per micron of silicon?" State H1 and H2. |
| 0:25–1:15 | Tool live | `formatscope demo`: one INT8 synthesis finishing on camera, then `recommend --min-acc` printing the winner. |
| 1:15–2:05 | The chart | Walk the frontier at the middle delay target; point at the INT8-vs-FP8 gap and where MXFP4 lands; flip to the breakdown bars to say where FP8's cost sits. |
| 2:05–2:45 | Proof | Green `pytest -q`; the "256/256 FP8 codes match" and "10000/10000 vectors exact" lines; the fidelity table. |
| 2:45–3:35 | Findings | Measured INT8-vs-FP8 DP32 area ratio at matched delay next to Qualcomm's 183% single-MAC figure; MXINT8's increment over INT8; MXFP4 after fine-tuning. |
| 3:35–4:20 | Limits and next | Pre-layout, ratios not absolutes, per-tensor activation scaling, place-and-route and energy as next steps; ASAP7 result if it exists. |

Narration split: decision D18 (recommended: each owner narrates their own lane's shot). Record on Sep 18, edit the same day, upload Sep 19 morning.

### 9.4 Submission checklist (Sep 19, before noon)

- [ ] `main` tagged `v1.0`; every CSV and figure regenerated from that tag
- [ ] Repository flipped to **public**; README renders; CI badge green
- [ ] Video uploaded (unlisted link or file per the organizers' form) and the link tested in a private window
- [ ] Submission form completed with the video and repo links
- [ ] `make freeze` run from the tag: final netlists and yosys/OpenSTA reports for every unit, delay target, window setting, and library committed under `results/netlists/` and `results/reports/`

### 9.5 Showcase Q&A (Sep 22, `video/qa.md`)

Prepared answers for: why a fused stage with a single rounding; why an FP32-format accumulator rather than fixed point; how FP8 subnormals are handled; why block size 32; why area at matched delay rather than area alone; why per-tensor activation scaling; what "spread across runs" means; how MXFP4 loses accuracy; why not energy; how our INT8-vs-FP8 ratio compares with Qualcomm's estimate.

---

## 10. Day-by-day calendar (revised Tue Sep 15 → Sat Sep 19)

SH = Si Heon, SN = Seungmin, RG = Rakshita, CL = Claude (drafts; never commits). Gates are checked at the next morning's sync. The original Sep 10 calendar is in git history (`500636f`); "Day N" references in the lane sections point at that calendar, and this table supersedes them.

### 10.1 Where we are (Tue Sep 15, evening)

The team kept the **full proposal scope** at the Sep 15 re-plan. Quantization is two days ahead of hardware, and the whole hardware side (RTL, verification, synthesis, timing) is four days behind.

| Item | Planned by | Status |
|------|-----------|--------|
| FP32 baseline | Thu Sep 10 | **Done**: 86.34% top-1 (PR #1, merged Sep 12) |
| `formats.py`, decode tables, `fakequant.py`, `calibrate.py`, tests | Fri Sep 11 | **Done** (PR #1) |
| PTQ rows for all six configurations | Sat Sep 12 | **Done**: INT8 86.39, FP8 85.50, MXINT8 86.39, INT4 61.65, MXFP4 60.71, INT4-b32 78.46 |
| Smoke tests and `versions.lock` (G0) | Thu Sep 10 | Not started. RG's half is scripted in `scripts/setup_lane_s_wsl.sh` |
| Makefile, `pyproject.toml`, CI | Thu Sep 10 | Drafted on `rg/synth-sta-tool` |
| `run_synth.py`, `run_sta.py`, templates, parsers | Fri Sep 11 | Drafted on `rg/synth-sta-tool`, unit-tested on fixtures, **never run against real Yosys or OpenSTA** |
| `formatscope` tool (`plot`, `recommend`, `table`, `demo`, `run`) | Sat Sep 12 → Mon Sep 14 | Drafted on `rg/synth-sta-tool` with tests |
| Any RTL, `tb/`, `refs.py` | Fri Sep 11 → Wed Sep 16 | **Not started**. This is the critical path |
| `qat.py`, `fidelity.py` | Sun Sep 13 | Not started |
| T1/T2 delay targets | Sun Sep 13 | Blocked on RTL |

**Check before QAT:** MXFP4 PTQ (60.71) came in *below* plain INT4 (61.65), and INT4-b32 (78.46) sits above both, when §4.5 expected INT4-b32 between INT4 and MXFP4. That may be genuine: the `floor(log2 max) − 2` shared-exponent rule clips block maxima in (6, 8)·2^e down to 6. It may also be a bug. SN confirms which before QAT numbers go into H2.

### 10.2 Revised calendar

| Day | SH (RTL + cocotb) | SN (quant + training) | RG (synth, STA, tool, GPUs) | End-of-day deliverable | Gate |
|-----|-------------------|------------------------|-----------------------------|------------------------|------|
| **Tue Sep 15** (evening) | Suite + cocotb hello (§3.7b); CL drafts `adder_tree.v`, `dp32_int8.v`/`int4`, `refs.py` INT, `test_dp32_int.py` tonight | `qat.py` (§4.6) with a one-line launch command in the PR; explain MXFP4 < INT4 (§10.1) | Review and merge `rg/synth-sta-tool`; run `scripts/setup_lane_s_wsl.sh` on the 4070 Ti (and the 5080, for parallel synth later); **G0 smoke passes**; launch QAT for all six overnight once `qat.py` lands (≈ 12 min on the 4070 Ti) | Toolchain proven on real Yosys + OpenSTA; CI green; QAT running | **G0** |
| **Wed Sep 16** | INT8 + INT4 10k green by noon; afternoon: `docs/fused-stage.md`, `refs.py` FP8, FP8/E2M1/E8M0 decoders + exhaustive tests; start `fused_stage.v` | Commit QAT rows (all six); `fidelity.py` with INT4/INT8 exact rows | INT8 + INT4 synth unconstrained + STA; determinism check (3× INT8); fix any parser mismatch against real tool output; first real `formatscope plot` / `recommend` | INT4/INT8 verified and measured; PTQ + QAT complete | **G1 + G2** |
| **Thu Sep 17** | `fused_stage.v` + `dp32_fp8e4m3.v` 10k green at `ALIGN_W` 24 and 32 **by noon**; afternoon `dp32_mxint8.v` (INT8 tree + two-term fused stage); evening `dp32_mxfp4.v` | Fidelity rows for FP8, then MXINT8/MXFP4 as each unit lands; `docs/results.md` H1/H2 draft | FP8 unconstrained STA → **fix T1/T2 by 14:00** (§7.2, D7) into `synth/targets.toml`; INT4/INT8/FP8 at T1/T2; hier breakdown; FP8 window sweep; MX units synthesized as they go green; ASAP7 libs set up; perturbation runs overnight, split across both GPU machines | FP8 at matched delay (floor) by afternoon; four or five formats measured by night | **G3 (floor)** by 18:00; **G4** by night |
| **Fri Sep 18** | MXFP4 green if not already; full `make test`; **tag `rtl-v1` by 11:00**; README RTL section; fresh-clone reproduction on Mac; record test-suite shot | Fresh-clone reproduction on Windows; results table PNG; H1/H2 final numbers; record chart/findings shots | Re-run every synthesis from `rtl-v1` (both machines); MXINT8/MXFP4 at both windows; breakdown for all; ASAP7 rerun + figure (H3); `make plot`; `make freeze`; README results section; record `formatscope demo`; edit video v1 in the evening | Complete results set, figures, README, video v1 | **G5 + G6** |
| **Sat Sep 19** | Final review of the repo as a stranger | Final review of the README numbers vs CSVs | Tag `v1.0`; flip public; upload video; submit **before noon** | **Submitted** | **Submit by noon** |

**Dated cut triggers (full scope stays the plan; these fire only if a checkpoint is missed).**

- **FP8 not bit-exact by Thu 18:00:** drop ASAP7. RG synthesizes the current FP8 RTL with a "not yet bit-exact" label so the floor figure still exists. MX units start Friday morning.
- **MX units not green by Fri 11:00:** drop the third delay target, tag `rtl-v1` on whatever is green, and report MX accuracy only.
- **QAT on the MX formats** is the proposal's third cut, but it costs minutes of GPU time and runs Tuesday night with the rest, so in practice it never gets cut.
- **Never below the floor:** INT4, INT8, and FP8 verified end to end at two matched delay targets, plus PTQ accuracy for all five formats.

**Why this can still close:** synthesis and timing are scripted end to end, so once RTL lands each unit costs RG about one command and minutes of machine time. MXINT8 and MXFP4 reuse the INT tree and the fused stage, so they come quickly *after* FP8. The single risk that decides the outcome is the fused stage on Thursday morning (risk 2). CL drafts all remaining RTL and references tonight and Wednesday so SH's time goes to simulating and debugging, not typing.

---

## 11. Risk register

| # | Risk | Detect | Mitigation |
|---|------|--------|------------|
| 1 | OpenSTA does not build on the Mac | Day 0 spike fails | Run all STA in WSL2 on the desktop; netlists travel through git |
| 2 | Fused stage never becomes bit-exact | FP8 vectors still failing Sep 14 | Reference first, widths documented, directed corners before random; debug on a `N_TERMS=2` instance |
| 3 | cocotb 1.x idioms | `<=` assignment, `units=`, `cocotb.runner` errors | Use the §6.1 idioms; the hello test is the template |
| 4 | Yosys optimizes logic away | `stat` ~0 cells | Registered, used outputs; `hierarchy -check` |
| 5 | Latch inference | "inferring latch" warning | Default assignments at the top of every `always @*` |
| 6 | `$signed` misuse | negatives wrong | Wrap both operands; alternating-sign corner |
| 7 | MX blocked along the wrong axis | accuracy fine but wrong | Unit test on a known vector |
| 8 | E8M0 bias off by one | all MX values 2× off | Test `shared_exp=0 → code 127` |
| 9 | BN folding wrong | `format=None` ≠ FP32 accuracy | §4.3 done-when check |
| 10 | ABC variance makes the ranking wobble | perturbation min/max overlap between formats | Report the overlap honestly; the claim is the ranking where it is clear |
| 11 | `-D` used inconsistently | unconstrained row has a target | `run_synth.py` refuses `-D` on `target=unc` |
| 12 | WSL repo under `/mnt/c` | sim/synth crawl | Keep the repo on the Linux filesystem |
| 13 | GPU runs need babysitting and Rakshita's hours are short | runs waiting on a human | Every run is one `nohup make …` with logs and checkpoints; SN prepares the exact command in the PR |
| 14 | Repo still private on submission | judges see 404 | Sep 19 checklist item; flip the day before if nothing secret remains |
| 15 | Video runs long or the live synth stalls | rehearsal on Sep 18 | Keep the live unit to INT8 (<60 s); pre-record a fallback clip |

---

## 12. Decision log (confirm at the first sync; record in `docs/decisions.md` with date and name)

| ID | Decision | Recommended default |
|----|----------|---------------------|
| D1 | INT code ranges | symmetric: ±7 and ±127, most-negative code unused |
| D2 | FP8 overflow | saturate to ±448 (OCP-permitted; matches inference practice) |
| D3 | FP8 NaN in hardware | lane term = 0, sticky `flag_nan` |
| D4 | E8M0 0xFF in hardware | block term = 0, sticky `flag_nan` |
| D5 | MXINT8 code −128 | quantizer never emits it; RTL treats it as an ordinary int8 |
| D6 | Alignment window widths | 24 (default) and 32 |
| D7 | Delay-target rule | T1 = 1.03 × (slowest unconstrained unit's arrival + setup, i.e. period − WNS from its unconstrained STA row), rounded up to 100 ps; the 3% covers ABC-vs-OpenSTA model error. No T2: the INT units are ~3× faster than the fused units at their fastest mapping, so no tighter shared period constrains both, and ABC cannot map a unit faster than its delay-optimal result. Each unit is reported unconstrained and at T1. Fixed Thu Sep 17 |
| D8 | "Spread across seeds" | five perturbation runs at ±1%, ±2% of `-D`, plus a determinism check |
| D9 | ResNet-8 definition and BN | 6n+2 with n=1, widths 16/32/64, zero-pad shortcuts; BN folded before quantization |
| D10 | Activation calibration statistic | 99.99th percentile of |x| over 512 images; `max` as a second row if time |
| D11 | FP8 weight/activation scale | max maps just under 448 (same style as INT); not restricted to powers of two |
| D12 | QAT recipe | 5 epochs, SGD 0.01 cosine to 0, STE on weights and activations, BN frozen |
| D13 | INT32 overflow | wraparound, documented; never reached by ResNet-8 |
| D14 | Pipeline and accumulator preload for tests | combinational datapath + one register; preload via `clear` + a first accumulate of the preload value |
| D15 | CI scope | decoders exhaustive + 300-vector DP32 subset on apt Icarus; synthesis local only |
| D16 | License | MIT |
| D17 | Default recommend margin | 1.0 point of top-1 below FP32 |
| D18 | Video narration | each owner narrates their lane's shot |
| D19 | When to flip the repo public | morning of Sep 19, or Sep 18 evening if clean |
| D20 | QAT for `int4_b32` | in scope, runs with the other five (last in the run order) |

---

## 13. Glossary (for teammates outside a lane)

- **PTQ / QAT / STE** — post-training quantization (quantize a trained model, no retraining); quantization-aware training (fine-tune with fake quantization in the loop); straight-through estimator (pretend the rounding step has derivative 1 so gradients flow).
- **Fake quantization** — quantize then immediately dequantize inside the float model, so the network "sees" quantization error while everything else stays FP32.
- **BN folding** — merging batch-norm's scale and shift into the preceding conv's weights and bias, which is what real hardware consumes.
- **E4M3, E2M1, E8M0** — floating formats named by exponent and mantissa bits; E8M0 is an exponent-only scale.
- **Microscaling (MX)** — 32 elements share one E8M0 scale so each element can be tiny.
- **DP32** — 32-lane dot-product unit; our unit of hardware comparison.
- **Fused stage** — align all products to one exponent, add exactly, normalize and round once.
- **Sticky bit** — one bit that remembers whether any nonzero bit was shifted out during alignment, so rounding stays correct.
- **RNE** — round to nearest, ties to even.
- **Liberty (`.lib`)** — the cell library file with each standard cell's area and timing; ours is `sky130_fd_sc_hd__tt_025C_1v80`.
- **Yosys / ABC / `dfflibmap`** — open synthesis: Yosys parses and optimizes; `dfflibmap` maps flip-flops to library cells; ABC maps combinational logic and can target a delay (`-D`).
- **OpenSTA** — static timing analyzer; reports the critical path delay of the netlist.
- **Pre-layout** — no placement, no routing, no wires: cell area and cell delay only.
- **Pareto front / knee** — the set of formats no other format beats on both accuracy and area; the knee is the best accuracy-per-area point on that set.

---

## Appendix A — CSV schemas (columns, in order)

- `results/accuracy.csv`: `format, stage(none|ptq|qat), top1, top5, calib_images, calib_stat, quantize_first_last(bool), epochs, seed, checkpoint, date`
- `results/area.csv`: `lib, format, unit, target(unc|t1|t2), dtarget_ps, align_w, run, area_um2, cell_count, yosys_version, date`
- `results/timing.csv`: `lib, format, unit, target, align_w, run, period_ns, delay_ps, wns_ps, sta_version, date`
- `results/breakdown.csv`: `lib, format, target, align_w, stage(decode|mul|align|tree|normacc|other), area_um2, date`
- `results/fidelity.csv`: `format, layer, n_outputs, max_abs_diff, mean_abs_diff, max_rel_diff, layer_output_change, exact(bool), date`

## Appendix B — Command cheat sheet

```bash
source ~/tools/oss-cad-suite/environment && source .venv/bin/activate
make test                      # full cocotb suite (10k vectors)
pytest tb -q -m "not slow"     # CI subset
make synth FMT=int8 TARGET=unc # one unit, one target
make synth                     # everything in synth/targets.toml
make sta                       # time every netlist in synth/out
make plot                      # regenerate results/figures
formatscope recommend --min-acc 0.87
```

## Appendix C — References

- OCP 8-bit Floating Point Specification (OFP8), E4M3 and E5M2 definitions.
- OCP Microscaling Formats (MX) Specification v1.0: MXINT8, MXFP4, E8M0 shared scale, block size 32.
- van Baalen et al., "FP8 versus INT8 for efficient deep learning inference," arXiv 2303.17951 (the 183% figure).
- Micikevicius et al., "FP8 Formats for Deep Learning," arXiv 2209.05433.
- Rouhani et al., "Microscaling Data Formats for Deep Learning," arXiv 2310.10537.
- MetaGrad (arXiv 2303.02347), Table 1: naive INT4 on ResNet-20/CIFAR-10.
- YosysHQ OSS CAD Suite releases; Yosys `abc -D`, `stat -liberty`, `dfflibmap` documentation.
- parallaxsw/OpenSTA README (build dependencies, Brewfile, Dockerfiles).
- The-OpenROAD-Project/asap7 (BSD-3).
