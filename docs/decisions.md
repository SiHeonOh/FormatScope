# Decision log

Design decisions the results depend on, numbered as in the build plan (§12).
Each was adopted as the recommended default at the Sep 15, 2026 re-plan unless
a later entry says otherwise; the *Evidence* column points at what the choice
rests on now that the numbers exist.

| ID | Decision | Choice | Evidence / status |
|----|----------|--------|-------------------|
| D1 | INT code ranges | Symmetric: ±7 (INT4) and ±127 (INT8); the most-negative code is never emitted by the quantizer and is treated as an ordinary code by the RTL | `quant/formats.py`; `tb/test_dp32_int.py` corner cases exercise the unused code |
| D2 | FP8 overflow | Saturate to ±448 (OCP-permitted; matches inference practice) | `quant/formats.py` |
| D3 | FP8 NaN in hardware | Lane term = 0, sticky `flag_nan` | `rtl/fp8e4m3/dp32_fp8e4m3.v`; `tb/test_dp32_fp8.py` |
| D4 | E8M0 code 0xFF in hardware | Block term = 0, sticky `flag_nan` | `rtl/mxint8/`, `rtl/mxfp4/`; `tb/test_dp32_mx.py` |
| D5 | MXINT8 code −128 | Quantizer never emits it; RTL treats it as an ordinary int8 | as D1 |
| D6 | Alignment window widths | 24 (default) and 32 | Measured Sep 17: at ALIGN_W 32 the FP8 unit is +17% area for no delay gain; MXINT8 and MXFP4 within 3%. `results/area.csv` |
| D7 | Delay-target rule | T1 = 1.03 × (slowest unconstrained unit's arrival + setup, i.e. period − WNS from its unconstrained STA row), rounded up to 100 ps. **No T2.** | **Amended Sep 17 (Rakshita):** the 3% guard band covers the ~1.5% gap between ABC's internal timing and OpenSTA, which otherwise left FP8 and MXINT8 ~0.6 ns short of a target set exactly at their arrival time. T2 = 0.7 × T1 was dropped: the INT units are ~3× faster than the fused units at their fastest mapping, so no period tighter than T1 constrains both groups, and ABC cannot map a unit faster than its delay-optimal result. Each unit is reported unconstrained and at T1. `synth/targets.toml` |
| D8 | "Spread across runs" | Five perturbation runs at ±1%, ±2% of `-D`, plus a determinism check | `make synth-perturb`; determinism confirmed Sep 17 (repeated INT8 runs identical) |
| D9 | ResNet-8 definition and BN | 6n+2 with n=1, widths 16/32/64, zero-pad shortcuts; BN folded before quantization | `models/resnet8.py`; `quant/fakequant.py` |
| D10 | Activation calibration statistic | 99.99th percentile of \|x\| over 512 images | `quant/calibrate.py`; `results/accuracy.csv` column `calib_stat` |
| D11 | FP8 weight/activation scale | Max maps just under 448 (same style as INT); not restricted to powers of two | `quant/formats.py` |
| D12 | QAT recipe | 5 epochs, SGD 0.01 cosine to 0, STE on weights and activations, BN frozen | pending `quant/qat.py` |
| D13 | INT32 overflow | Wraparound, documented; never reached by ResNet-8 | `tb/test_dp32_int.py::dp32_int_wraparound` |
| D14 | Pipeline and accumulator preload for tests | Combinational datapath + one register; the testbench builds the prior accumulator from the stream itself | `tb/test_dp32_int.py` docstring |
| D15 | CI scope | Decoders exhaustive + 300-vector DP32 subset on apt Icarus; the 10,000-vector runs are local (`make test`); synthesis is local only | `.github/workflows/ci.yml`; 10k runs green on Sep 17 |
| D16 | License | MIT | `LICENSE` |
| D17 | Default recommend margin | 1.0 point of top-1 below FP32 | `tool/formatscope/recommend.py` |
| D18 | Video narration | Each owner narrates their lane's shot | — |
| D19 | When to flip the repo public | Morning of Sep 19, or Sep 18 evening if clean | — |
| D20 | QAT for `int4_b32` | In scope, runs with the other five (last in the run order) | pending `quant/qat.py` |

## Synthesis flow decisions not in the original list

| ID | Decision | Choice | Evidence / status |
|----|----------|--------|-------------------|
| S1 | `synth -noalumacc` on every unit | Keep the 32 multipliers and the adder tree as separate cells for ABC | Sep 17 (Si Heon): with the default `$macc` fusion ABC took >150 s for INT8 and >17 min for MXINT8 and produced larger netlists; without it every unit maps in seconds and INT8 is 11% smaller. `results/archive/area_macc.csv` holds the fused-flow numbers |
| S2 | ABC constraint file on every unit | `synth/abc_sky130hd.constr`: driving cell `sky130_fd_sc_hd__buf_1`, output load 3.898 fF (the OpenROAD-flow-scripts sky130hd values) | Sep 17 (Rakshita): without `-constr`, Yosys hands ABC a genlib with unit gate delays, so `-D` counts logic levels and any target above the depth is a no-op — the first T1/T2 rows were identical to unconstrained. With it, ABC reads the liberty and its reported delay matches OpenSTA within 1.5%. `results/archive/area_genlib.csv` holds the pre-fix numbers |
