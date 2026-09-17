---
artifact_contract: "ce-handoff/v1"
created_at: "2026-09-17T03:50:00Z"
title: "FormatScope desktop bring-up: verify INT8/INT4, then area and delay"
summary: "Hands a Windows 11 / RTX 4070 Ti session the first real run of the cocotb suite on Si Heon's INT8/INT4 units, then the first real Yosys+OpenSTA numbers, then the CUDA PTQ reproduction."
keywords: ["formatscope", "wsl2", "yosys", "opensta", "sky130", "cocotb", "cuda", "ptq", "lane-s", "desktop-bringup", "rg/integration"]
cwd: "/Users/rakki/Documents/Projects/FORGE"
resume_focus: "On rg/integration: make test-ci (first cocotb run), then INT8/INT4 synthesis + STA for the area/delay numbers Si Heon is waiting on, then reproduce_ptq.sh."
repository: "FormatScope"
repo_root_sha: "639ee3be7e7f98bb74015cfb805cdb96f308c0a3"
branch: "rg/integration"
head: "4059e92cf5bed8fab14c283509f56aecfe63b616"
worktree_path: "/Users/rakki/Documents/Projects/FORGE"
---

# FormatScope desktop bring-up

## Objective

Stand up FormatScope on the Windows 11 desktop (RTX 4070 Ti) inside WSL2 Ubuntu and produce three things that have never been produced anywhere, in this order:

1. **Verification** — the cocotb suite in `tb/` against Si Heon's `dp32_int8` and `dp32_int4`. Rule 2 of the build plan: no RTL merges to `main` without its cocotb test green. It has never run.
2. **The efficiency numbers Si Heon is waiting on** — area (µm², from Yosys `stat`) and delay (ps, from OpenSTA) for the INT8 and INT4 units on sky130 HD, unconstrained. These are the first real rows of `results/area.csv` and `results/timing.csv`, and the first contact between the lane-S scripts and real tools.
3. **The CUDA reproduction** — retrain the FP32 ResNet-8 baseline and re-run PTQ for all six formats, checked against the committed table.

Written on the user's MacBook Air (M5), which has no `torch`, `yosys`, `sta`, or `iverilog`. Nothing below has been verified against real tools by construction.

## People

- **Rakshita Gupta (RG)** — the user. Owns synthesis, OpenSTA, the `formatscope` tool, and the GPU machines.
- **Si Heon Oh (SH)** — appears in git as **"Ocean"**. Owns RTL and cocotb. Pushed `sh/rtl-int` tonight and is asking RG for the area/delay numbers.
- **Seungmin Nam (SN)** — appears as **"Ryan Seungmin Nam"** / "immaGodgivenmasterpiece". Owns quantization and training.

## Branch to use: `rg/integration`

Built tonight on the Mac: `rg/synth-sta-tool` (lane S + tool + plumbing) merged with `origin/sh/rtl-int` (Si Heon's RTL + `tb/`). The merge was clean — the two branches touch disjoint files. Neither parent alone can run the real flow: one has `synth/` but only smoke RTL, the other has RTL but no `synth/`.

```bash
git clone https://github.com/SiHeonOh/FormatScope.git ~/FormatScope
cd ~/FormatScope && git checkout rg/integration
```

`main` still holds only `docs/`, `models/`, `quant/`, `results/`.

## Current state by maturity

**Complete, merged to `main` (PR #1, Sep 12):**
- `quant/` — formats, fakequant, calibration, eval. `models/` — ResNet-8, loaders, training. `results/accuracy.csv` — the reference table, produced on Apple MPS.

**Complete RTL, tests never run (`sh/rtl-int`, now in `rg/integration`):**
- `rtl/common/adder_tree.v`, `rtl/int8/dp32_int8.v`, `rtl/int4/dp32_int4.v`.
- `tb/conftest.py`, `tb/refs.py`, `tb/test_dp32_int.py` — 300-vector CI subset + corners + INT8 wraparound; 10k vectors behind `@pytest.mark.slow`.
- Interface check passed on the Mac: `synth/run_synth.py` lines 61–62 name exactly these source paths, and `top_name()` yields `dp32_int8`/`dp32_int4`, matching the module names. The `ALIGN_W` parameter both units carry ("unused; keeps the port list common") is what the synth template's `chparam` binds to.

**Written, never executed against real tools:**
- `synth/run_synth.py` + `.ys` templates; `sta/run_sta.py` + `sta.tcl.template`. Their output parsers (`synth/tests/test_parsers.py`, `sta/tests/test_parsers.py`) were written from *recalled* Yosys/OpenSTA formats. **Expect them to be wrong in small ways** — fixing them against captured output is expected work.
- `quant/tests/test_fakequant.py` — needs torch. First run will be on the PC.

**Verified on the Mac tonight:** `pytest tool/tests synth/tests sta/tests quant/tests/test_formats.py` → 64 passed on `rg/integration`. `bash -n` on all scripts; `compare_accuracy.py` self-test passes.

**Not started (not this session's job):**
- RTL for `fp8e4m3`, `mxint8`, `mxfp4` and their shared files (`lzc.v`, `fused_stage.v`, `decode_*.v`). `run_synth.py` lists them; they will fail with missing sources until Si Heon lands them. FP8 is due bit-exact Thu 18:00.
- QAT — `qat.py` does not exist. `fidelity.py` does not exist. README and LICENSE do not exist.

## Desktop blocker history

`wsl --install -d Ubuntu-24.04` failed with `0x80370114` (required feature not installed). RG ran the DISM feature enables and `bcdedit /set hypervisorlaunchtype auto`, then rebooted. **Whether that fixed it is unconfirmed** — the first thing this session does is check. If `wsl --version` fails or Ubuntu won't register, Virtualization in Task Manager → Performance → CPU decides it: Disabled means BIOS (VT-x / AMD SVM), and no Windows command helps.

If a broken registration is left over: `wsl --unregister Ubuntu-24.04` is safe (nothing was ever set up in it).

## The scripts

| Path | Runs on | Purpose |
|---|---|---|
| `scripts/setup_desktop_wsl.sh` | Ubuntu | Full bring-up. Finds the repo (clone, or seeds from a Windows folder if one exists), builds the venv with CUDA torch, delegates to `setup_lane_s_wsl.sh`, runs `make test-ci`. |
| `scripts/setup_lane_s_wsl.sh` | Ubuntu | Pre-existing. OSS CAD Suite 2026-09-09, sky130 HD liberty via `ciel`, OpenSTA (Docker if available, else native build), both smoke tests. |
| `scripts/reproduce_ptq.sh` | Ubuntu | Train → PTQ ×6 → diff against the committed table via `scripts/compare_accuracy.py`. |
| `scripts/setup_desktop_win.ps1` | Windows | Optional; WSL install + driver check. Its core is one command. |

Read `scripts/setup_desktop_wsl.sh` lines 1–20 for why it copies to `~/FormatScope` before changing it.

`make test-ci` now expands to `pytest quant/tests tool/tests synth/tests sta/tests tb -q -m "not slow"` — `tb` is picked up by `$(wildcard tb)` and needs `iverilog` (from OSS CAD Suite) and `cocotb~=2.0` (in the `[test]` extras). CI triggers only on pushes to `main` and on PRs, so branch pushes don't run it.

## Intended sequence

```bash
bash scripts/setup_desktop_wsl.sh            # env + lane S smoke + make test-ci (verification runs here)
make synth FMT=int8 && make synth FMT=int4   # unconstrained, ALIGN_W 24 → results/area.csv
make sta                                     # → results/timing.csv
make synth FMT=int8 && make synth FMT=int8   # determinism check: three INT8 runs should agree
cd ~/FormatScope && bash scripts/reproduce_ptq.sh   # EPOCHS=5 first, then the full 60
```

The synth/STA lines are the deliverable Si Heon is asking for. Send him `results/area.csv` and `results/timing.csv` rows, or the numbers, as soon as they exist.

## Decisions already made — do not relitigate

- **Build in `~/FormatScope`, not `/mnt/c`.** ~10x slower, and a sync client plus git on one `.git` corrupts the index. Risk #12 in the plan.
- **Full scope kept** at the Sep 15 re-plan; cut points are dated instead (section 10.2 of `docs/build-plan.md`).
- **Exact accuracy reproduction is not the bar.** Different backend → different checkpoint → every PTQ number shifts. `compare_accuracy.py` checks tolerance and ordering invariants; see its docstring.
- **Unconstrained runs never get `-D`.** `run_synth.py` refuses it (risk #11). T1/T2 come later from D7 and are RG's call, due Thu 14:00.

## Wrong paths the next agent is likely to retry

- **Do not install an NVIDIA driver inside WSL.** The Windows driver serves the GPU through `/dev/dxg`. If `torch.cuda.is_available()` is False: update the Windows driver, `wsl --shutdown`, retry.
- **Do not check out `rg/synth-sta-tool` or `sh/rtl-int` alone** expecting synthesis on real RTL. Use `rg/integration`.
- **Do not "fix" MXFP4 scoring below INT4** (60.71 vs 61.65 in the committed table). Known open question for Seungmin, possibly real (the `floor(log2 max) − 2` rule clipping block maxima), possibly a bug. `compare_accuracy.py` deliberately doesn't assert that ordering.
- **Do not synthesize `fp8e4m3`/`mxint8`/`mxfp4`.** Sources don't exist yet; `run_synth.py` will error on missing files. That is correct behaviour, not a script bug.
- **Do not treat a zero-area or vanished-module `stat` as success.** Risk #4: Yosys optimized the design away. `run_synth.py` refuses zero-area rows; if it does, the RTL or the `hierarchy -check -top` line is the problem.
- **Do not run `make train` from Windows cmd/PowerShell.** The Makefile uses POSIX `VAR=value cmd` prefixes.

## Authoritative references

- `docs/build-plan.md` §10 (line 646 onward) — status table and the Sep 15–19 calendar with cut points. §10.2 row "Wed Sep 16" for RG is exactly this session's list: INT8+INT4 synth unconstrained + STA, determinism check (3× INT8), fix parser mismatches, first real `formatscope plot`.
- `docs/build-plan.md` §3.7 — the four Day-0 smoke tests and what their passing output looks like. (a) and (d) are RG's.
- `docs/build-plan.md` §7 — lane S in full; §7.2 for the T1/T2 rule (D7).
- `quant/formats.py` line 18 — `FORMAT_IDS`. `quant/eval.py` lines 99–109 — accuracy guardrails.
- `synth/libs.toml` — `[desktop-wsl]` section already points at `$HOME/.ciel/...`.
- `tb/test_dp32_int.py` — what verification actually checks; `FORMATSCOPE_NVEC` env var sets vector count.

## Deadline

Submission **Sat Sep 19, 2026, before noon.** At handoff it is late Wed Sep 16. Gates G0 (Tue) and G1+G2 (Wed) are missed; G3 (FP8 at matched delay) is due Thu 18:00 and depends on Si Heon's fused stage, not on this session. This session's output — verified INT8/INT4 with area and delay — is G1 and the first half of G2, one day late.

## Plausible next steps

Sequential, not alternatives:

1. Confirm WSL boots. Then `setup_desktop_wsl.sh`; triage in the likely order — `ciel` liberty fetch, OpenSTA build, then the parsers. Capture real Yosys and OpenSTA output into the test fixtures as they get fixed.
2. Verification passes (`make test-ci` green including `tb`) → synthesize and time INT8 and INT4 → send Si Heon the numbers.
3. `reproduce_ptq.sh`, `EPOCHS=5` then 60.

## Relevant installed skills

- `superpowers:systematic-debugging` — for parser and tool failures.
- `superpowers:verification-before-completion` — "smoke test printed PASS" and "pytest shows tb passed" are the bars, not "the script ran."
- `compound-engineering:ce-commit` — if asked to commit. **Standing rule: no `Co-Authored-By`, session, or "Generated with" lines in FormatScope commits or PRs.** Commit messages are plain imperative sentences.
