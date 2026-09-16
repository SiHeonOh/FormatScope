---
artifact_contract: "ce-handoff/v1"
created_at: "2026-09-16T04:40:31Z"
title: "FormatScope desktop bring-up: WSL2 lane S + CUDA PTQ reproduction"
summary: "Hands a Windows 11 / RTX 4070 Ti session the first real Yosys+OpenSTA run of the lane-S scripts and the CUDA reproduction of the PTQ accuracy table."
keywords: ["formatscope", "wsl2", "yosys", "opensta", "sky130", "cuda", "ptq", "quantization", "lane-s", "desktop-bringup"]
cwd: "/Users/rakki/Documents/Projects/FORGE"
resume_focus: "Run scripts/setup_desktop_wsl.sh then scripts/reproduce_ptq.sh on the 4070 Ti; fix the synth/STA parsers against real tool output."
repository: "FormatScope"
repo_root_sha: "639ee3be7e7f98bb74015cfb805cdb96f308c0a3"
branch: "rg/synth-sta-tool"
head: "06a8fba109809d2b78d946575bdea9e9c183e9be"
worktree_path: "/Users/rakki/Documents/Projects/FORGE"
---

# FormatScope desktop bring-up

## Objective

Stand up the FormatScope environment on the user's Windows 11 desktop (RTX 4070 Ti) inside WSL2 Ubuntu, then run two things that have never run anywhere:

1. **Lane S** — Yosys synthesis and OpenSTA timing against the sky130 HD liberty. The scripts exist but have **never touched a real Yosys or OpenSTA binary**.
2. **The CUDA reproduction** — retrain the FP32 ResNet-8 baseline and re-run post-training quantization for all six formats, then check the result against the committed accuracy table.

This handoff was written on the user's MacBook Air (M5), which has no `torch`, no `yosys`, no `sta`, and no `iverilog`. Everything below is therefore *unverified against real tools by construction* — the desktop run is the first real test.

## Machine and sync context

- The Mac's `~/Documents` syncs to `C:\Users\rakki\iCloudDrive\Documents` via iCloud Desktop & Documents. The repo therefore reaches the PC **without being pushed**.
- The PC copy lives at `C:\Users\rakki\iCloudDrive\Documents\Projects\FORGE`, reachable from WSL as `/mnt/c/Users/rakki/iCloudDrive/Documents/Projects/FORGE`.
- There may also be a Mac mini with its own copy. Unconfirmed, and nobody has reconciled them. If `synth/` or `rtl/` content on the desktop disagrees with what this handoff describes, stop and ask the user rather than merging.

## Current state by maturity

**Complete and verified (syntax/logic only, on the Mac):**
- `quant/` — formats, fakequant, calibration, eval. Merged to `main` as PR #1 on Sep 12.
- `models/` — ResNet-8, CIFAR-10 loaders, training loop. Also from PR #1.
- `results/accuracy.csv` — the committed reference table, produced on Apple MPS.

**Written, never executed against real tools (the whole point of this session):**
- `synth/run_synth.py` + `synth/synth_flat.ys.template`, `synth_hier.ys.template`
- `sta/run_sta.py` + `sta/sta.tcl.template`
- Their parsers in `synth/tests/test_parsers.py` and `sta/tests/test_parsers.py` were written from *recalled* Yosys/OpenSTA output formats, not from captured output. **Expect these to be wrong in small ways.** Fixing them is the expected work of this session, not a surprise.

**Written, never executed at all:**
- `quant/tests/test_fakequant.py` — needs `torch`, which the Mac lacks. First run will be on the PC.

**Not started:**
- RTL beyond `rtl/smoke/` (two trivial Verilog files: `smoke_add.v`, `smoke_add_reg.v`). No INT8/INT4/FP8 dot-product units exist. This is the project's critical path and is not this session's job.
- QAT. Depends on a `qat.py` that does not exist yet (Seungmin's).

## Uncommitted and fragile state (machine-local)

Branch is `rg/synth-sta-tool` at `06a8fba`. `main` and `origin/main` contain **only** `docs/`, `models/`, `quant/`, `results/`.

Untracked, existing nowhere but this Mac and the synced PC copy:

```
.github/  Makefile  pyproject.toml  rtl/  scripts/  sta/  synth/  tool/
```

Plus modified `docs/build-plan.md` (+43/-20 — the revised Sep 15–19 calendar in section 10).

**None of this is pushed.** A `git clone` on the PC would get none of it. That is why the setup script seeds from the synced Windows folder rather than cloning. The user has been asked twice whether to push a branch and has not yet said yes — **do not push without asking.**

## The scripts

All four are new and unrun. Repository-relative:

| Path | Runs on | Purpose |
|---|---|---|
| `scripts/setup_desktop_win.ps1` | Windows, admin PowerShell | WSL2 + Ubuntu 24.04 install, NVIDIA driver check. Optional — its core is one command, `wsl --install -d Ubuntu-24.04`. |
| `scripts/setup_desktop_wsl.sh` | Ubuntu | Full bring-up. Seeds the repo, builds the venv with CUDA torch, delegates to `setup_lane_s_wsl.sh`, runs `make test-ci`. |
| `scripts/setup_lane_s_wsl.sh` | Ubuntu | **Pre-existing, written by an earlier session.** OSS CAD Suite 2026-09-09, sky130 HD liberty via `ciel`, OpenSTA (Docker if available, else native build with CUDD), then both smoke tests. |
| `scripts/reproduce_ptq.sh` | Ubuntu | Train → PTQ ×6 → diff against the committed table via `scripts/compare_accuracy.py`. |

Read `scripts/setup_desktop_wsl.sh` lines 1–20 for the design rationale before changing anything in it.

### Intended sequence

```bash
# Windows, admin PowerShell (once):   wsl --install -d Ubuntu-24.04   then reboot
bash /mnt/c/Users/rakki/iCloudDrive/Documents/Projects/FORGE/scripts/setup_desktop_wsl.sh
cd ~/FormatScope && bash scripts/reproduce_ptq.sh
```

## Decisions already made — do not relitigate

**The project is copied to `~/FormatScope`, not built in `/mnt/c`.** Two reasons: the Windows filesystem is roughly 10x slower under WSL, and a sync client plus git writing the same `.git` from two machines corrupts the index. The copy keeps `origin`, so pull/push still work. `setup_lane_s_wsl.sh` line 29 independently warns about `/mnt/*`.

**Full scope was kept** when the timeline was re-planned on Sep 15, against the option of cutting. Section 10.2 of `docs/build-plan.md` holds the dated cut points instead: drop ASAP7 if FP8 isn't bit-exact by Thu 18:00, drop the third delay target if the MX units aren't passing by Fri 11:00.

**Exact accuracy reproduction is explicitly not the bar.** The committed numbers came from an MPS run; a CUDA run produces a different FP32 checkpoint and every PTQ number moves with it. `scripts/compare_accuracy.py` checks each row within ±2 points *and* the ordering invariants, which are the project's actual claims. See its module docstring.

**Lane S installs nothing on the Mac.** The user directed "scripts only, run on desktop" when an earlier session offered to install the EDA tools locally.

## Wrong paths the next agent is likely to retry

- **Do not install an NVIDIA driver inside WSL.** The Windows driver already exposes the GPU through `/dev/dxg`; a second one breaks it. If `torch.cuda.is_available()` is False, the fix is updating the *Windows* driver and running `wsl --shutdown`.
- **Do not `git clone` and expect lane S.** `synth/`, `sta/`, `tool/`, `Makefile` are not on any remote. Seed from the Windows folder.
- **Do not use `make train` / `EPOCHS=60 make train` from Windows cmd or PowerShell.** The Makefile uses POSIX `VAR=value cmd` prefixes. Run inside WSL.
- **Do not "fix" MXFP4 scoring below INT4.** In the committed table MXFP4 is 60.71% and plain INT4 is 61.65%, while INT4 with block scales is 78.46%. This is a **known open question** — possibly a real effect of the MX scale rule clipping large values, possibly a bug in `quant/formats.py`. It is flagged for Seungmin. `compare_accuracy.py` deliberately does *not* assert MXFP4 > INT4, so it won't mask the anomaly either way.
- **Do not treat an out-of-tolerance row as automatic failure** without checking the invariants block underneath it. A uniformly shifted table with intact ordering means a different-but-valid checkpoint.

## Verification actually performed

On the Mac, before handoff:

- `bash -n` on all four shell scripts — pass.
- `python3 -m py_compile scripts/compare_accuracy.py` — pass.
- `scripts/compare_accuracy.py` self-test with `results/accuracy.csv` as its own candidate — 7/7 rows ok, 4/4 invariants ok, "PASS".

**Not verified, and cannot be on this machine:** anything importing `torch`; anything invoking `yosys`, `sta`, `iverilog`, or `ciel`; the sky130 liberty download; the OpenSTA Docker build and its native fallback; every `apt-get` line; the `find_icloud` glob against a real `/mnt/c`. A previous session recorded 64 passing tests covering the tool, parsers, and Seungmin's format tests — `quant/tests/test_fakequant.py` was excluded for lack of torch.

## Authoritative references

- `docs/build-plan.md` section 10 (line 646 onward) — status table and the revised Sep 15–19 calendar with cut points. Section 10.1 is where we are; 10.2 is the plan.
- `docs/glossary.md` — format and EDA vocabulary.
- `quant/formats.py` line 18 — `FORMAT_IDS`, the canonical six.
- `quant/eval.py` lines 99–109 — `_sanity_check`, the accuracy guardrails and what a scale/rounding bug looks like.
- `synth/libs.toml` — per-machine liberty paths. The `[desktop-wsl]` section already exists and points at `$HOME/.ciel/...`; `setup_lane_s_wsl.sh` appends a section for `$FORMATSCOPE_MACHINE` if absent.
- `results/accuracy.csv` — the reference table. `reproduce_ptq.sh` restores it after each run, so it should stay unmodified in git.

## Deadline

Submission is **Sat Sep 19, 2026**; the video is due that day. Four days out at the time of writing, with RTL not started. Lane S working on the desktop unblocks the area/delay half of every claim in the proposal.

## Plausible next steps

1. **Run the bring-up and fix what breaks.** `setup_desktop_wsl.sh`, then triage. Most likely failure points, in order: the sky130 liberty fetch through `ciel`, the OpenSTA build, then the synth/STA output parsers. Capture real Yosys and OpenSTA output into the test fixtures as you fix them, so the parsers are pinned to observed reality rather than recall.
2. **Then reproduce the accuracy table** with `reproduce_ptq.sh`. Consider `EPOCHS=5` first as an end-to-end smoke, then the full 60.

These are sequential, not alternatives.

## Relevant installed skills

- `superpowers:systematic-debugging` — for the parser failures, which are the expected work here.
- `superpowers:verification-before-completion` — before claiming lane S works; "the smoke test printed PASS" is the bar, not "the script ran".
- `compound-engineering:ce-commit` — if the user asks to commit the fixes. **Note the user's standing rule: no `Co-Authored-By`, session, or "Generated with" lines in FormatScope commits or PRs.**
