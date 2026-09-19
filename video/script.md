# Demo video script (build-plan.md §9.3)

Target length **4:10** (the window is 3–5 minutes). 1080p screen capture, terminal font ≥ 18 pt,
captions on. Narration is written at ~140 words per minute; each shot lists its word budget, and
the narration under it is inside that budget. Record shots separately and cut them together — no
shot needs to survive in one take.

Every number below is from the committed CSVs at the `v1.0` tag. If a number changes before the
tag, change it here first, then re-record only the shot it appears in.

**Open before recording:** a terminal in the repo with the venv and OSS CAD Suite sourced,
`results/figures/frontier_sky130hd_t1.png`, `results/figures/breakdown.png`, and
`results/figures/results_table.png`.

| Time | Shot | Owner | Words |
|------|------|-------|-------|
| 0:00–0:25 | Hook | RG | 58 |
| 0:25–1:15 | `formatscope demo` live | RG | 115 |
| 1:15–2:05 | The frontier and the breakdown | SN | 115 |
| 2:05–2:45 | Proof: the test suite | SH | 92 |
| 2:45–3:35 | Findings: H1 and H2 | SN | 115 |
| 3:35–4:10 | Limits and next | RG | 80 |

---

## 0:00–0:25 — Hook (RG)

**On screen.** Title card: *FormatScope — accuracy per micron of silicon*. Then
`results/figures/frontier_sky130hd_t1.png` fading in, held to the end of the shot.

> Which number format buys the most accuracy per micron of silicon? Papers answer half of that
> question: they publish accuracy, and estimate area from a single multiplier. We built both halves.
> Five 32-lane dot-product units, one per format, verified bit-exactly, synthesized to open sky130
> cells, and joined to the accuracy of the same network by one command.

---

## 0:25–1:15 — `formatscope demo` live (RG)

**On screen.** Clear terminal, then type and run:

```
formatscope demo
```

That runs one INT8 synthesis on camera (keep it to INT8; it is the fastest unit) and then prints the
recommendation. Let the Yosys output scroll; do not cut away from it — the point is that it is real.

> This is the tool, running now. It synthesizes the INT8 unit against the sky130 cell library, in
> the open Yosys flow, with nothing cached.
>
> [when the area line prints] There it is: the INT8 dot-product unit, about a hundred thousand
> square microns.
>
> Then it asks the question a designer actually has: what is the smallest unit that stays within one
> point of the FP32 network's accuracy? INT8, at ninety-nine thousand square microns. Change the
> question to an area budget instead, and the answer changes with it.

**Second command, same shot** (this one is instant):

```
formatscope recommend --area-budget 40000 --stage qat
```

> Under forty thousand square microns, the answer is INT4 with a block scale, after a five-epoch fine-tune: 83 percent accuracy for a third of INT8's area.

---

## 1:15–2:05 — The frontier and the breakdown (SN)

**On screen.** `frontier_sky130hd_t1.png` full frame. Point at INT8, then FP8, then MXFP4. Then cut
to `breakdown.png`.

> Every unit here meets the same 36.6 nanosecond clock, so this is area at matched delay, not area
> alone. Accuracy is top-1 on CIFAR-10; filled markers are post-training quantization, hollow
> markers are after a five-epoch fine-tune.
>
> INT8 sits here, at ninety-nine thousand square microns and 86.4 percent. FP8-E4M3 is twelve
> percent larger for the same accuracy. And MXFP4 — four bits with a shared block scale — is a third
> of INT8's area and, after fine-tuning, within 3.2 points of it.
>
> The breakdown says *why* each format costs what it does, because we synthesize each stage
> separately.

---

## 2:05–2:45 — Proof: the test suite (SH)

**On screen.** `make test` running to green, then scroll back to the decoder and vector lines.

> None of that means anything if the hardware is wrong. Every decoder is tested over every code it
> can receive: 256 FP8 codes, 256 block scales, 16 for the 4-bit format. Every unit runs ten
> thousand random vectors, checking the accumulator on every single cycle, against a NumPy reference
> written before the RTL — and the corner cases each assert that they really exercise the path they
> are named after. Then one real layer of the network goes through both paths: the INT and MX units
> are bit-exact against PyTorch.

---

## 2:45–3:35 — Findings: H1 and H2 (SN)

**On screen.** `results_table.png`, then back to `breakdown.png` for the alignment bar.

> Our first hypothesis was that a fused dot product makes FP8 much cheaper than single-multiplier
> estimates suggest. Qualcomm's widely cited figure is 50 to 180 percent more area for FP8 over
> INT8. Ours is **12.6 percent** — because the 32 products and the accumulator are aligned once,
> summed in one integer tree, and rounded once. The honest caveat: at that shared clock the FP8 unit
> is also about three times slower, so this is area at a relaxed clock, not at equal throughput.
>
> Second: block scaling is the cheap lever. MXINT8 costs eleven percent over INT8 and buys nothing
> on this model. MXFP4 costs twenty-seven percent over INT4 and, after fine-tuning, goes from 60.7
> to 82.8 percent — closing 87 percent of the gap to INT8 at a third of its area.

---

## 3:35–4:10 — Limits and next (RG)

**On screen.** The `## Limitations` section of the README, scrolling slowly. End on the repo URL.

> The limits, plainly. These are pre-layout cell numbers: no routing, no wires, so the ratios
> transfer and the absolute micron counts do not. One clock, not a sweep, because the INT units are
> three times faster than the fused ones and no tighter clock constrains both. Power is out of
> scope, and the 7-nanometer rerun did not happen.
>
> Everything here regenerates from six make commands, on open tools, from a public repository.

---

## Pre-flight checklist

- [ ] `v1.0` numbers and this script agree (area, delay, accuracy, percentages)
- [ ] `formatscope demo` rehearsed once; INT8 synthesis finishes in under 60 s on the recording machine
- [ ] A pre-recorded fallback clip of `formatscope demo`, in case the live run stalls (risk 15)
- [ ] Figures regenerated after the last CSV change (`make plot`) — the frontier caption and the
      hollow QAT markers must match what the accuracy CSV now holds
- [ ] Terminal font ≥ 18 pt, window sized so nothing wraps mid-command
- [ ] Captions burned in or uploaded as a track
- [ ] Final cut between 3:00 and 5:00
- [ ] Uploaded, link opened in a private window, and pasted into the submission form

## INT4-b32 (measured Sep 19, already in the numbers above)

All six units are now synthesized and timed. The H2 shot has room for one more sentence:

> INT4 elements with the same block-32 scale reach 83.2 percent for thirty-six thousand square
> microns — the same area as MXFP4, slightly better accuracy — so the block scale is what buys the
> accuracy, not the floating-point element encoding.
