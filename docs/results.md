# Results write-up

Numbers below are from `results/accuracy.csv` (quant lane) and `results/area.csv`
(synthesis lane, sky130hd, unconstrained, `align_w=24`), both current as of 2026-09-18.

## H2 — block scaling is the cheap lever

> Block scaling is the cheap lever: MXINT8 costs a small increment over INT8, and
> MXFP4 costs little more than INT4 while closing most of the accuracy gap to INT8
> after fine-tuning.

**Area.** MXINT8 costs 110,692 µm² against INT8's 100,177 µm² -- a 10.5% increment
for the block-exponent logic and the fused-stage's normalize/round path, against
INT8's exact-integer datapath. MXFP4 costs 35,967 µm² against INT4's 27,906 µm² --
a 28.9% increment, noticeably more than MXINT8's, but MXFP4 is still 2.8x smaller
than INT8 and 3.1x smaller than FP8-E4M3 (111,259 µm²). Both increments are the
same fused-stage/block-exponent hardware relative to their INT baseline; the size
of the increment scales with how much of the datapath the exponent logic touches
(INT8's tree is already 21 bits wide before MXINT8 adds its exponent path, while
INT4's is much narrower, so the E2M1 decoders and fused stage are a proportionally
bigger addition).

**Accuracy.** Confirmed strongly. At PTQ, MXFP4 (60.71%) sits at parity with plain
INT4 (61.65%) and 25.68pp below INT8 (86.39%) -- the E2M1 encoding on its own
isn't paying for itself yet at this stage (see the ablation below). After the
5-epoch QAT fine-tune, MXFP4 reaches 82.79%, closing 87% of that gap
(25.68pp -> 3.24pp). MXINT8 needed no real recovery: 86.39% PTQ -> 86.19% QAT,
already within noise of INT8 throughout.

**Verdict: H2 holds**, with the accuracy half specifically dependent on QAT --
the PTQ-stage numbers alone would not have supported "closes most of the gap."

## Ablation: INT4 vs INT4-b32 vs MXFP4

`int4_b32` (plain INT4 elements, block-32 power-of-two scale, no hardware unit)
isolates how much of MXFP4's story is block scaling alone versus the E2M1
encoding on top of it.

| stage | int4 | int4_b32 | mxfp4 |
|---|---|---|---|
| PTQ | 61.65% | 78.46% | 60.71% |
| QAT | 81.36% | 83.21% | 82.79% |

At PTQ, block scaling alone (int4_b32) already recovers 16.81 of INT4's points
without touching the element encoding -- most of the naive-INT4 damage is from a
single per-tensor scale getting stretched by outliers, and per-block scaling
fixes that directly. E2M1 on top of block scaling (mxfp4) does *not* help at PTQ
(60.71%, indistinguishable from plain int4) -- its coarse, log-spaced codes
{0, 0.5, 1, 1.5, 2, 3, 4, 6} are worse than uniform INT4 steps for weights that
haven't been trained around them yet. QAT changes this: mxfp4 (82.79%) closes to
within 0.4pp of int4_b32 (83.21%), i.e. once the network adapts, E2M1's extra bit
of dynamic range stops being a liability and roughly matches block-scaled INT4.
**One-sentence version**: block scaling explains most of MXFP4's advantage over
naive INT4 at every stage, and E2M1 only pulls its weight after fine-tuning.

## H1, H3

Silicon-side hypotheses (fused-stage area ratio vs. Qualcomm's 183% single-MAC
estimate; ASAP7 ranking) are Lane S's to write up -- not duplicated here.
