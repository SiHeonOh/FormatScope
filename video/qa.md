# Showcase Q&A (build-plan.md §9.5)

Prepared answers for the Sep 22 showcase. Each owner writes their lane's answers; the
"Owner" line says who. Answers are meant to be spoken in under a minute, then point at
the file that proves it.

## Why a fused stage with a single rounding?

*Owner: Si Heon.*

Because that is what makes an FP8 dot product cheap. Thirty-two separate FP8 multiply-adds
would each need a full floating-point adder: align two operands, add, normalize, round.
That is 32 shifters, 32 normalizers, and 32 rounders, and Qualcomm's 50–180% single-MAC
penalty over INT8 is exactly that cost. The fused stage does the work once for all 33 terms
(32 products plus the running total): one maximum-exponent search, one shift per term into a
shared window, one integer adder tree, one normalize, one round. The per-lane cost drops to a
4×4-bit significand multiplier and a shifter. Measured on sky130 that is a 12.6% increment
over INT8 at the same clock, and the breakdown shows where it went: the alignment stage is
54% of the FP8 unit, the multipliers 16%.

The price is that rounding once is not what PyTorch does (it accumulates in FP32 and rounds
after every product). We did not assume that away: `quant/fidelity.py` pushes a real layer's
dot products through both paths. INT and MX agree bit for bit; FP8 differs by at most 1.1e-7
absolute over 300 outputs. Files: `rtl/common/fused_stage.v`, `docs/fused-stage.md`,
`results/fidelity.csv`.

## Why an FP32-format accumulator rather than fixed point?

*Owner: Si Heon.*

Range. FP8 products span 2⁻¹⁸ to 448², about 41 binades, and the MX block exponent
(two E8M0 scales) spans ±254 on top of that. A fixed-point register wide enough to hold
every reachable value exactly would be hundreds of bits for MX, and a narrower one would
need a designer to pick where to clip, which changes the answer per model. An FP32-format
register (sign, 8-bit exponent, 23-bit fraction) covers the whole range in 32 bits, keeps
the output in the format the next layer and the software baseline consume, and makes the
fidelity comparison against PyTorch's FP32 accumulate a like-for-like check. It also keeps
the INT and floating units comparable: every unit ends in one 32-bit register.

What we did not build is the parts of FP32 the register can never need. The range argument
in `docs/fused-stage.md` shows an FP8 unit cannot produce a subnormal or an infinity in any
realistic number of accumulations, so there is no subnormal or infinity logic; the tests
assert it instead. MX units can leave the range with extreme scales, so they saturate to the
largest finite value or return +0 (two 10-bit compares), and the tests exercise both.

## How are FP8 subnormals handled?

*Owner: Si Heon.*

Exactly, with no flush to zero. The decoder (`rtl/common/decode_fp8e4m3.v`) gives a normal
code a hidden bit of 1 and an exponent of E−1, and a subnormal code (E = 0) a hidden bit of
0 and exponent 0, on the same scale, so `value = sig × 2^(exp−9)` holds for both and the
fused stage never has to know which it saw. The decoder is tested over all 256 codes against
the software decoder in `quant/formats.py`.

The one precision consequence is documented rather than hidden: products are not
renormalized before alignment (that would cost 32 more leading-zero counters), so a product
involving a subnormal can carry up to 7 leading zeros in its field and push that many
low-order bits of the other terms into sticky. The random test has a subnormal-heavy input
mix, and `tb/test_refs.py` bounds the effect against exact arithmetic. Overflow is the
quantizer's job: it saturates at ±448 (D2), so the hardware never sees a code above the
maximum finite value.

## Why the 24-bit alignment window, and what does 32 buy?

*Owner: Si Heon, with Rakshita for the numbers.*

Twenty-four is FP32's own significand precision, so a 24-bit window with guard, round, and
sticky bits is the narrowest window that can still produce the correctly rounded FP32 result
whenever no bit is shifted out. Thirty-two shows what extra precision costs: +17% area for
the FP8 unit at the shared clock, and under 4% for the MX units, for no delay gain and no
change in the fidelity result. 24 is the default; every 32-bit row is in `results/area.csv`
with `align_w = 32`. Decision D6.

## Why block size 32?

*Owner: Seungmin (software side), Si Heon (hardware side).*

Hardware side: 32 is the OCP Microscaling block size, and it is also the natural width of
a dot-product unit, so one block scale pair covers one whole accumulation and the MX
increment over the INT unit is one exponent add, one normalize, and a two-term fused stage.
Measured: MXINT8 is +11.0% over INT8. A smaller block would need several scale pairs per
cycle and a multi-term fused stage; a larger one would not fit a 32-lane unit.

Software side: *(Seungmin to fill: accuracy effect, the INT4-b32 control row.)*

## Why area at matched delay rather than area alone?

*Owner: Rakshita.*

*(Rakshita to fill: T1 rule, the ABC margin, why there is no T2.)*

## Why per-tensor activation scaling?

*Owner: Seungmin.*

*(Seungmin to fill: D10, D11, what per-channel would change.)*

## What does "spread across runs" mean?

*Owner: Rakshita.*

*(Rakshita to fill: D8, the five perturbed targets, which units move.)*

## How does MXFP4 lose accuracy, and does fine-tuning recover it?

*Owner: Seungmin.*

*(Seungmin to fill: PTQ 60.71 below INT4 61.65, INT4-b32 78.46, QAT 82.79 above INT4's
81.36; the E2M1 element encoding versus the block scale.)*

## Why not energy?

*Owner: Rakshita.*

*(Rakshita to fill: pre-layout, no switching activity, ratios not absolutes.)*

## How does the INT8-vs-FP8 ratio compare with Qualcomm's estimate?

*Owner: Rakshita, with Si Heon for the fused-stage reason.*

*(Rakshita to fill: 12.6% at T1, 11–16% across the perturbation runs, against the 50–180%
single-MAC figure in van Baalen et al.; the fused stage is the difference.)*
