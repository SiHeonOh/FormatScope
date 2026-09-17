# The fused stage: FP8-E4M3 and MX accumulation

This is the bit-width derivation required by build-plan.md §5.4. `rtl/common/fused_stage.v` and `tb/refs.py::fused_stage` implement exactly the steps below, so any mismatch between them is an RTL bug or a documented reference bug, never ambiguity.

## In plain words

An INT unit adds whole numbers, so its sum is exact. FP8 and MX numbers carry an exponent (a "zoom level"), so before 33 of them can be added they must be lined up to a common zoom level, the way 3.2 × 10⁵ and 4.1 × 10² are rewritten over the same power of ten before adding. The fused stage lines up the 32 products and the running total in one step, adds them in one tree, and rounds **once** into a 32-bit floating-point register. Rounding once instead of 32 times is what makes the unit cheaper than 32 separate FP adders, and it is also why its answer can differ very slightly from PyTorch's (measured by `quant/fidelity.py`, not assumed away).

## Parameters

| Parameter | FP8 | MXINT8 | MXFP4 | Meaning |
|-----------|-----|--------|-------|---------|
| `N_PROD` | 32 | 1 | 1 | Product terms entering the stage (the accumulator is one more; `N_TERMS = N_PROD + 1`) |
| `SIG_W` | 8 | 20 | 13 | Width of each product term's magnitude field (MX: the block sum's magnitude, one bit less than its signed width) |
| `ALIGN_W` | 24, 32 | 24, 32 | 24, 32 | Alignment window (decision D6). 24 is FP32's own significand precision |
| `EXP_W` | 10 | 10 | 10 | Signed width of every exponent inside the stage |

`N_PROD` must be a power of two (it feeds `adder_tree`), and `ALIGN_W ≥ 24` (step 7 needs at least one bit below the round bit).

## Exponent convention: top of field

Every term, including the accumulator, is a magnitude field `F` of a fixed width `Wf`, a sign `s`, and a signed exponent `T` such that

    value = (−1)^s × F × 2^(T − Wf)

`T` is the weight of the bit just above the field's MSB. Aligning on `T` lines up the fields bit for bit, whatever their widths.

| Term | `Wf` | `F` | `T` |
|------|------|-----|-----|
| FP8 lane product | 8 | `sig_a × sig_b` | `exp_a + exp_b − 10` |
| Accumulator (FP32) | 24 | `{1, fraction[22:0]}` | `biased_exp − 126` |
| MX block sum | 20 / 13 | `abs(S) << lz` (normalized) | `scale_a + scale_b − OFFSET + SIG_W − lz` (see "MX units" below) |

**FP8 decoder** (`decode_fp8e4m3.v`, `refs.decode_fp8`): code `S.EEEE.MMM`.

- Normal (`E ≠ 0`): `sig = 8 + M` (hidden bit 1), `exp = E − 1`.
- Subnormal (`E = 0`): `sig = M` (hidden bit 0), `exp = 0`.
- Both cases: `value = sig × 2^(exp − 9)`. Normal: `(8+M) × 2^(E−10)`. Subnormal: `M × 2^(−9)`.
- NaN: `E = 15, M = 7`.

The product of two lanes is `sig_a·sig_b × 2^(exp_a + exp_b − 18)`, so with `Wf = 8`, `T = exp_a + exp_b − 10`. This follows the plan's `−12 − 6` constant, written in the top-of-field convention.

The plan (§5.2) gives the decoder exponent as `exp[4:0]`; it is `exp[3:0]` here because `E − 1 ≤ 14`. The lane exponent sum is 5 bits (`≤ 28`).

## The eight steps

Let `WW = ALIGN_W + 2` (the window plus guard and round positions) and `K = clog2(N_PROD)`.

1. **Unpack the accumulator.** `biased_exp = acc[30:23]`. If it is 0 the accumulator is zero (the register never holds subnormals). Otherwise `F = {1, acc[22:0]}`, `T = biased_exp − 126`, `s = acc[31]`.
2. **Max exponent.** `T_max = max(T_i)` over terms with `F ≠ 0`. Zero terms (including NaN lanes, which are forced to zero) take the floor value −2^(EXP_W−1), so a zero product with a large exponent sum cannot pull the window up. FP8: a 32-input max tree, then one compare with the accumulator.
3. **Align.** Place each field at the top of a `WW`-bit window (`F << (WW − Wf)`), then shift right by `d_i = min(T_max − T_i, WW)`. Bits shifted below the window set that term's sticky bit. Implemented as `{top, WW zeros} >> d`: the upper half is the aligned term, and the OR of the lower half is its sticky. Clamping `d` at `WW` makes an over-long shift give zero plus sticky, with no special case.
4. **Negate.** Terms with `s = 1` become two's complement in `WW + 1` bits. Truncation happens on the magnitude, before negation (toward zero).
5. **Sum.** The 32 product terms go through `adder_tree` (`WW + 1 + K` bits), then one adder adds the accumulator term: `SUM_W = WW + 2 + K`. This matches the plan's carry bits: 6 extra for 33 terms, 1 for 2 terms.
6. **Sign-magnitude.** `|sum| ≤ (N_PROD + 1)(2^WW − 1) < 2^(WW + K + 1)`, so the magnitude fits `MAG_W = SUM_W − 1` bits. Result sign = sum sign.
7. **Normalize and round once, RNE.** `lzc` counts leading zeros of the `MAG_W`-bit magnitude, which is then shifted left by that count.
   - The top 24 bits are the significand (hidden bit + 23 fraction bits).
   - Next is the guard bit, then the round bit.
   - The remaining `REST_W = MAG_W − 26` bits OR together with every term's sticky bit to form sticky.
   - Round up when `guard & (round | sticky | sig[0])`. If that carries out of 24 bits, the significand becomes `0x800000` and the exponent increments.
   - Unbiased exponent: `e = T_max + K − lzc (+1 on round overflow)`.
8. **Pack** `{sign, e + 127, fraction}`. A zero magnitude packs as all zeros (+0), even if sticky is set. If `e > 127` the result saturates to the largest finite value `{sign, 0x7F7FFFFF}`; if `e < −126` it is +0. Only extreme MX scales reach either case (see "Range analysis").

## Widths

| Signal | FP8, `ALIGN_W=24` | FP8, `ALIGN_W=32` | MX, `ALIGN_W=24` | Why |
|--------|------|------|------|-----|
| Decoder `sig` / `exp` | 4 / 4 | 4 / 4 | – | hidden bit + 3 mantissa bits / `E − 1 ≤ 14` |
| Lane product / exponent sum | 8 / 5 | 8 / 5 | – | `15 × 15 = 225` / `14 + 14 = 28` |
| Exponent `T`, `T_max` | 10 signed | 10 signed | 10 signed | Accumulator `T ∈ [−125, 128]`; MX scale sums need about ±280; floor is −512 |
| Window `WW` | 26 | 34 | 26 | `ALIGN_W` + guard + round |
| Shift amount `d` | 5 | 6 | 5 | `clog2(WW + 1)`, clamped at `WW` |
| Signed aligned term | 27 | 35 | 27 | `WW` + sign |
| Product tree sum | 32 | 40 | 27 | + `K` (5 for 32 terms, 0 for 1) |
| Total sum `SUM_W` | 33 | 41 | 28 | + 1 for the accumulator term |
| Magnitude `MAG_W` | 32 | 40 | 27 | `SUM_W − 1` (step 6) |
| Leading-zero count | 6 | 6 | 5 | `clog2(MAG_W + 1)` |
| Bits folded into sticky (`REST_W`) | 6 | 14 | 1 | `MAG_W − 26` |
| Register | 32 | 32 | 32 | FP32 format: sign, 8-bit biased exponent, 23-bit fraction |

## Range analysis (why the register needs no subnormals or infinities)

- FP8 products lie in `[2⁻¹⁸, 448²]`. A product's `T` lies in `[−10, 18]`.
- One accumulation adds at most `32 × 448² < 2²³`. Overflowing FP32 (`2¹²⁸`) would take more than 2¹⁰⁴ accumulations.
- Every product is a multiple of 2⁻¹⁸, and a register value carries at most 24 significant bits. So after any cancellation the smallest nonzero result is at least about 2⁻⁴¹, far above FP32's normal minimum `2⁻¹²⁶`.
- MX units are different: an E8M0 scale spans 2^−127 to 2^127, so two extreme scales multiply to far outside FP32 (both 254: about 2^261; both 0: about 2^−247). The policy for all fused units is **saturate to ±max finite on overflow, +0 on underflow**. It costs two 10-bit compares. `tb/test_refs.py` asserts FP8 never reaches either case, and the MX tests exercise both. Real model scales sit near 127, nowhere near these limits.

## MX units (§5.6, §5.7)

Each MX unit forms an exact integer block sum `S` with the INT tree, then enters the fused stage with a single product term (`N_PROD = 1`).

| | MXINT8 | MXFP4 |
|---|---|---|
| Element value | `code / 64` (INT8, two's complement) | `mag / 2`, `mag ∈ {0,1,2,3,4,6,8,12}` (E2M1 decoded ×2) |
| Lane product | 16-bit signed, `mul_stage_int` (reused from INT8) | 9-bit signed, `|mag_a × mag_b| ≤ 144` |
| Block sum `S` | 21-bit signed, `|S| ≤ 32 × 128² = 2¹⁹` | 14-bit signed, `|S| ≤ 32 × 144 = 4608 < 2¹³` |
| Block value | `S × 2^(scale_a + scale_b − 266)` (two biases 254 + 2⁻⁶ per element) | `S × 2^(scale_a + scale_b − 256)` (two biases 254 + ×2 per magnitude) |
| `OFFSET` | 266 | 256 |

`|S| ≤ 2¹⁹` needs 20 magnitude bits, not the plan's 21. The plan's "block sum exactly at ±2²⁰" corner cannot occur; the tests use the real extremes, `+2¹⁹` from (−128)² and `−520192` from −128 × 127.

**Normalization (`align_stage_mxnorm`).** The MX front takes `|S|`, counts its leading zeros `lz` over `SIG_W` bits, and passes `mag = |S| << lz` with `T = scale_a + scale_b − OFFSET + SIG_W − lz`. This departs from the plan's "feed `|block_sum|`". A small block sum in an unnormalized 20-bit field would push the window up to 19 bits above its leading one and truncate that many bits of the accumulator. One leading-zero counter and one shifter avoid it. FP8 does not do this, because it would need 32 of each.

**E8M0 NaN (decision D4).** If either scale is 0xFF, the block term is forced to zero and `flag_nan` sets on that accumulation cycle, with the same sticky and clear rules as FP8.

**MXFP4 products.** The proposal describes the E2M1 product as a table. Each product magnitude is a function of 6 input bits (two 3-bit magnitude codes), so a table and a 4-bit × 4-bit multiply of the decoded magnitudes reduce to the same logic in synthesis. The RTL writes the multiply for readability.

## Precision notes (reported, not hidden)

- **Unnormalized products.** Product fields are not renormalized before alignment (that would cost 32 more leading-zero counters). A normal × normal product has at most one leading zero in its 8-bit field, so the window loses at most 1 bit. A product involving a subnormal can waste up to 7 bits at the top of the window, pushing low-order bits of the other terms into sticky.
- **Sticky with mixed signs.** Each term is truncated toward zero before negation, and stickies are ORed regardless of sign. When any bit is dropped (sticky from step 3), the result is within `1 ulp + 33 × 2^(T_max − WW)` of the exact sum, not always the correctly rounded value. When no bit is dropped, the result is exactly the round-to-nearest-even of the true sum. `tb/test_refs.py` checks both statements against exact rational arithmetic.
- The `ALIGN_W = 32` sweep row shows what the extra 8 bits of window cost in area and buy in fidelity.

## Policies

- **NaN (decisions D3, D4).** A lane with a NaN code on either input (FP8), or a block with a 0xFF scale (MX), contributes zero and does not affect `T_max`. `flag_nan` is sticky: it sets on any accumulation cycle (`en = 1`, `clear = 0`) that saw one, and it clears on `rst_n` or `clear`, the same as the accumulator.
- **Out of range.** Saturate to ±max finite on overflow, +0 on underflow (step 8).
- **Negative zero.** FP8 code `0x80` decodes to `sig = 0` and contributes nothing. A zero result is always +0.
- **Control.** `clear` wins over `en`, as in the INT units. Reset is synchronous.
