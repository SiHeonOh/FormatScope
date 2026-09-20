# FormatScope Glossary

Every technical term in the build plan, explained from scratch for the three of us. Read it in order the first time (each part builds on the one before), then use it as a lookup. Every entry says what the thing is, why it matters to FormatScope, and where possible shows a real number you can check by hand.

Parts: **1** how computers store numbers · **2** quantizing a neural network · **3** describing hardware (RTL) · **4** turning RTL into area and delay · **5** proving the hardware is right · **6** reading the results · **7** project words.

---

## Part 1 — How computers store numbers

### Bit, byte, binary
A **bit** is one wire that is either 0 or 1. A **byte** is 8 bits. With *n* bits you can make 2ⁿ different patterns: 4 bits give 16 patterns, 8 bits give 256. Everything below is a rule for what each pattern *means*. The pattern itself is called a **code**; the rule is called a **format**. The same 8-bit pattern `01110000` means 112 as an integer and 0.5 as an FP8-E4M3 float (you will be able to check that by the end of this part).

### Unsigned and signed integers, two's complement
**Unsigned**: read the bits as plain base-2. 8 bits cover 0 … 255.
**Signed, two's complement**: the top bit counts as −128 instead of +128. 8 bits then cover −128 … +127. To negate a number, flip every bit and add 1: 5 is `00000101`, flip → `11111010`, add 1 → `11111011` = −5. Check: `11111011` = −128 + 64 + 32 + 16 + 8 + 2 + 1 = −5. Two's complement is used everywhere because the same adder circuit adds positive and negative numbers correctly, with no special sign logic. Adding `11111011` (−5) and `00000111` (7) gives `1 00000010`; drop the carry-out and you have `00000010` = 2.

### INT8 and INT4
Two's complement integers with 8 or 4 bits. INT8 covers −128 … 127, INT4 covers −8 … 7. On their own they are just integers; they only become "the weight 0.031" when paired with a **scale** (Part 2). FormatScope deliberately never uses the most-negative code (−128 or −8) so that the usable range is symmetric, ±127 and ±7. That keeps the quantizer sign-symmetric and keeps the hardware comparison with FP8 (also symmetric) fair.

### Fixed-point
An integer plus a scale factor that everyone agrees on in advance and that is not stored. If the rule is "value = code × 2⁻⁶", then the INT8 code 64 means 1.0 and the code 51 means 0.796875. That is exactly how an MXINT8 element works (see Microscaling below): an 8-bit two's-complement integer with an implicit 2⁻⁶, so every element lies in −2.0 … +1.984375. The point of fixed-point is that the arithmetic is integer arithmetic; only your interpretation of the answer changes.

### Floating-point: sign, exponent, mantissa, hidden bit, bias
Scientific notation in binary. A float stores three fields: a **sign** bit, an **exponent** field (which power of two), and a **mantissa** field (the digits after the binary point). Value = (−1)^sign × 2^(exponent − bias) × 1.mantissa.
- The **bias** is a fixed offset so the exponent field can be stored as an unsigned number. FP8-E4M3 has bias 7, so an exponent field of 7 means 2⁰ and a field of 10 means 2³.
- The **hidden bit** is the leading "1." in 1.mantissa: every normal float starts with 1, so it is not stored. The **significand** is the mantissa *with* the hidden bit put back (1.mantissa). In hardware we usually keep the significand as an integer: for E4M3, 1.mmm becomes the 4-bit integer 8 … 15.

Worked example, FP8-E4M3 code `0 0011 101`: sign 0, exponent field 0011 = 3, mantissa 101 = 5/8. Value = 2^(3−7) × (1 + 5/8) = 1/16 × 1.625 = 0.1015625.

### FP32
The ordinary 32-bit float: 1 sign, 8 exponent bits (bias 127), 23 mantissa bits (24 significant bits with the hidden one). Precision is about 7 decimal digits; range is about 10⁻³⁸ … 10³⁸. It is what PyTorch computes in, what the FP32 baseline is measured in, and the format of the accumulator register in the FP8 and MX dot-product units. 1.0 in FP32 is `0 01111111 00000000000000000000000` (exponent field 127 = 2⁰).

### FP8-E4M3
An 8-bit float: 1 sign, 4 exponent bits (bias 7), 3 mantissa bits. Defined by the OCP FP8 specification. Properties to memorize:
- **Largest finite value 448** = code `0 1111 110`: 2^(15−7) × (1 + 6/8) = 256 × 1.75.
- **No infinities.** The pattern that would be infinity in other formats is a number here; that is where the extra range (448 instead of 240) comes from.
- **Exactly one NaN pattern per sign**: `S 1111 111`. Every other pattern is a number.
- **Smallest positive normal** `0 0001 000` = 2⁻⁶ ≈ 0.0156. Below that, subnormals (next entry) go down to 2⁻⁹ ≈ 0.00195.
- 256 codes in total, which is why we test the decoder exhaustively.

Its sibling **E5M2** (5 exponent bits, 2 mantissa bits, bias 15, has infinities and IEEE-style NaNs, max 57,344) trades precision for range and is not one of our five formats.

### Subnormal (denormal) numbers
What happens when the exponent field is all zeros. Instead of "1.mantissa × 2^(0−bias)" the rule becomes "0.mantissa × 2^(1−bias)": the hidden bit is 0 and the exponent is pinned at the smallest normal exponent. For E4M3 that is 0.mmm × 2⁻⁶, giving 2⁻⁹, 2⁻⁸, 3×2⁻⁹, … up to 7×2⁻⁹. Subnormals fill the gap between zero and the smallest normal so that tiny values shrink gradually instead of snapping to zero. Our FP8 decoder handles them with one rule: if E = 0, hidden bit = 0 and the exponent is treated as 1. The exhaustive 256-code test is what proves that rule right.

### NaN, infinity, overflow, saturation
- **NaN** (not a number) is a code reserved for "no valid value". E4M3 has one per sign; E8M0 uses 0xFF. Inference data never contains NaN, so FormatScope's hardware treats a NaN input as zero and raises a sticky `flag_nan` output; the policy is documented so a judge does not read it as a bug.
- **Infinity** is a code for "too big". E4M3 has none.
- **Overflow** is when a result is bigger than the format can hold. Two policies exist: wrap around (integers in two's complement do this silently) or **saturate** (clamp to the largest value). We saturate to ±448 when quantizing to FP8, and we document that the INT32 accumulator wraps, which never actually happens for ResNet-8 because the sums are far too small.

### E2M1 (FP4)
The 4-bit float used inside MXFP4: 1 sign, 2 exponent bits (bias 1), 1 mantissa bit. Only 16 codes, 8 magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6, each with a sign. The 0.5 is a subnormal (exponent field 00, mantissa 1 → 0.1₂ × 2⁰). There is no NaN and no infinity. The gaps grow with the value: between 4 and 6 there is nothing, which is why MXFP4 loses accuracy on values that happen to land at 5.

| code (S EE M) | value |
|---------------|-------|
| 0 00 0 | 0 |
| 0 00 1 | 0.5 |
| 0 01 0 | 1 |
| 0 01 1 | 1.5 |
| 0 10 0 | 2 |
| 0 10 1 | 3 |
| 0 11 0 | 4 |
| 0 11 1 | 6 |

### E8M0
An 8-bit number that is *only* an exponent: value = 2^(code − 127). Code 127 = 1, code 130 = 8, code 120 = 2⁻⁷, code 0 = 2⁻¹²⁷, code 254 = 2¹²⁷, and code 255 (0xFF) is NaN. There is no sign and no zero. It exists to be a **scale** that costs 8 bits and multiplies by a power of two, which in hardware is just a shift.

### Dynamic range and precision
Two different things a format can be good at.
- **Dynamic range**: the ratio between the biggest and smallest value it can represent. FP8-E4M3 spans 2⁻⁹ … 448, about 2¹⁸. INT8 with one scale spans 1 … 127, about 2⁷.
- **Precision**: how fine the steps are. INT8 has *uniform* steps (every neighbor is 1 code apart). Floats have *relative* steps: E4M3 has 8 codes between 1 and 2 (step 0.125) but also 8 codes between 256 and 448 (step 32). Picture a ruler with evenly spaced ticks versus a ruler whose ticks get farther apart as the numbers grow.
Neural-network weights are mostly tiny with a few large outliers, which is why floats and block scaling do well on them and why INT4 with one scale per tensor does badly: the outlier stretches the scale and the tiny values all round to zero.

### Rounding: round-to-nearest-even (RNE)
When a value falls between two representable numbers, take the nearer one; when it is exactly halfway, take the one whose last bit is even. 2.5 → 2, 3.5 → 4, 2.6 → 3. Ordinary "round half up" always rounds ties upward, which adds a tiny positive bias every time; over thousands of accumulations that bias shows up as error. RNE cancels it. FormatScope uses RNE everywhere: in the Python quantizer, in the NumPy reference, and in the hardware's single rounding step, so all three agree bit for bit.

### Guard, round, and sticky bits
When you shift a number right to line it up with a bigger one (see Alignment in Part 3), bits fall off the end. To round correctly afterward you need to remember three things about what fell off: the first bit (**guard**), the second bit (**round**), and whether *anything* nonzero fell off beyond those (**sticky**, an OR of all the rest). With those three bits, RNE gives the same answer it would have given with infinite precision. The sticky bit is the one everybody forgets, and forgetting it produces results that are off by exactly one unit in the last place on some inputs.

### Microscaling (MX), block, shared scale, MXINT8, MXFP4
The OCP MX v1.0 idea: instead of one scale for a whole tensor, give every **block** of 32 consecutive elements its own **shared scale**, stored as one E8M0 (a power of two). Each element is then a small number relative to that scale: an INT8 with implicit 2⁻⁶ (**MXINT8**) or an E2M1 (**MXFP4**). Overhead: 8 bits per 32 elements, a quarter of a bit per element.

Value of element *i* in a block = 2^(X − 127) × element_i, where X is the block's E8M0 code.

How the scale is chosen: `shared_exp = floor(log2(max |v_i|)) − emax_elem`, where emax_elem is the exponent of the largest normal element value (2 for E2M1, whose largest value is 6 = 1.5 × 2²; 0 for the INT8 element, whose largest value is just under 2 = 1.984 × 2⁰). Then every element is divided by 2^shared_exp, rounded to nearest-even in the element format, and clamped to the element maximum.

Worked example, block values [3.2, −7.9, 0.5, 0, …]: max |v| = 7.9, floor(log2 7.9) = 2.
- MXFP4: shared_exp = 2 − 2 = 0, scale = 1. Elements: 3.2 → 3, 0.5 → 0.5, 0 → 0, and −7.9 → −6 because 7.9 exceeds the E2M1 maximum of 6 and gets **clamped**. That clamp is a real property of the spec's rule, not a bug.
- MXINT8: shared_exp = 2 − 0 = 2, scale = 4. 3.2/4 = 0.8 → code round(0.8 × 64) = 51 → value 51/64 × 4 = 3.1875. −7.9/4 = −1.975 → code −126 → −7.875. Errors of 0.0125 and 0.025.

Edge cases the code must handle: an all-zero block (scale code 127, all elements 0), and a block that is not full because the reduction length is not a multiple of 32 (pad with zeros; padding adds 0 to a dot product).

### OCP
The Open Compute Project, an industry consortium (Microsoft, NVIDIA, AMD, Intel, Meta, and others) that published the FP8 and MX format specifications we implement. When the plan says "OCP FP8" or "OCP MX v1.0", it means those public documents are the ground truth for every code's meaning.

---

## Part 2 — Quantizing a neural network (the accuracy side)

### Layer, weights, activations
A neural network is a stack of layers. Each **weight** is a number learned during training and fixed afterward. An **activation** is a number flowing through the network for one particular input image. A convolution or linear layer computes many **dot products** between its weights and its input activations. Quantizing a layer means storing its weights *and* its activations in a small format; FormatScope quantizes both, in every layer, including the first and last, which are the ones people usually leave in high precision because they are the most sensitive.

### Dot product, MAC, reduction dimension, accumulator
A dot product multiplies two lists element by element and adds everything up: (1, 2, 3)·(4, 5, 6) = 4 + 10 + 18 = 32. One multiply-then-add is a **MAC** (multiply-accumulate). The length of the lists is the **reduction dimension**, because the sum reduces many numbers to one. The running total lives in the **accumulator**. A 3×3 convolution over 16 input channels has a reduction dimension of 3 × 3 × 16 = 144, so each output value is a 144-term dot product; our DP32 computes 32 of those terms per clock cycle and needs 5 cycles (with the last block padded from 16 terms to 32) to finish one output.

### Quantization, scale, symmetric vs asymmetric, zero point
**Quantization** maps real numbers onto a small set of codes. The **scale** *s* says how big one code step is: value ≈ code × s. **Symmetric** quantization uses the same range on both sides of zero (−127 s … +127 s) and needs only the scale. **Asymmetric** quantization adds a **zero point** to shift the range (useful for values that are all positive, such as after a ReLU) at the cost of extra hardware. FormatScope is symmetric throughout, so there is no zero point anywhere, and the hardware comparison stays clean.

For INT8 with symmetric per-tensor scaling: s = max|x| / 127, code = clamp(rne(x / s), −127, 127), reconstructed value = code × s. If a weight tensor's largest magnitude is 0.254, s = 0.002, and a weight of 0.0311 becomes code 16 → 0.032.

### Scale granularity: per-tensor, per-channel, per-block
How many numbers share one scale.
- **Per-tensor**: one scale for the whole weight tensor or the whole activation tensor. Cheapest, coarsest, worst for INT4.
- **Per-output-channel**: one scale per output channel of the weight tensor (a conv with 32 output channels has 32 scales). Each channel's weights get their own ruler, so a big weight in channel 7 does not crush channel 3. FormatScope uses this for INT and FP8 weights, and per-tensor for activations.
- **Per-block**: one scale per 32 elements along the reduction dimension. That is MX, and it is the finest of the three.

Why the hardware comparison excludes the per-channel scale: after the dot product finishes, every format has to multiply the accumulator by (weight scale × activation scale) and round into the next layer's format. That step, **requantization**, is identical work for every format, so it is left out of every unit uniformly; the DP32 areas compare only the arithmetic that differs between formats.

### Requantization
The stage after the accumulator that converts the wide accumulated value back into a small format for the next layer: multiply by the combined scale, round, clamp. Every format needs it, which is why the plan can exclude it from the hardware comparison without favoring anyone.

### Calibration, clipping, percentile
Weights are known in advance, so their scales are computed directly. Activations depend on the input image, so their scale has to be estimated by running some images through the network and watching the values: that is **calibration**. FormatScope uses 512 training images. Choosing the scale from the very largest value seen (the max) lets one freak outlier stretch the ruler for everyone; choosing it from the 99.99th **percentile** of |x| **clips** those outliers (they saturate) in exchange for finer steps for the other 99.99%. The plan records which statistic was used in the results CSV so the number is reproducible.

### Fake quantization (quant-dequant)
PyTorch does not compute in INT8. To measure what a format does to accuracy, we insert a function that quantizes a tensor to the format and immediately dequantizes it back to FP32: x → code → code × s. The result is an FP32 tensor that can only take the values the format can represent, so the network experiences the exact rounding error of the format while everything else stays FP32. Plotted, it is a staircase: inputs along a step all produce the same output. The wrappers `QConv2d` and `QLinear` do this to their weights (once) and to their input activations (every forward pass).

### PTQ and QAT
**Post-training quantization (PTQ)**: take the trained FP32 model, insert fake quantization, calibrate, measure. No training. Fast, and it shows the raw damage each format does.
**Quantization-aware training (QAT)**: keep the fake quantization in place and train for a few more epochs, so the weights learn to sit where the format can represent them. FormatScope does 5 epochs. QAT recovers most of INT4's and MXFP4's loss, which is what hypothesis H2 is about.

### Straight-through estimator (STE)
Training needs gradients, and the derivative of a staircase is zero everywhere (and undefined at the steps), so nothing would learn. The STE is the trick of pretending, during the backward pass only, that the rounding function was the identity (derivative 1) inside the clamp range and 0 outside. The forward pass still uses the real rounded values. It is a hack with no rigorous justification that works remarkably well and is what every QAT paper uses.

### Batch normalization and BN folding
**Batch norm** is a layer after each conv that rescales and shifts each channel (x → γ (x − μ)/σ + β) using statistics learned in training. At inference time γ, μ, σ, β are constants, so the whole operation can be **folded** into the preceding conv: multiply that conv's weights by γ/σ per channel and adjust its bias. Real hardware runs the folded conv, so the weights we quantize must be the folded ones; quantizing unfolded weights would measure the wrong tensors. The plan's check is simple: with quantization switched off, the folded model must reproduce the FP32 accuracy exactly.

### ResNet-8, residual block, shortcut
Our model. A **ResNet** is a stack of **residual blocks**; each block computes two convolutions and then adds its own input back (the **shortcut**), which makes deep networks trainable. "ResNet-8" counts the weighted layers: one stem conv, three blocks of two convs, and the final linear layer, 8 in total. At the two places where the channel count doubles (16 → 32 → 64), the shortcut cannot add tensors of different shapes, so it downsamples and zero-pads the channels (the parameter-free "option A") instead of adding another conv. All 8 weighted layers are quantized.

### CIFAR-10, top-1 accuracy, test set, epoch
**CIFAR-10** is 60,000 tiny 32×32 color images in 10 classes (airplane, cat, truck, …): 50,000 for training, 10,000 for testing. **Top-1 accuracy** is the fraction of test images whose single most likely predicted class is right; 10% is chance. An **epoch** is one pass over all 50,000 training images. The plan's expected FP32 ceiling is 85–88%; the reason the project uses CIFAR-10 rather than MNIST is that MNIST saturates at 99% in every format and would show no differences.

### Training recipe words
- **SGD** (stochastic gradient descent): update the weights a little in the direction that reduces the loss, using one **batch** (128 images) at a time.
- **Learning rate**: the size of each update. **Cosine annealing** shrinks it smoothly from 0.1 to 0 over the run, following a cosine curve.
- **Momentum** (0.9): keep a running average of update directions so the path is smoother.
- **Weight decay** (5e-4): a small pull of every weight toward zero, which discourages huge weights and helps generalization.
- **Augmentation**: random crops (after padding by 4 pixels) and horizontal flips of the training images, so the network sees slightly different versions each epoch.
- **Seed**: the starting value for every random choice; fixing it at 0 makes a run reproducible.

### Ablation
An experiment that removes one ingredient to see what it was contributing. MXFP4 differs from plain INT4 in two ways at once: block scaling and the E2M1 encoding. The configuration `int4_b32` (INT4 elements with a block-32 scale) has only the first ingredient, so comparing INT4 vs INT4-b32 vs MXFP4 says how much each ingredient is worth.

### Numerical fidelity (the PyTorch-vs-hardware cross-check)
PyTorch fake quantization adds up dequantized products in FP32, rounding after every add. The FP8 and MX hardware adds 32 products exactly and rounds once per block. Those two procedures give slightly different answers, so the accuracy numbers measured in PyTorch are not *exactly* what the hardware would produce. Rather than assume the gap is negligible, the plan pushes one real layer's dot products through both procedures and reports the measured difference. For INT4 and INT8 the two are exactly equal, because integer sums of this size are exact in FP32.

---

## Part 3 — Describing hardware (RTL)

### RTL and Verilog
**Register-transfer level** is the style of hardware description where you say what values registers hold and what logic sits between them, and a synthesis tool turns that into gates. **Verilog** is the language (we use the 2005 dialect, which every open tool reads). It looks like code but describes wires and gates that all exist at once; there is no "next line runs after this one". Two lines in a module are two circuits operating in parallel.

### Module, port, wire, reg, parameter, localparam
- **Module**: a hardware block with a name; `dp32_int8` is one.
- **Port**: a module's input or output. The plan's common interface has `clk`, `rst_n`, `en`, `clear`, `a_flat`, `b_flat`, `scale_a`, `scale_b`, `acc_out`, `flag_nan`.
- **wire**: a connection whose value is driven continuously by logic.
- **reg**: a variable assigned inside an `always` block. A `reg` assigned on a clock edge becomes a real register; a `reg` assigned in a combinational block is just a named wire.
- **parameter**: a compile-time constant you can override per instance (`W = 8`, `ALIGN_W = 24`). **localparam** is the same but cannot be overridden, used for derived widths.

### Combinational vs sequential logic, clock, register, reset
**Combinational** logic has no memory: outputs are a pure function of current inputs (adders, multipliers, muxes). In Verilog: `always @*` or `assign`. **Sequential** logic has state that updates only on a **clock** edge: a **register** (flip-flop) captures its input at each rising edge and holds it until the next. In Verilog: `always @(posedge clk)`. **Reset** (`rst_n`, active low) forces registers to a known value at start-up. The DP32 is one big combinational cloud (multiply, add, normalize) feeding one register (the accumulator); that structure is why its delay is the delay of the cloud.

### Latch inference
The classic Verilog bug. If a combinational `always @*` block leaves an output unassigned on some path (for example an `if` with no `else`), the tool must make the output *remember* its old value, and it builds a **latch** to do so. Latches are slow, break timing analysis, and were never intended. Yosys prints "inferring latch" when it happens. The cure is mechanical: assign every output a default at the top of the block.

### Bit width and bit growth, sign extension
Adding two *n*-bit numbers can need *n* + 1 bits (127 + 127 = 254 does not fit in 8 bits). Multiplying two *n*-bit numbers needs 2*n* bits. Adding 2ᵏ numbers grows the width by *k*. So in the INT8 unit: 8-bit × 8-bit → 16-bit products, and 32 of them summed (2⁵) → 21 bits. In the INT4 unit: 8-bit products, 13-bit sum. **Sign extension** is how you widen a signed number without changing its value: copy the sign bit into the new top bits (−5 as 8 bits is `11111011`, as 12 bits is `111111111011`). Getting a width wrong by one bit gives results that are right for small inputs and silently wrong for large ones, which is why the tests include all-max vectors.

### Multiplier, adder tree, accumulator register
The three parts of every DP32. Thirty-two **multipliers** work in parallel, one per **lane**. A balanced **adder tree** adds their 32 outputs in pairs: 32 → 16 → 8 → 4 → 2 → 1, five levels deep, so the sum is ready after five additions of delay rather than thirty-one. The **accumulator register** holds the running total across clock cycles (`clear` zeroes it, `en` lets it accumulate). "DP32" = dot product, 32 lanes.

### Single MAC vs DP32
A **single MAC** is one multiplier and one adder. Comparing formats on a single MAC is common but hides the cost that MX formats put *between* lanes: the shared-scale logic and the block-level exponent handling only appear when you build all 32 lanes together. That is why FormatScope's unit of comparison is the 32-lane DP32, and why the accumulator is counted as part of each format's cost.

### The fused stage: align, add, normalize, round once
The FP8 (and MX) datapath. Adding floats normally means: line up exponents, add, round; do that 32 times and you round 32 times, accumulating error and hardware. The **fused** design does it once: bring all 32 products and the current accumulator to a common exponent, add them all exactly in an integer tree, then normalize and round a single time into the FP32-format register. Fewer roundings (more accurate) and one shared normalize/round circuit instead of 32 (cheaper). Hypothesis H1 says this amortization is why the FP8 penalty over INT8 should be smaller than the 183% single-MAC estimate.

### Alignment, shift, window width (`ALIGN_W`)
To add numbers with different exponents you shift the smaller ones right until their exponents match the largest (that is **alignment**). 1.5 × 2⁵ + 1.0 × 2² = 1.5 × 2⁵ + 0.125 × 2⁵ = 1.625 × 2⁵. In hardware the shifted significands land in a fixed-width **window** of `ALIGN_W` bits (plus guard/round/sticky). A term whose exponent is far below the maximum gets shifted almost entirely out of the window, contributing only to the sticky bit. A wider window keeps more of the small terms (more exact, more area, more delay); that is why the plan sweeps `ALIGN_W` = 24 and 32 rather than defend one choice. 24 matches FP32's own significand width, so it is the smallest setting that can keep the largest term exact.

### Sticky bit (inside the fused stage)
One OR of every bit any term lost while being shifted into the window. It cannot change the sum's magnitude noticeably, but it decides ties in the final RNE step correctly. Leaving it out is the most common way a "bit-exact" design ends up one unit off.

### Normalization and leading-zero count (LZC)
After the big integer add, the result is some integer like `000010110…` that needs to be turned back into "1.xxx × 2^e" form. A **leading-zero counter** finds how many zeros precede the first 1; shifting left by that count and subtracting it from the exponent **normalizes** the value. Then the guard/round/sticky bits decide the single rounding, and the sign, exponent, and 23 mantissa bits are packed into the FP32 register.

### Sign-magnitude and two's complement inside the adder
Floats carry a separate sign bit (**sign-magnitude**), but an adder tree wants two's complement. So each aligned term is negated into two's complement if its sign is set, the tree adds them, and the result is converted back to a sign and a magnitude before normalization. Each conversion is a row of inverters and an increment; the tests include exact-cancellation vectors (a + (−a)) because that is where sign handling breaks.

### Decoder
The small combinational block that turns a code into its fields: sign, exponent, significand (with hidden bit), NaN flag. Every format has one (for INT it is the identity). Because a decoder has at most 256 inputs, it is tested against every single code, and the Python decoder in `quant/formats.py` is the reference. If those 256 agree, the two lanes of the project agree on what every code means.

### `$signed`, `generate`, `hierarchy -check`
Three Verilog/Yosys things that decide whether the RTL week goes well.
- **`$signed(x)`** tells Verilog to treat a vector as two's complement. Without it, `a * b` is an unsigned multiply and negative inputs produce garbage that only shows on negative test vectors.
- **`generate`** loops instantiate hardware repeatedly (32 lanes, 5 tree levels) from one description, so one fix applies everywhere.
- **`hierarchy -check`** in Yosys verifies every module and port is connected; an unconnected output is how a whole multiplier gets optimized away and "area" becomes 0.

---

## Part 4 — Turning RTL into area and delay (synthesis and timing)

### Logic synthesis, Yosys
**Synthesis** converts RTL into a network of real gates from a specific cell library, optimizing as it goes. **Yosys** is the open-source synthesis tool in the OSS CAD Suite. The output is a netlist and a report of what it used.

### Netlist
A text file listing every gate instance (cells) and every wire between them. Nothing about position or size on the chip yet; that is layout, which we do not do. The netlist is what OpenSTA reads to compute delay, and it is what we commit under `results/netlists/` as evidence.

### Standard cell, cell library, liberty file (`.lib`)
A **standard cell** is a pre-designed, pre-characterized gate: NAND2, NOR3, a full adder, a D flip-flop (DFF), each a fixed-height rectangle. A **cell library** is a family of them for one process. The **liberty file** is the library's data sheet in machine-readable form: for every cell, its **area** in µm² and its **delay** as a function of input slope and output load. In sky130's high-density library a NAND2 is about 3.75 µm² and a DFF about 20 µm² (open the `.lib` and search for `area :` to check). When Yosys reports "Chip area", it has added up the liberty areas of every cell it used; when OpenSTA reports a path delay, it has added up liberty delays along that path.

### PDK, sky130, ASAP7, process node
A **PDK** (process design kit) is everything a foundry gives designers about one manufacturing process. **sky130** is SkyWater's open 130 nm process, real silicon you can actually tape out, with the `sky130_fd_sc_hd` ("high density") cell library we use. **ASAP7** is a *predictive* 7 nm PDK from Arizona State: not a real foundry process, but a realistic model of one, used for research. The **node** (130 nm, 7 nm) is the nominal feature size; cells in a 7 nm library are dozens of times smaller and several times faster than in 130 nm. That is why the plan says absolute µm² do not transfer between nodes but the *ratios* between formats mostly do, and why hypothesis H3 checks that the ranking survives the switch.

### Corner (tt, 25 °C, 1.80 V)
Chips vary with manufacturing luck (**process**), supply **voltage**, and **temperature**, so libraries are characterized at several **corners**. "tt" is typical-typical (typical NMOS, typical PMOS), at 25 °C and 1.80 V: the middle-of-the-road case. Using one corner for every format keeps the comparison fair; a sign-off flow would also check slow and fast corners.

### Yosys passes: `synth`, `-flatten`, `dfflibmap`, `abc`, `opt_clean`, `stat`
The script every unit runs, one line at a time:
- `read_liberty -lib` loads the cell library so later passes know what cells exist.
- `read_verilog -sv` parses the RTL. `hierarchy -check -top` picks the top module and checks connections.
- `synth -top X -flatten` runs Yosys's generic synthesis (coarse optimization, then conversion to simple gates) and **flattens** the module hierarchy into one big module so optimization can cross module boundaries.
- `dfflibmap -liberty` replaces generic flip-flops with the library's real DFF cells.
- `abc -liberty` hands the combinational logic to **ABC**, which does technology mapping (next entry).
- `opt_clean` removes unused wires and cells.
- `stat -liberty` prints the cell count and the total area.

### ABC, technology mapping, the `-D` delay target
**ABC** is the logic optimizer inside Yosys that maps generic gates onto the library's actual cells while minimizing what you ask for. With no target it minimizes area. With `-D <picoseconds>` it tries to make every path faster than the target, which usually costs area (bigger cells, duplicated logic, shallower trees). That is the knob behind "area at matched delay": each unit is synthesized unconstrained and at two shared targets, and we read off how much area each format needs to reach the same speed. ABC's choices also change a little with the target and script, which is the "spread" the plan reports; the unconstrained run must never carry a `-D`, or the area numbers stop being comparable.

### Area (µm²) and cell count
The sum of the liberty areas of every cell in the netlist, in square micrometers (a micrometer is a millionth of a meter; a human hair is about 70 µm across). It excludes wiring and empty space, so it is a lower bound on real chip area, but it is the same lower bound for every format, which is what a comparison needs. Cell count is the raw number of gates and is reported alongside as a sanity check.

### Hierarchical vs flat synthesis, stage breakdown
Flat synthesis merges everything and gets the smallest total; it also loses track of which part of the design each cell came from. **Hierarchical** synthesis (no `-flatten`) keeps the module boundaries, so `stat` can report the multiply stage, the align stage, the tree, and the normalize/accumulate stage separately: that is the **stage breakdown** the plan uses to say *where* FP8's extra cost sits. Because boundaries block some optimization, the hierarchical total is a little larger than the flat total, and the plan reports both.

### Delay, propagation delay, critical path
Every gate takes time to switch (tens to hundreds of picoseconds in sky130). The **propagation delay** of a path is the sum along it. Among all paths from the inputs to the accumulator register, the slowest one is the **critical path**; it sets how fast the unit can be clocked. In a DP32 it runs through one multiplier, five adder levels, and (for FP8) the normalize-and-round logic.

### Clock period, setup time, slack, WNS, TNS
The **clock period** is how often the register captures. The register needs its input stable a little before the edge, the **setup time**. **Slack** = period − path delay − setup. Positive slack means the path makes it ("MET"); negative means it does not ("VIOLATED"). Example: period 10 ns, path 6.8 ns, setup 0.1 ns → slack 3.1 ns. **WNS** (worst negative slack) is the slack of the worst path; **TNS** (total negative slack) adds up every failing path. Our STA script reports both.

### Static timing analysis (STA), OpenSTA
Computing every path's delay from the netlist and the liberty file *without* simulating any inputs (hence "static"). It is exhaustive and fast. **OpenSTA** is the open-source timing analyzer (the same engine inside OpenROAD). The plan's script: load the library, load the netlist, define a clock, set input and output delays to zero, then `report_checks` for the worst path and `report_wns` / `report_tns`. The number we extract is the worst path's **data arrival time**, which is the combinational delay from the input ports to the accumulator register.

### Area at matched delay
Any adder tree can be made smaller by making it slower, so "unit A is smaller than unit B" is meaningless unless both run at the same speed. Synthesizing every format to the same delay targets and comparing area *there* removes that loophole. It is the proposal's central methodological choice and the reason OpenSTA is on the critical path of the project.

### Logic depth (`ltp`) vs timed delay
Yosys can print the **longest topological path**: the largest number of gates in series between a register and the next. It is a structural count, not a time; a path of 40 fast inverters can be quicker than a path of 15 slow XORs. The earlier draft of the proposal reported logic depth; the final proposal reports timed delay from OpenSTA instead, which is what the plan builds.

### Pre-layout, parasitics, place and route
We stop at synthesis. **Place and route** would put each cell at a location and draw the wires; the wires add resistance and capacitance (**parasitics**) that slow every path and add area. Our numbers are therefore **pre-layout**: cell area and cell delay only. They are honest for ranking formats against each other on the same design and they are not sign-off numbers, and the README says so.

### Power (out of scope)
Energy per operation depends on how often each wire switches (activity) and on wire capacitance (layout), neither of which we have. A pre-layout power number would be indicative at best, so the proposal excludes it rather than report something weak.

---

## Part 5 — Proving the hardware is right (verification)

### Simulation, Icarus Verilog, DUT, testbench
**Simulation** runs the RTL on a computer, cycle by cycle, with inputs you choose. **Icarus Verilog** (`iverilog`) is the open-source simulator we use. The module being tested is the **DUT** (device under test); the code that drives its inputs and checks its outputs is the **testbench**. In FormatScope the testbench is Python.

### cocotb
A framework that lets you write testbenches in Python: it starts the simulator, and your Python coroutine sets signal values (`dut.a_flat.value = …`), waits for clock edges (`await RisingEdge(dut.clk)`), and reads results (`int(dut.acc_out.value)`). Version 2.0 changed the API (assignment with `=` instead of `<=`, `unit="ns"` instead of `units=`, `cocotb_tools.runner` for building), which is why the plan pins `cocotb~=2.0` and shows the exact idioms. The **runner** compiles the Verilog and launches the simulator from `pytest`.

### Reference model, bit-exact
A **reference model** (or golden model) is an independent implementation of what the hardware should compute, in a language where mistakes are easy to see; ours is NumPy in `tb/refs.py`. A test drives the same inputs into the DUT and the reference and requires the outputs to be **bit-exact**: identical in every bit, not "close". Bit-exactness is possible here because the reference mirrors the hardware's exact shifts, widths, and single rounding, and it is what lets the project claim any mismatch is a bug rather than a rounding difference.

### Exhaustive testing, random vectors, seeds, corner cases
- **Exhaustive**: try every possible input. Feasible for decoders (256 or 16 codes), impossible for a DP32 (2⁵¹² inputs).
- **Random vectors**: 10,000 inputs drawn from a random generator with a fixed **seed** (0), so the same 10,000 come back on every run and a failure can be reproduced.
- **Corner cases**: hand-chosen inputs where bugs live: all zeros, all maxima, alternating signs, subnormals, NaN codes, extreme scales, exact cancellation, an accumulator much larger than the new products and vice versa.
Random vectors catch the bugs you did not think of; corners catch the ones you did.

### Waveform, VCD, GTKWave
A **waveform** is the recorded value of every signal over time. The simulator can dump it to a **VCD** file, and **GTKWave** displays it so you can see, cycle by cycle, where a value went wrong. Dumping slows simulation, so the plan turns it on only with an environment variable when debugging.

### pytest, continuous integration (CI), smoke test
**pytest** is the Python test runner; `pytest -q` prints a green summary. **CI** is a server (GitHub Actions) that runs the tests automatically on every pull request and blocks merging when they fail; it is how "no RTL merges without its cocotb test green" is enforced rather than remembered. A **smoke test** is a tiny check that a tool chain works at all (a 4-bit adder synthesizes, a trivial cocotb test passes, one epoch trains), run before investing in real code.

---

## Part 6 — Reading the results

### Pareto front, dominated point, knee
Plot each format as a point: area on the x-axis, accuracy on the y-axis. A point is **dominated** if some other point is at least as good on both axes (smaller-or-equal area *and* higher-or-equal accuracy); nobody would ever choose it. The points that are not dominated form the **Pareto front**: the menu of sensible choices, each trading some area for some accuracy. The **knee** is where the front bends, the point beyond which each extra µm² buys noticeably less accuracy. The `formatscope` tool draws the front as a step line and computes it per delay target.

### Constraint-based recommendation
Rather than declaring one "best" format (which depends on what you care about), the tool answers a question with a constraint: `--min-acc 0.87` returns the smallest-area format that reaches 87% top-1; `--area-budget 30000` returns the most accurate format that fits in 30,000 µm². Both are one pass over the front.

### Hypothesis, "reported either way", the 183% figure
A **hypothesis** is a prediction written down before the measurement. Reporting it "either way" means the result is a deliverable whether it confirms or refutes the prediction. The **183%** comes from a Qualcomm AI Research paper (arXiv 2303.17951) that estimated an FP8-E4M3 multiply-accumulate with a 32-bit accumulator needs 183% more gates than an INT8 one. It is a single-MAC gate-count estimate; H1 predicts our measured DP32 area ratio will be above 1 but below 2.83, and the stage breakdown will show why.

### Spread, variance, ranking vs absolute numbers, perturbation runs
Synthesis results move a little when the script or target changes slightly, so a single area number carries false precision. The plan runs each unit five times at slightly different delay targets (±1%, ±2%) and reports the min, median, and max: the **spread**. The claim the project makes is the **ranking** of formats and the ratios between them, which are stable, not the absolute µm², which are specific to sky130, this script, and this tool version.

### Frontier at each delay target
Because area depends on the delay target, there is not one frontier but one per target: unconstrained, T1, and T2. A format can move relative to another between targets (a design that is small when slow may grow faster than its neighbor when pushed), which is itself a result.

---

## Part 7 — Project words

### Gate (G0 … G6), floor, cut order, stretch
A **gate** is an end-of-day checkpoint with a yes-or-no test ("INT8 verified with area and delay recorded"); the next day's plan assumes it passed. The **floor** is the smallest result we would still be glad to submit. The **cut order** is the pre-agreed sequence of things to drop if a gate slips, taken from the proposal: ASAP7 first, then the third delay target, then QAT on the MX formats. **Stretch** items are the ones in that list; they are in scope and scheduled, and they are what gets cut if something must be.

### Lane, owner
A **lane** is one of the five parallel workstreams (quantization, RTL, verification, synthesis and timing, the tool). Each has one **owner** who reviews, runs, and commits that lane's code, even when the first draft was written by someone (or something) else.

### Makefile target, venv, `versions.lock`, reproducibility
A **Makefile target** is a named command (`make test`) that runs a longer recipe, so a judge can regenerate a result without reading our scripts. A **venv** is an isolated Python environment so package versions do not collide with anything else on the machine. **`versions.lock`** records the exact tool versions (OSS CAD Suite date, PDK hash, pip freeze) so anyone can rebuild the same environment. Together they are what "every number regenerates with one command" means in practice.

### Repository, branch, pull request, tag, CI green
The **repository** is the project's version-controlled folder on GitHub. A **branch** is a parallel line of work (`sh/fp8-dp32`); a **pull request** proposes merging it into `main` and is where the CI runs and a teammate skims the change. A **tag** (`v1.0`) is a permanent name for one exact commit, so the results we submit point at code that cannot drift. "**CI green**" means every automated test passed on that pull request.

### WSL2, Docker, OSS CAD Suite
**WSL2** is Windows' built-in Linux; the EDA tools are Linux programs, so on a Windows machine they run inside it (and the repo must live on the Linux side of it for speed). **Docker** packages a program with everything it needs into a container, which is the least painful way to get OpenSTA on a machine where building it from source fights the system's Tcl. The **OSS CAD Suite** is a single download from YosysHQ containing Yosys, ABC, Icarus Verilog, Verilator, GTKWave, and cocotb, all built to work together; it does not contain OpenSTA.
