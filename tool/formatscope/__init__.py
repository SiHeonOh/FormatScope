"""FormatScope: join quantized accuracy with DP32 area and delay (build-plan.md S8)."""

# int4_b32 (INT4 elements, block-32 power-of-two scale) has a unit too:
# rtl/int4_b32/, the MXINT8 design at 4 bits. Until its area rows exist it shows
# as accuracy-only, exactly as before.
HARDWARE_FORMATS = ["int4", "int8", "fp8e4m3", "mxint8", "mxfp4", "int4_b32"]
ALL_FORMATS = list(HARDWARE_FORMATS)
# Each unit at its own fastest mapping and at the shared clock T1. There is no
# T2: no period tighter than T1 constrains both the INT and the fused units
# (build-plan.md D7).
TARGETS = ["unc", "t1"]

# Storage cost per element, amortizing the 8-bit E8M0 shared scale over the
# 32-element block for the block-scaled formats (S1.3).
BITS_PER_NUMBER = {
    "fp32": 32.0,
    "int4": 4.0,
    "int8": 8.0,
    "fp8e4m3": 8.0,
    "mxint8": 8.0 + 8.0 / 32,
    "mxfp4": 4.0 + 8.0 / 32,
    "int4_b32": 4.0 + 8.0 / 32,
}
