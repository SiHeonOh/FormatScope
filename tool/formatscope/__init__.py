"""FormatScope: join quantized accuracy with DP32 area and delay (build-plan.md S8)."""

HARDWARE_FORMATS = ["int4", "int8", "fp8e4m3", "mxint8", "mxfp4"]
ALL_FORMATS = HARDWARE_FORMATS + ["int4_b32"]
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
