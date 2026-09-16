"""FormatScope: join quantized accuracy with DP32 area and delay (build-plan.md S8)."""

HARDWARE_FORMATS = ["int4", "int8", "fp8e4m3", "mxint8", "mxfp4"]
ALL_FORMATS = HARDWARE_FORMATS + ["int4_b32"]
TARGETS = ["unc", "t1", "t2"]
