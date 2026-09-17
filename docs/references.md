# References

Format definitions

- Open Compute Project, *OCP 8-bit Floating Point Specification (OFP8)*, revision 1.0, 2023. Defines E4M3 (bias 7, max 448, one NaN code, no infinities) and E5M2. `quant/formats.py` and `rtl/common/decode_fp8e4m3.v` implement E4M3 as specified; overflow saturates to ±448 (decision D2).
- Open Compute Project, *OCP Microscaling Formats (MX) Specification*, version 1.0, 2023. Defines the block-32 shared E8M0 scale, the MXINT8 element (INT8 with an implicit 2⁻⁶), and the MXFP4 element (E2M1). `quant/formats.py` implements the shared-exponent rule; `rtl/common/decode_e8m0.v` and `decode_e2m1.v` decode the codes exhaustively tested in `tb/test_decode.py`.

Papers

- M. van Baalen et al., *FP8 versus INT8 for efficient deep learning inference*, arXiv:2303.17951, 2023. Source of the 50–180% single-MAC hardware-cost estimate that hypothesis H1 tests against a fused 32-lane dot product.
- P. Micikevicius et al., *FP8 Formats for Deep Learning*, arXiv:2209.05433, 2022. The E4M3/E5M2 proposal that the OCP OFP8 specification standardized.
- B. Darvish Rouhani et al., *Microscaling Data Formats for Deep Learning*, arXiv:2310.10537, 2023. Accuracy results for MXINT8 and MXFP4 that hypothesis H2 compares against on ResNet-8.

Tools and libraries (versions pinned in `versions.lock`)

- Yosys and ABC, via the YosysHQ OSS CAD Suite build of 2026-09-09.
- OpenSTA 3.1.0 (parallaxsw/OpenSTA, commit `3c32e3a2`).
- SkyWater sky130 PDK, `sky130_fd_sc_hd` high-density library, `tt_025C_1v80` corner, open_pdks `1689ac3f`, installed with `ciel`.
- Icarus Verilog 14.0 and cocotb 2.x for verification.
- The ABC driver-cell and output-load values in `synth/abc_sky130hd.constr` are the sky130hd defaults of OpenROAD-flow-scripts.
