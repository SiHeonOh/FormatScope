# References

Format definitions

- Open Compute Project, *OCP 8-bit Floating Point Specification (OFP8)*, revision 1.0, 2023. Defines E4M3 (bias 7, max 448, one NaN code, no infinities) and E5M2. `quant/formats.py` and `rtl/common/decode_fp8e4m3.v` implement E4M3 as specified; overflow saturates to ±448 (decision D2).
- Open Compute Project, *OCP Microscaling Formats (MX) Specification*, version 1.0, 2023. Defines the block-32 shared E8M0 scale, the MXINT8 element (INT8 with an implicit 2⁻⁶), and the MXFP4 element (E2M1). `quant/formats.py` implements the shared-exponent rule; `rtl/common/decode_e8m0.v` and `decode_e2m1.v` decode the codes exhaustively tested in `tb/test_decode.py`.

Papers

- M. van Baalen et al., *FP8 versus INT8 for efficient deep learning inference*, arXiv:2303.17951, 2023. Source of the 50–180% single-MAC hardware-cost estimate that hypothesis H1 tests against a fused 32-lane dot product.
- P. Micikevicius et al., *FP8 Formats for Deep Learning*, arXiv:2209.05433, 2022. The E4M3/E5M2 proposal that the OCP OFP8 specification standardized.
- B. Darvish Rouhani et al., *Microscaling Data Formats for Deep Learning*, arXiv:2310.10537, 2023. Accuracy results for MXINT8 and MXFP4 that hypothesis H2 compares against on ResNet-8.

Related work (what FormatScope is closest to, and how it differs; summarized in the README)

- E. Samson, N. Mellempudi, W. Luk, G. A. Constantinides, *Exploring FPGA designs for MX and beyond*, FPL 2024, arXiv:2407.01475. The first open-source FPGA implementation of the MX standard's arithmetic, with a Brevitas-integrated PyTorch quantization library and an accuracy-versus-FPGA-area Pareto study on ResNet-18/ImageNet. The closest prior work to the FormatScope frontier. Differences here: ASIC standard-cell area and OpenSTA delay rather than FPGA LUT estimates, INT and FP8 baselines measured in the same flow, a per-stage area breakdown, and bit-exact verification of every unit against a reference shared with the quantizer.
- M. Chen et al., *INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats*, arXiv:2510.25602, 2025. LLM accuracy plus a hardware cost model: MXINT8 beats MXFP8 on accuracy and cost, while at 4 bits the FP element formats usually hold the accuracy advantage. Our `int4_b32` versus `mxfp4` ablation lands the other way on ResNet-8/CIFAR-10 (`docs/results.md`); the settings differ (a small CNN with every layer quantized, against LLMs with activation outliers), so this is a discussion point, not a contradiction.
- B. Darvish Rouhani et al., *With Shared Microexponents, A Little Shifting Goes a Long Way*, ISCA 2023, arXiv:2302.08007. The Block Data Representations framework from which the MX formats were identified.
- S. Cuyckens et al., *Precision-Scalable Microscaling Datapaths with Optimized Reduction Tree for Efficient NPU Integration*, arXiv:2511.06313, 2025. States the trade-off the fused stage sits in: integer accumulation needs costly conversion of narrow FP products, FP32 accumulation pays for normalization and rounding loss. Proposes a hybrid reduction tree.
- *MXDOTP: A RISC-V ISA Extension for Enabling Microscaling (MX) Floating-Point Dot Products*, arXiv:2505.13159, 2025 (ETH Zurich, Snitch core, 12 nm). A fused MXFP8 dot-product-accumulate unit: scale handling, dot product, and accumulation in one datapath. The same idea as `rtl/common/fused_stage.v` (exact integer tree, then one shift and one rounding); we cite it as validation of the architecture and do not claim the architecture.

Tools and libraries (versions pinned in `versions.lock`)

- Yosys and ABC, via the YosysHQ OSS CAD Suite build of 2026-09-09.
- OpenSTA 3.1.0 (parallaxsw/OpenSTA, commit `3c32e3a2`).
- SkyWater sky130 PDK, `sky130_fd_sc_hd` high-density library, `tt_025C_1v80` corner, open_pdks `1689ac3f`, installed with `ciel`.
- Icarus Verilog 14.0 and cocotb 2.x for verification.
- The ABC driver-cell and output-load values in `synth/abc_sky130hd.constr` are the sky130hd defaults of OpenROAD-flow-scripts.
