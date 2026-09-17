// FP8-E4M3 32-element dot-product accumulator (build-plan.md S5.1, S5.5).
// Per lane: decode, significand product, exponent sum, sign XOR. All 32 terms
// and the FP32-format accumulator then go through one fused stage.

// 32 lanes of decode + multiply. NaN lanes contribute zero (decision D3).
module mul_stage_fp8 #(
  parameter N     = 32,
  parameter EXP_W = 10
) (
  input  wire [N*8-1:0]     a_flat,
  input  wire [N*8-1:0]     b_flat,
  output wire [N-1:0]       sign_flat,
  output wire [N*EXP_W-1:0] exp_flat,   // T = exp_a + exp_b - 10 (top-of-field, docs/fused-stage.md)
  output wire [N*8-1:0]     mag_flat,   // sig_a * sig_b <= 225
  output wire               nan_any
);
  wire [N-1:0] nan;

  genvar i;
  generate
    for (i = 0; i < N; i = i + 1) begin : lane
      wire       sa, sb, na, nb;
      wire [3:0] ea, eb, ga, gb;
      wire [4:0] es;

      decode_fp8e4m3 u_dec_a (.code(a_flat[i*8 +: 8]), .sign(sa), .exp(ea), .sig(ga), .is_nan(na));
      decode_fp8e4m3 u_dec_b (.code(b_flat[i*8 +: 8]), .sign(sb), .exp(eb), .sig(gb), .is_nan(nb));

      assign es                          = ea + eb;
      assign nan[i]                      = na | nb;
      assign sign_flat[i]                = sa ^ sb;
      assign mag_flat[i*8 +: 8]          = nan[i] ? 8'd0 : ga * gb;
      assign exp_flat[i*EXP_W +: EXP_W]  = {{(EXP_W-5){1'b0}}, es} - 10;
    end
  endgenerate

  assign nan_any = |nan;
endmodule

module dp32_fp8e4m3 #(
  parameter ALIGN_W = 24              // alignment window, 24 or 32 (decision D6)
) (
  input  wire         clk,
  input  wire         rst_n,          // synchronous, active low
  input  wire         en,             // accumulate this cycle
  input  wire         clear,          // zero the accumulator and flag_nan; wins over en
  input  wire [255:0] a_flat,         // 32 FP8-E4M3 codes, element 0 in the LSBs
  input  wire [255:0] b_flat,
  input  wire [7:0]   scale_a,        // unused (MX units only); tie to 8'd127
  input  wire [7:0]   scale_b,
  output reg  [31:0]  acc_out,        // FP32 format
  output reg          flag_nan        // sticky: an accumulation saw a NaN code
);
  localparam N     = 32;
  localparam EXP_W = 10;

  wire [N-1:0]       sign_flat;
  wire [N*EXP_W-1:0] exp_flat;
  wire [N*8-1:0]     mag_flat;
  wire               nan_any;
  wire [31:0]        acc_next;

  mul_stage_fp8 #(.N(N), .EXP_W(EXP_W)) u_mul (
    .a_flat(a_flat), .b_flat(b_flat),
    .sign_flat(sign_flat), .exp_flat(exp_flat), .mag_flat(mag_flat), .nan_any(nan_any)
  );

  fused_stage #(.N_PROD(N), .SIG_W(8), .ALIGN_W(ALIGN_W), .EXP_W(EXP_W)) u_fused (
    .sign_flat(sign_flat), .exp_flat(exp_flat), .mag_flat(mag_flat),
    .acc(acc_out), .acc_next(acc_next)
  );

  always @(posedge clk) begin
    if (!rst_n || clear) begin
      acc_out  <= 32'd0;
      flag_nan <= 1'b0;
    end else if (en) begin
      acc_out  <= acc_next;
      flag_nan <= flag_nan | nan_any;
    end
  end
endmodule
