// MXINT8 32-element dot-product accumulator (build-plan.md S5.1, S5.6).
// The INT8 multiply and tree stages give the exact 21-bit block sum; the two
// E8M0 scales set its exponent; one normalized term plus the FP32-format
// accumulator go through the fused stage with N_PROD = 1.
module dp32_mxint8 #(
  parameter ALIGN_W = 24              // alignment window, 24 or 32 (decision D6)
) (
  input  wire         clk,
  input  wire         rst_n,          // synchronous, active low
  input  wire         en,             // accumulate this cycle
  input  wire         clear,          // zero the accumulator and flag_nan; wins over en
  input  wire [255:0] a_flat,         // 32 INT8 elements (value = code / 64), element 0 in the LSBs
  input  wire [255:0] b_flat,
  input  wire [7:0]   scale_a,        // E8M0 shared scale of block a
  input  wire [7:0]   scale_b,
  output reg  [31:0]  acc_out,        // FP32 format
  output reg          flag_nan        // sticky: an accumulation saw a 0xFF scale
);
  localparam N     = 32;
  localparam W     = 8;
  localparam P_W   = 2 * W;
  localparam SUM_W = P_W + $clog2(N);   // 21
  localparam EXP_W = 10;

  wire [N*P_W-1:0] p_flat;
  wire [SUM_W-1:0] block_sum;
  wire [7:0]       exp_a, exp_b;
  wire             nan_a, nan_b;
  wire             sign;
  wire [EXP_W-1:0] t;
  wire [SUM_W-2:0] mag;
  wire [31:0]      acc_next;

  mul_stage_int #(.N(N), .W(W)) u_mul (
    .a_flat(a_flat), .b_flat(b_flat), .p_flat(p_flat)
  );

  tree_stage_int #(.N(N), .P_W(P_W)) u_tree (
    .p_flat(p_flat), .sum(block_sum)
  );

  decode_e8m0 u_scale_a (.code(scale_a), .exp(exp_a), .is_nan(nan_a));
  decode_e8m0 u_scale_b (.code(scale_b), .exp(exp_b), .is_nan(nan_b));

  // value = S * 2^(scale_a + scale_b - 254 - 12): elements carry an implicit 2^-6 each.
  align_stage_mxnorm #(.SUM_W(SUM_W), .EXP_W(EXP_W), .OFFSET(266)) u_norm (
    .block_sum(block_sum), .exp_a(exp_a), .exp_b(exp_b), .force_zero(nan_a | nan_b),
    .sign(sign), .t(t), .mag(mag)
  );

  fused_stage #(.N_PROD(1), .SIG_W(SUM_W-1), .ALIGN_W(ALIGN_W), .EXP_W(EXP_W)) u_fused (
    .sign_flat(sign), .exp_flat(t), .mag_flat(mag), .acc(acc_out), .acc_next(acc_next)
  );

  always @(posedge clk) begin
    if (!rst_n || clear) begin
      acc_out  <= 32'd0;
      flag_nan <= 1'b0;
    end else if (en) begin
      acc_out  <= acc_next;
      flag_nan <= flag_nan | nan_a | nan_b;
    end
  end
endmodule
