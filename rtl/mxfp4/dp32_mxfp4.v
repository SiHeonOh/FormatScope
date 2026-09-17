// MXFP4 32-element dot-product accumulator (build-plan.md S5.1, S5.7).
// E2M1 elements decode to integer magnitudes (value * 2), so every product is
// an exact integer and the block sum is an exact 14-bit tree before the same
// two-term fused stage the MXINT8 unit uses.

// 32 lanes of E2M1 decode + signed product. |mag_a * mag_b| <= 12 * 12 = 144,
// so each signed product fits 9 bits. The magnitude product is a function of
// 6 input bits; synthesis reduces it the same whether written as a table or a
// multiply, so it is written as the multiply for readability.
module mul_stage_mxfp4 #(
  parameter N = 32
) (
  input  wire [N*4-1:0] a_flat,
  input  wire [N*4-1:0] b_flat,
  output wire [N*9-1:0] p_flat
);
  genvar i;
  generate
    for (i = 0; i < N; i = i + 1) begin : lane
      wire       sa, sb;
      wire [3:0] ma, mb;
      wire [7:0] pm;
      wire [8:0] p;

      decode_e2m1 u_dec_a (.code(a_flat[i*4 +: 4]), .sign(sa), .mag(ma));
      decode_e2m1 u_dec_b (.code(b_flat[i*4 +: 4]), .sign(sb), .mag(mb));

      // Sized wire first: inside a concatenation the product would be only 4 bits.
      assign pm               = ma * mb;
      assign p                = {1'b0, pm};
      assign p_flat[i*9 +: 9] = (sa ^ sb) ? (~p + 1'b1) : p;
    end
  endgenerate
endmodule

module dp32_mxfp4 #(
  parameter ALIGN_W = 24              // alignment window, 24 or 32 (decision D6)
) (
  input  wire         clk,
  input  wire         rst_n,          // synchronous, active low
  input  wire         en,             // accumulate this cycle
  input  wire         clear,          // zero the accumulator and flag_nan; wins over en
  input  wire [127:0] a_flat,         // 32 E2M1 elements, element 0 in the LSBs
  input  wire [127:0] b_flat,
  input  wire [7:0]   scale_a,        // E8M0 shared scale of block a
  input  wire [7:0]   scale_b,
  output reg  [31:0]  acc_out,        // FP32 format
  output reg          flag_nan        // sticky: an accumulation saw a 0xFF scale
);
  localparam N     = 32;
  localparam P_W   = 9;
  localparam SUM_W = P_W + $clog2(N);   // 14; |S| <= 32 * 144 = 4608 < 2^13
  localparam EXP_W = 10;

  wire [N*P_W-1:0] p_flat;
  wire [SUM_W-1:0] block_sum;
  wire [7:0]       exp_a, exp_b;
  wire             nan_a, nan_b;
  wire             sign;
  wire [EXP_W-1:0] t;
  wire [SUM_W-2:0] mag;
  wire [31:0]      acc_next;

  mul_stage_mxfp4 #(.N(N)) u_mul (
    .a_flat(a_flat), .b_flat(b_flat), .p_flat(p_flat)
  );

  tree_stage_int #(.N(N), .P_W(P_W)) u_tree (
    .p_flat(p_flat), .sum(block_sum)
  );

  decode_e8m0 u_scale_a (.code(scale_a), .exp(exp_a), .is_nan(nan_a));
  decode_e8m0 u_scale_b (.code(scale_b), .exp(exp_b), .is_nan(nan_b));

  // value = S * 2^(scale_a + scale_b - 254 - 2): each magnitude carries a factor of 2.
  align_stage_mxnorm #(.SUM_W(SUM_W), .EXP_W(EXP_W), .OFFSET(256)) u_norm (
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
