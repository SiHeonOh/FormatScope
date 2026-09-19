// Fused align -> tree -> normalize -> round-once stage shared by the FP8 and
// MX units (build-plan.md S5.4). Every step and width is derived in
// docs/fused-stage.md; tb/refs.py::fused_stage mirrors it step for step.
//
// Terms use the top-of-field exponent convention:
//   value = (-1)^sign * mag * 2^(T - SIG_W)
// The accumulator is the extra term, unpacked from its FP32-format register.

// Signed max over N (a power of two) W-bit values, one wire per node.
module max_tree_signed #(
  parameter N = 32,
  parameter W = 10
) (
  input  wire [N*W-1:0] in_flat,
  output wire [W-1:0]   max
);
  localparam L = $clog2(N);

  genvar l, i;
  generate
    for (l = 0; l <= L; l = l + 1) begin : level
      for (i = 0; i < (N >> l); i = i + 1) begin : node
        wire [W-1:0] m;
        if (l == 0) begin : leaf
          assign m = in_flat[i*W +: W];
        end else begin : pick
          assign m = ($signed(level[l-1].node[2*i].m) > $signed(level[l-1].node[2*i+1].m))
                   ? level[l-1].node[2*i].m : level[l-1].node[2*i+1].m;
        end
      end
    end
  endgenerate

  assign max = level[L].node[0].m;
endmodule

// Steps 1-4: unpack the accumulator, find the max exponent, align every term
// into the WW-bit window with a sticky bit, and negate. Accumulator term last.
module align_stage_fused #(
  parameter N_PROD  = 32,
  parameter SIG_W   = 8,
  parameter ALIGN_W = 24,
  parameter EXP_W   = 10
) (
  input  wire [N_PROD-1:0]                 sign_flat,
  input  wire [N_PROD*EXP_W-1:0]           exp_flat,
  input  wire [N_PROD*SIG_W-1:0]           mag_flat,
  input  wire [31:0]                       acc,
  output wire [(N_PROD+1)*(ALIGN_W+3)-1:0] term_flat,
  output wire [EXP_W-1:0]                  t_max,
  output wire                              sticky
);
  localparam WW   = ALIGN_W + 2;            // window + guard + round
  localparam TW   = WW + 1;                 // signed aligned term
  localparam SH_W = $clog2(WW + 1);         // clamped shift amount
  localparam N    = N_PROD + 1;
  localparam [SH_W-1:0]  SH_MAX = WW;
  localparam [EXP_W-1:0] FLOOR  = {1'b1, {(EXP_W-1){1'b0}}};

  // Every term as (sign, T, top-aligned WW-bit field), accumulator at index N_PROD.
  wire [N-1:0]       sign_all;
  wire [N*EXP_W-1:0] t_all;
  wire [N*WW-1:0]    top_all;
  wire [N_PROD*EXP_W-1:0] t_eff_prod;       // zero terms get FLOOR for the max

  genvar i;
  generate
    for (i = 0; i < N_PROD; i = i + 1) begin : prod
      wire [SIG_W-1:0] mag = mag_flat[i*SIG_W +: SIG_W];
      wire [WW-1:0]    top;
      assign top = {{(WW-SIG_W){1'b0}}, mag} << (WW - SIG_W);
      assign sign_all[i]              = sign_flat[i];
      assign t_all[i*EXP_W +: EXP_W]  = exp_flat[i*EXP_W +: EXP_W];
      assign top_all[i*WW +: WW]      = top;
      assign t_eff_prod[i*EXP_W +: EXP_W] = (mag != {SIG_W{1'b0}}) ? exp_flat[i*EXP_W +: EXP_W] : FLOOR;
    end
  endgenerate

  // Step 1: the accumulator (biased exponent 0 means zero; no subnormals are stored).
  wire [7:0]       acc_eb      = acc[30:23];
  wire             acc_nonzero = (acc_eb != 8'd0);
  wire [EXP_W-1:0] acc_t       = {{(EXP_W-8){1'b0}}, acc_eb} - 126;
  wire [WW-1:0]    acc_top;
  assign acc_top = acc_nonzero ? ({{(WW-24){1'b0}}, 1'b1, acc[22:0]} << (WW - 24)) : {WW{1'b0}};
  assign sign_all[N_PROD]                 = acc[31];
  assign t_all[N_PROD*EXP_W +: EXP_W]     = acc_t;
  assign top_all[N_PROD*WW +: WW]         = acc_top;

  // Step 2: max exponent over nonzero terms.
  wire [EXP_W-1:0] prod_max;
  max_tree_signed #(.N(N_PROD), .W(EXP_W)) u_max (.in_flat(t_eff_prod), .max(prod_max));
  wire [EXP_W-1:0] acc_t_eff = acc_nonzero ? acc_t : FLOOR;
  assign t_max = ($signed(acc_t_eff) > $signed(prod_max)) ? acc_t_eff : prod_max;

  // Steps 3-4: shift right by min(T_max - T, WW) with sticky, then negate.
  wire [N-1:0] sticky_all;
  generate
    for (i = 0; i < N; i = i + 1) begin : term
      wire [EXP_W-1:0] t_i  = t_all[i*EXP_W +: EXP_W];
      wire [EXP_W:0]   diff = {t_max[EXP_W-1], t_max} - {t_i[EXP_W-1], t_i};
      // A negative diff only occurs for zero terms (their T is ignored by the
      // max); treating it as a large shift keeps them zero.
      wire [SH_W-1:0]  d    = (diff > WW) ? SH_MAX : diff[SH_W-1:0];
      wire [2*WW-1:0]  shifted = {top_all[i*WW +: WW], {WW{1'b0}}} >> d;
      wire [TW-1:0]    ext  = {1'b0, shifted[2*WW-1:WW]};
      assign sticky_all[i]         = |shifted[WW-1:0];
      assign term_flat[i*TW +: TW] = sign_all[i] ? (~ext + 1'b1) : ext;
    end
  endgenerate

  assign sticky = |sticky_all;
endmodule

// Step 5: sum the product terms in one balanced tree, then add the accumulator.
module tree_stage_fused #(
  parameter N_PROD  = 32,
  parameter ALIGN_W = 24
) (
  input  wire [(N_PROD+1)*(ALIGN_W+3)-1:0]      term_flat,
  output wire [ALIGN_W+4+$clog2(N_PROD)-1:0]    sum
);
  localparam TW = ALIGN_W + 3;
  localparam K  = $clog2(N_PROD);

  wire [TW+K-1:0] prod_sum;
  wire [TW-1:0]   acc_term = term_flat[N_PROD*TW +: TW];
  adder_tree #(.N(N_PROD), .W(TW)) u_tree (
    .in_flat(term_flat[N_PROD*TW-1:0]), .sum(prod_sum)
  );
  assign sum = {prod_sum[TW+K-1], prod_sum} + {{(K+1){acc_term[TW-1]}}, acc_term};
endmodule

// Steps 6-8: sign-magnitude, normalize, round once to nearest even, pack FP32.
module normacc_stage_fused #(
  parameter N_PROD  = 32,
  parameter ALIGN_W = 24,
  parameter EXP_W   = 10
) (
  input  wire [ALIGN_W+4+$clog2(N_PROD)-1:0] sum,
  input  wire [EXP_W-1:0]                    t_max,
  input  wire                                sticky_in,
  output wire [31:0]                         acc_next
);
  localparam K      = $clog2(N_PROD);
  localparam SUM_W  = ALIGN_W + 4 + K;
  localparam MAG_W  = SUM_W - 1;
  localparam REST_W = MAG_W - 26;           // >= 1 because ALIGN_W >= 24
  localparam LZ_W   = $clog2(MAG_W + 1);
  localparam [EXP_W-1:0] K_E    = K;
  localparam [EXP_W-1:0] BIAS_E = 127;
  localparam [EXP_W-1:0] EMAX_E = 127;      // largest normal FP32 exponent
  localparam [EXP_W-1:0] EMIN_E = -126;     // smallest

  // Step 6.
  wire             neg = sum[SUM_W-1];
  wire [SUM_W-1:0] abs_sum = neg ? (~sum + 1'b1) : sum;
  wire [MAG_W-1:0] mag = abs_sum[MAG_W-1:0];

  // Step 7.
  wire [LZ_W-1:0]  lz;
  lzc #(.W(MAG_W)) u_lzc (.in(mag), .count(lz));
  wire [MAG_W-1:0] norm   = mag << lz;
  wire [23:0]      sig    = norm[MAG_W-1 -: 24];
  wire             guard  = norm[REST_W+1];
  wire             rnd    = norm[REST_W];
  wire             sticky = sticky_in | (|norm[REST_W-1:0]);
  wire             inc    = guard & (rnd | sticky | sig[0]);
  wire [24:0]      sig_r  = {1'b0, sig} + {24'd0, inc};
  wire             ovf    = sig_r[24];
  // On overflow sig_r is exactly 2^24, so its renormalized fraction (0x800000
  // without the hidden bit) is zero, the same as sig_r[22:0]: no mux needed.
  wire [22:0]      frac   = sig_r[22:0];
  wire [EXP_W-1:0] lz_e   = {{(EXP_W-LZ_W){1'b0}}, lz};
  wire [EXP_W-1:0] e      = t_max + K_E - lz_e + {{(EXP_W-1){1'b0}}, ovf};   // unbiased

  // Step 8. Outside FP32's normal range (reachable only with extreme MX scales):
  // saturate to the largest finite value, or return +0 below the smallest normal.
  wire [EXP_W-1:0] eb        = e + BIAS_E;
  wire             saturate  = $signed(e) > $signed(EMAX_E);
  wire             underflow = $signed(e) < $signed(EMIN_E);
  assign acc_next = (mag == {MAG_W{1'b0}} || underflow) ? 32'd0
                  : saturate ? {neg, 31'h7F7FFFFF}
                  : {neg, eb[7:0], frac};
endmodule

// MX front of the fused stage (S5.6, S5.7): turn the exact signed block sum and
// the two E8M0 scale exponents into one normalized term, so the window starts at
// the block sum's leading one instead of at the top of its field.
//   block value = S * 2^(exp_a + exp_b - OFFSET)
//   term: mag = |S| << lz, T = exp_a + exp_b - OFFSET + SIG_W - lz
module align_stage_mxnorm #(
  parameter SUM_W  = 21,      // signed block sum width; |S| < 2^(SUM_W-1)
  parameter EXP_W  = 10,
  parameter OFFSET = 266      // 254 (two E8M0 biases) + the elements' implicit scale
) (
  input  wire [SUM_W-1:0] block_sum,
  input  wire [7:0]       exp_a,
  input  wire [7:0]       exp_b,
  input  wire             force_zero,   // a NaN scale: the block contributes nothing (D4)
  output wire             sign,
  output wire [EXP_W-1:0] t,
  output wire [SUM_W-2:0] mag
);
  localparam SIG_W = SUM_W - 1;
  localparam LZ_W  = $clog2(SIG_W + 1);
  localparam [EXP_W-1:0] T_OFFSET = OFFSET - SIG_W;

  wire             neg     = block_sum[SUM_W-1];
  wire [SUM_W-1:0] abs_sum = neg ? (~block_sum + 1'b1) : block_sum;
  wire [SIG_W-1:0] raw     = abs_sum[SIG_W-1:0];
  wire [LZ_W-1:0]  lz;
  lzc #(.W(SIG_W)) u_lzc (.in(raw), .count(lz));

  wire [8:0] es = {1'b0, exp_a} + {1'b0, exp_b};
  assign sign = neg;
  assign mag  = force_zero ? {SIG_W{1'b0}} : (raw << lz);
  assign t    = {{(EXP_W-9){1'b0}}, es} - T_OFFSET - {{(EXP_W-LZ_W){1'b0}}, lz};
endmodule

module fused_stage #(
  parameter N_PROD  = 32,
  parameter SIG_W   = 8,
  parameter ALIGN_W = 24,
  parameter EXP_W   = 10
) (
  input  wire [N_PROD-1:0]       sign_flat,
  input  wire [N_PROD*EXP_W-1:0] exp_flat,
  input  wire [N_PROD*SIG_W-1:0] mag_flat,
  input  wire [31:0]             acc,
  output wire [31:0]             acc_next
);
  localparam TW    = ALIGN_W + 3;
  localparam SUM_W = ALIGN_W + 4 + $clog2(N_PROD);

  wire [(N_PROD+1)*TW-1:0] term_flat;
  wire [EXP_W-1:0]         t_max;
  wire                     sticky;
  wire [SUM_W-1:0]         sum;

  align_stage_fused #(.N_PROD(N_PROD), .SIG_W(SIG_W), .ALIGN_W(ALIGN_W), .EXP_W(EXP_W)) u_align (
    .sign_flat(sign_flat), .exp_flat(exp_flat), .mag_flat(mag_flat), .acc(acc),
    .term_flat(term_flat), .t_max(t_max), .sticky(sticky)
  );

  tree_stage_fused #(.N_PROD(N_PROD), .ALIGN_W(ALIGN_W)) u_tree (
    .term_flat(term_flat), .sum(sum)
  );

  normacc_stage_fused #(.N_PROD(N_PROD), .ALIGN_W(ALIGN_W), .EXP_W(EXP_W)) u_normacc (
    .sum(sum), .t_max(t_max), .sticky_in(sticky), .acc_next(acc_next)
  );
endmodule
