// Balanced signed adder tree plus the INT datapath stages built on it
// (build-plan.md S5.1, S5.3).
//
// The INT stages live in this file rather than their own because
// synth/run_synth.py compiles each unit from a fixed source list, and
// adder_tree.v is the one common file every unit (int4, int8, mxint8, and the
// fused-stage units) already reads. Stage modules keep the mul_stage /
// tree_stage / normacc_stage names that the hierarchical breakdown keys on.

// N signed W-bit inputs -> one signed (W + log2 N)-bit sum. N must be a power
// of two. Each level sign-extends by one bit, so no level can overflow.
module adder_tree #(
  parameter N = 32,
  parameter W = 16
) (
  input  wire [N*W-1:0]         in_flat,  // element 0 in the LSBs
  output wire [W+$clog2(N)-1:0] sum
);
  localparam L = $clog2(N);

  // One wire per node (level l holds N>>l values of W+l bits). Separate wires,
  // not one packed vector for all levels: a shared vector makes every node
  // sensitive to every other node, which slows Icarus roughly tenfold and
  // reads as a combinational loop to Verilator.
  genvar l, i;
  generate
    for (l = 0; l <= L; l = l + 1) begin : level
      for (i = 0; i < (N >> l); i = i + 1) begin : node
        wire [W+l-1:0] s;
        if (l == 0) begin : leaf
          assign s = in_flat[i*W +: W];
        end else begin : add
          assign s = $signed(level[l-1].node[2*i].s) + $signed(level[l-1].node[2*i+1].s);
        end
      end
    end
  endgenerate

  assign sum = level[L].node[0].s;
endmodule

// N signed W-bit products, each exactly 2W bits (-2^(W-1) squared still fits).
module mul_stage_int #(
  parameter N = 32,
  parameter W = 8
) (
  input  wire [N*W-1:0]   a_flat,
  input  wire [N*W-1:0]   b_flat,
  output wire [N*2*W-1:0] p_flat
);
  genvar i;
  generate
    for (i = 0; i < N; i = i + 1) begin : lane
      assign p_flat[i*2*W +: 2*W] = $signed(a_flat[i*W +: W]) * $signed(b_flat[i*W +: W]);
    end
  endgenerate
endmodule

// Exact block sum of the products: 21 bits for INT8 (|sum| <= 32 * 128^2 = 2^19),
// 13 bits for INT4 (|sum| <= 32 * 8^2 = 2^11).
module tree_stage_int #(
  parameter N   = 32,
  parameter P_W = 16
) (
  input  wire [N*P_W-1:0]         p_flat,
  output wire [P_W+$clog2(N)-1:0] sum
);
  adder_tree #(.N(N), .W(P_W)) u_tree (.in_flat(p_flat), .sum(sum));
endmodule

// INT has nothing to normalize or round: sign-extend the block sum into the
// INT32 accumulator. Overflow wraps in two's complement (decision D13).
module normacc_stage_int #(
  parameter SUM_W = 21
) (
  input  wire [SUM_W-1:0] sum,
  input  wire [31:0]      acc,
  output wire [31:0]      acc_next
);
  assign acc_next = acc + {{(32-SUM_W){sum[SUM_W-1]}}, sum};
endmodule
