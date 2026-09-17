// INT8 32-element dot-product accumulator (build-plan.md S5.1, S5.3).
// Combinational multiply -> tree -> accumulate, feeding one registered INT32.
module dp32_int8 #(
  parameter ALIGN_W = 24              // unused; keeps the port list common to all five units
) (
  input  wire         clk,
  input  wire         rst_n,          // synchronous, active low
  input  wire         en,             // accumulate this cycle
  input  wire         clear,          // zero the accumulator; wins over en
  input  wire [255:0] a_flat,         // 32 two's-complement codes, element 0 in the LSBs
  input  wire [255:0] b_flat,
  input  wire [7:0]   scale_a,        // unused (MX units only); tie to 8'd127
  input  wire [7:0]   scale_b,
  output reg  [31:0]  acc_out,        // INT32, wraps on overflow (D13)
  output reg          flag_nan        // INT has no NaN codes; always 0
);
  localparam N     = 32;
  localparam W     = 8;
  localparam P_W   = 2 * W;
  localparam SUM_W = P_W + $clog2(N);

  wire [N*P_W-1:0] p_flat;
  wire [SUM_W-1:0] sum;
  wire [31:0]      acc_next;

  mul_stage_int #(.N(N), .W(W)) u_mul (
    .a_flat(a_flat), .b_flat(b_flat), .p_flat(p_flat)
  );

  tree_stage_int #(.N(N), .P_W(P_W)) u_tree (
    .p_flat(p_flat), .sum(sum)
  );

  normacc_stage_int #(.SUM_W(SUM_W)) u_normacc (
    .sum(sum), .acc(acc_out), .acc_next(acc_next)
  );

  always @(posedge clk) begin
    if (!rst_n || clear) begin
      acc_out  <= 32'd0;
      flag_nan <= 1'b0;
    end else if (en) begin
      acc_out  <= acc_next;
    end
  end
endmodule
