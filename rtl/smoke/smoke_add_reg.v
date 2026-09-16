// Day 0 smoke test (build-plan.md S3.7d): the S3.7a adder with a registered
// output, so OpenSTA has a real clock and a register endpoint to time.
module smoke_add_reg(
  input  wire       clk,
  input  wire [3:0] a,
  input  wire [3:0] b,
  output reg  [4:0] y
);
  always @(posedge clk)
    y <= a + b;
endmodule
