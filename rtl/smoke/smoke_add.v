// Day 0 smoke test (build-plan.md S3.7a): a 4-bit adder must synthesize to a
// positive area against sky130_fd_sc_hd.
module smoke_add(input [3:0] a, b, output [4:0] y);
  assign y = a + b;
endmodule
