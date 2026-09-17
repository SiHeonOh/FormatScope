// E8M0 (MX shared scale) decoder (build-plan.md S5.2).
// value = 2^(exp - 127); 0xFF is NaN; there is no zero and no infinity.
module decode_e8m0 (
  input  wire [7:0] code,
  output wire [7:0] exp,
  output wire       is_nan
);
  assign exp    = code;
  assign is_nan = (code == 8'hFF);
endmodule
