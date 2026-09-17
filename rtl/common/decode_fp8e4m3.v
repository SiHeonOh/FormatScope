// FP8-E4M3 code decoder (build-plan.md S5.2, docs/fused-stage.md).
// value = (-1)^sign * sig * 2^(exp - 9) for every non-NaN code.
module decode_fp8e4m3 (
  input  wire [7:0] code,
  output wire       sign,
  output wire [3:0] exp,     // E - 1 for normals, 0 for subnormals
  output wire [3:0] sig,     // hidden bit (1 for normals) + 3 mantissa bits
  output wire       is_nan   // S.1111.111; E4M3 has no infinities
);
  wire [3:0] e = code[6:3];
  wire       normal = (e != 4'd0);

  assign sign   = code[7];
  assign sig    = {normal, code[2:0]};
  assign exp    = normal ? e - 4'd1 : 4'd0;
  assign is_nan = (e == 4'hF) && (code[2:0] == 3'h7);
endmodule
