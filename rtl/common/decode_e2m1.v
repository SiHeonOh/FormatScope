// E2M1 (MXFP4 element) decoder (build-plan.md S5.2).
// mag = element value * 2, so every magnitude is an integer.
module decode_e2m1 (
  input  wire [3:0] code,
  output wire       sign,
  output reg  [3:0] mag      // {0, 1, 2, 3, 4, 6, 8, 12} = {0, 0.5, 1, 1.5, 2, 3, 4, 6} * 2
);
  assign sign = code[3];

  always @* begin
    case (code[2:0])
      3'd0:    mag = 4'd0;
      3'd1:    mag = 4'd1;
      3'd2:    mag = 4'd2;
      3'd3:    mag = 4'd3;
      3'd4:    mag = 4'd4;
      3'd5:    mag = 4'd6;
      3'd6:    mag = 4'd8;
      default: mag = 4'd12;
    endcase
  end
endmodule
