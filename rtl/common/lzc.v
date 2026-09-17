// Leading-zero counter as a balanced tree (docs/fused-stage.md, step 7).
// count = number of zeros above the most significant 1; W when in == 0.
module lzc #(
  parameter W = 32
) (
  input  wire [W-1:0]           in,
  output wire [$clog2(W+1)-1:0] count
);
  localparam L = $clog2(W);
  localparam P = 1 << L;       // W rounded up to a power of two

  // Pad below the LSB with ones so a zero input still counts exactly W.
  wire [P-1:0] padded;
  generate
    if (P == W) begin : exact
      assign padded = in;
    end else begin : pad
      assign padded = {in, {(P-W){1'b1}}};
    end
  endgenerate

  // Level l node i covers 2^l bits, node 0 being the most significant.
  // v: any 1 in the span; c: leading zeros in the span when v = 1 (l bits).
  genvar l, i;
  generate
    for (l = 0; l <= L; l = l + 1) begin : level
      for (i = 0; i < (P >> l); i = i + 1) begin : node
        wire                       v;
        wire [(l > 0 ? l : 1)-1:0] c;
        if (l == 0) begin : leaf
          assign v = padded[P-1-i];
          assign c = 1'b0;
        end else if (l == 1) begin : pair
          assign v = level[0].node[2*i].v | level[0].node[2*i+1].v;
          assign c = ~level[0].node[2*i].v;
        end else begin : merge
          assign v = level[l-1].node[2*i].v | level[l-1].node[2*i+1].v;
          assign c = level[l-1].node[2*i].v ? {1'b0, level[l-1].node[2*i].c}
                                            : {1'b1, level[l-1].node[2*i+1].c};
        end
      end
    end
  endgenerate

  // When P == W the count needs one more bit than the tree, for in == 0 -> W = {1, 0...}.
  // When P > W the padding guarantees a 1, so the tree's count is already complete.
  generate
    if (P == W) begin : full
      assign count = {~level[L].node[0].v, level[L].node[0].c & {L{level[L].node[0].v}}};
    end else begin : padded_count
      assign count = level[L].node[0].c;
    end
  endgenerate
endmodule
