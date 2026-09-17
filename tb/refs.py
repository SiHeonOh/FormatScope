"""Bit-exact reference models for the DP32 units (build-plan.md S6.2).

Each dp32_* function returns the exact 32-bit accumulator register value the
RTL must hold after one accumulation. Code decoding comes from quant/formats.py
so the quantization and hardware lanes cannot drift apart.
"""

import numpy as np

from quant import formats

BLOCK = formats.BLOCK_SIZE
ACC_MASK = (1 << 32) - 1


def _codes(codes, w):
    codes = np.asarray(codes, dtype=np.int64)
    if codes.shape != (BLOCK,):
        raise ValueError(f"expected {BLOCK} codes, got shape {codes.shape}")
    if codes.min() < 0 or codes.max() >= (1 << w):
        raise ValueError(f"codes out of range for {w} bits")
    return codes


def pack(codes, w):
    """Flatten BLOCK w-bit codes into one integer, element 0 in the LSBs (a_flat/b_flat)."""
    packed = 0
    for i, code in enumerate(_codes(codes, w)):
        packed |= int(code) << (i * w)
    return packed


def to_signed32(value):
    value &= ACC_MASK
    return value - (1 << 32) if value >> 31 else value


def dp32_int(a_codes, b_codes, acc, w):
    """INT4/INT8 (S5.3): acc + sum(a_i * b_i), wrapping at 32 bits (D13).

    Codes are raw two's-complement bit patterns, including the most-negative
    code the quantizer never emits (D1); the hardware still has to handle it.
    """
    a = formats._int_code_to_value(_codes(a_codes, w), w).astype(np.int64)
    b = formats._int_code_to_value(_codes(b_codes, w), w).astype(np.int64)
    return (int(acc) + int(np.dot(a, b))) & ACC_MASK
