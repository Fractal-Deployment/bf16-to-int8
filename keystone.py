"""Keystone hole+plug: split BF16 bitplanes. Inflate is bitwise, not a codebook.

VRAM hole = 4 MSBs/weight packed. RAM plug = 12 LSBs.
Complete plug → bit-exact BF16. Then optional INT8 for Ampere GEMM.
hole_only is legal BF16 (LSBs zero) so attention/MLP can still run blurry.

Source must be dense BF16/F16/F32. NF4 has no within-cell bits to recover.
train_ok=false.
"""
from __future__ import annotations

import struct
from typing import List, Sequence, Tuple

from dtype_io import pack_bf16, unpack_bf16
from nf4 import dequant_int8, max_abs_err, quantize_int8_symmetric, rmse


def split_bf16(vals: Sequence[float]) -> Tuple[bytes, bytes]:
    raw = pack_bf16(vals)
    n = len(vals)
    hole = bytearray((n + 1) // 2)  # 4 bits each
    plug = bytearray(n * 2)  # 12 bits in u16 low
    for i in range(n):
        u16 = struct.unpack_from("<H", raw, i * 2)[0]
        hi4 = (u16 >> 12) & 0xF
        lo12 = u16 & 0x0FFF
        if i & 1 == 0:
            hole[i // 2] = (hole[i // 2] & 0xF0) | hi4
        else:
            hole[i // 2] = (hole[i // 2] & 0x0F) | (hi4 << 4)
        struct.pack_into("<H", plug, i * 2, lo12)
    return bytes(hole), bytes(plug)


def inflate(hole: bytes, plug: bytes, n: int) -> List[float]:
    raw = bytearray(n * 2)
    for i in range(n):
        b = hole[i // 2]
        hi4 = (b & 0x0F) if (i & 1) == 0 else ((b >> 4) & 0x0F)
        lo12 = struct.unpack_from("<H", plug, i * 2)[0] & 0x0FFF
        struct.pack_into("<H", raw, i * 2, (hi4 << 12) | lo12)
    return unpack_bf16(bytes(raw))


def hole_only(hole: bytes, n: int) -> List[float]:
    """Missing plug: LSBs zero. Legal BF16, wrong values. MMA can still run."""
    raw = bytearray(n * 2)
    for i in range(n):
        b = hole[i // 2]
        hi4 = (b & 0x0F) if (i & 1) == 0 else ((b >> 4) & 0x0F)
        struct.pack_into("<H", raw, i * 2, hi4 << 12)
    return unpack_bf16(bytes(raw))


def inflate_then_int8(
    hole: bytes, plug: bytes, n: int, blocksize: int = 64
) -> Tuple[bytearray, List[float], List[float]]:
    """Runtime: plug lands → exact BF16 → symmetric INT8 for GEMM."""
    rec = inflate(hole, plug, n)
    q8, scales = quantize_int8_symmetric(rec, blocksize=blocksize)
    return q8, scales, rec


def hole_only_then_int8(
    hole: bytes, n: int, blocksize: int = 64
) -> Tuple[bytearray, List[float], List[float]]:
    blur = hole_only(hole, n)
    q8, scales = quantize_int8_symmetric(blur, blocksize=blocksize)
    return q8, scales, blur


def compare_paths(w: Sequence[float], blocksize: int = 64) -> dict:
    from nested_nf import decode_nested, encode_nested
    from nf4 import dequant_nf4, quantize_nf4

    n = len(w)
    hole, plug = split_bf16(w)
    exact = inflate(hole, plug, n)
    blur = hole_only(hole, n)
    w_bf = unpack_bf16(pack_bf16(w))
    q8, s8 = quantize_int8_symmetric(w_bf, blocksize)
    h8 = dequant_int8(q8, s8, blocksize)
    q8i, s8i, _ = inflate_then_int8(hole, plug, n, blocksize)
    h8i = dequant_int8(q8i, s8i, blocksize)
    qn, an = quantize_nf4(w, blocksize)
    hn = dequant_nf4(qn, an, n, blocksize)
    nf4, nplug, nam = encode_nested(w, blocksize)
    hnest = decode_nested(nf4, nplug, nam, blocksize)
    packed = pack_bf16(w)
    rec_p = pack_bf16(exact)
    return {
        "n": n,
        "vram_hole_bytes": len(hole),
        "ram_plug_bytes": len(plug),
        "bits_vram": 8 * len(hole) / n,
        "inflate_bit_exact_bf16": packed == rec_p,
        "rmse_hole_only": rmse(w, blur),
        "rmse_keystone_inflate": rmse(w_bf, exact),
        "rmse_nf4": rmse(w, hn),
        "rmse_nested_nf8": rmse(w, hnest),
        "rmse_uniform_int8": rmse(w, h8),
        "rmse_keystone_then_int8": rmse(w_bf, h8i),
        "max_abs_hole_only": max_abs_err(w, blur),
        "int8_bytes_match_source": bytes(q8) == bytes(q8i),
        "note": "bitplane plug = exact BF16; nested plug ≈ INT8. Source is BF16. train_ok=false",
    }


if __name__ == "__main__":
    from nf4_to_int8 import demo_weights

    w = demo_weights(256)
    hole, plug = split_bf16(w)
    rec = inflate(hole, plug, len(w))
    packed = pack_bf16(w)
    rec_p = pack_bf16(rec)
    exact = packed == rec_p
    blur = hole_only(hole, len(w))
    cmp = compare_paths(w)
    print(
        {
            "n": len(w),
            "vram_hole_bytes": len(hole),
            "ram_plug_bytes": len(plug),
            "bits_vram": 8 * len(hole) / len(w),
            "inflate_bit_exact_bf16": exact,
            "hole_only_not_exact": pack_bf16(blur) != packed,
            **{
                k: cmp[k]
                for k in (
                    "rmse_hole_only",
                    "rmse_nested_nf8",
                    "rmse_uniform_int8",
                    "rmse_keystone_then_int8",
                    "int8_bytes_match_source",
                )
            },
        }
    )
