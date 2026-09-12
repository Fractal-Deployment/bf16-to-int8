"""Unpack BF16 / FP16 / F32 storage to host f32. Stdlib only."""
from __future__ import annotations
import math
import struct
from typing import List, Sequence, Tuple
from nf4 import pack_f32, unpack_f32
def unpack_bf16(buf: bytes | bytearray) -> List[float]:
    n = len(buf) // 2
    out = [0.0] * n
    for i in range(n):
        u16 = struct.unpack_from("<H", buf, i * 2)[0]
        (out[i],) = struct.unpack("<f", struct.pack("<I", u16 << 16))
    return out
def _f16_to_f32(h: int) -> float:
    s = (h >> 15) & 1
    e = (h >> 10) & 0x1F
    f = h & 0x3FF
    if e == 0:
        v = 0.0 if f == 0 else math.ldexp(f / 1024.0, -14)
    elif e == 31:
        v = float("inf") if f == 0 else float("nan")
    else:
        v = math.ldexp(1.0 + f / 1024.0, e - 15)
    return -v if s else v
def unpack_f16(buf: bytes | bytearray) -> List[float]:
    n = len(buf) // 2
    out = [0.0] * n
    for i in range(n):
        h = struct.unpack_from("<H", buf, i * 2)[0]
        out[i] = _f16_to_f32(h)
    return out
def pack_bf16(vals: Sequence[float]) -> bytes:
    """Round-to-nearest even-ish: take high 16 bits of f32 (trunc toward 0 on ties)."""
    out = bytearray(len(vals) * 2)
    for i, v in enumerate(vals):
        (u,) = struct.unpack("<I", struct.pack("<f", float(v)))
        struct.pack_into("<H", out, i * 2, (u >> 16) & 0xFFFF)
    return bytes(out)
def unpack_dense(dtype: str, buf: bytes, shape: Tuple[int, ...]) -> List[float]:
    if dtype == "BF16":
        return unpack_bf16(buf)
    if dtype == "F16":
        return unpack_f16(buf)
    if dtype == "F32":
        return unpack_f32(buf)
    raise ValueError(f"dense unpack does not support dtype {dtype}")
def numel_of(shape: Tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n
