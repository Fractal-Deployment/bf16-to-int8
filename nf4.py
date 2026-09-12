"""NF4 codebook + dequant. bitsandbytes / Unsloth SSOT.
w = codebook[nibble] * absmax[block]
nibble_order 0 = lo then hi (bnb / Unsloth default).
"""
from __future__ import annotations
import math
import struct
from typing import Iterable, List, Optional, Sequence, Tuple
# Measured 16-level table. Same literals as orch nf4_codebook.h.
NF4_CODEBOOK: Tuple[float, ...] = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)
NIBBLE_LO_THEN_HI = 0
NIBBLE_HI_THEN_LO = 1
def qweight_nbytes(n_elem: int) -> int:
    return (n_elem + 1) // 2
def n_absmax(n_elem: int, blocksize: int) -> int:
    return (n_elem + blocksize - 1) // blocksize
def extract_nibble(byte: int, which: int, nibble_order: int) -> int:
    if nibble_order == NIBBLE_LO_THEN_HI:
        return (byte & 0x0F) if which == 0 else ((byte >> 4) & 0x0F)
    return ((byte >> 4) & 0x0F) if which == 0 else (byte & 0x0F)
def pack_nibbles(first: int, second: int, nibble_order: int) -> int:
    first &= 0x0F
    second &= 0x0F
    if nibble_order == NIBBLE_LO_THEN_HI:
        return first | (second << 4)
    return second | (first << 4)
def pack_indices(
    idx: Sequence[int],
    nibble_order: int = NIBBLE_LO_THEN_HI,
) -> bytearray:
    n = len(idx)
    packed = bytearray(qweight_nbytes(n))
    for i in range(0, n, 2):
        first = int(idx[i]) & 0x0F
        second = int(idx[i + 1]) & 0x0F if i + 1 < n else 0
        packed[i // 2] = pack_nibbles(first, second, nibble_order)
    return packed
def unpack_indices(
    packed: bytes | bytearray,
    n_elem: int,
    nibble_order: int = NIBBLE_LO_THEN_HI,
) -> List[int]:
    out = [0] * n_elem
    for i in range(n_elem):
        out[i] = extract_nibble(packed[i // 2], i & 1, nibble_order)
    return out
def nearest_code(v: float) -> int:
    best = 0
    best_d = abs(v - NF4_CODEBOOK[0])
    for j in range(1, 16):
        d = abs(v - NF4_CODEBOOK[j])
        if d < best_d:
            best_d = d
            best = j
    return best
def dequant_nf4(
    qweight: bytes | bytearray,
    absmax: Sequence[float],
    n_elem: int,
    blocksize: int = 64,
    nibble_order: int = NIBBLE_LO_THEN_HI,
    codebook: Sequence[float] | None = None,
) -> List[float]:
    if blocksize <= 0:
        raise ValueError("blocksize must be > 0")
    cb = codebook if codebook is not None else NF4_CODEBOOK
    if len(cb) != 16:
        raise ValueError("4-bit codebook must have 16 levels")
    need_q = qweight_nbytes(n_elem)
    need_a = n_absmax(n_elem, blocksize)
    if len(qweight) < need_q:
        raise ValueError(f"qweight short: have {len(qweight)} need {need_q}")
    if len(absmax) < need_a:
        raise ValueError(f"absmax short: have {len(absmax)} need {need_a}")
    out = [0.0] * n_elem
    for i in range(n_elem):
        b = qweight[i // 2]
        idx = extract_nibble(b, i & 1, nibble_order)
        scale = float(absmax[i // blocksize])
        out[i] = float(cb[idx]) * scale
    return out
def quantize_nf4(
    weights: Sequence[float],
    blocksize: int = 64,
    nibble_order: int = NIBBLE_LO_THEN_HI,
) -> Tuple[bytearray, List[float]]:
    n_elem = len(weights)
    n_blocks = n_absmax(n_elem, blocksize)
    absmax = [0.0] * n_blocks
    for i, w in enumerate(weights):
        a = abs(float(w))
        b = i // blocksize
        if a > absmax[b]:
            absmax[b] = a
    packed = bytearray(qweight_nbytes(n_elem))
    for i, w in enumerate(weights):
        scale = absmax[i // blocksize]
        idx = 7 if scale == 0.0 else nearest_code(float(w) / scale)
        byte_i = i // 2
        which = i & 1
        cur = packed[byte_i]
        if nibble_order == NIBBLE_LO_THEN_HI:
            first, second = cur & 0x0F, (cur >> 4) & 0x0F
        else:
            first, second = (cur >> 4) & 0x0F, cur & 0x0F
        if which == 0:
            first = idx
        else:
            second = idx
        packed[byte_i] = pack_nibbles(first, second, nibble_order)
    return packed, absmax
def quantize_int8_symmetric(
    weights: Sequence[float],
    blocksize: int = 64,
) -> Tuple[bytearray, List[float]]:
    """q = clip(round(w / scale), -127, 127), scale = amax/127, zp=0."""
    n_elem = len(weights)
    n_blocks = n_absmax(n_elem, blocksize)
    scales = [0.0] * n_blocks
    for i, w in enumerate(weights):
        a = abs(float(w))
        b = i // blocksize
        if a > scales[b]:
            scales[b] = a
    for b in range(n_blocks):
        scales[b] = scales[b] / 127.0 if scales[b] > 0.0 else 1.0
    q = bytearray(n_elem)
    for i, w in enumerate(weights):
        s = scales[i // blocksize]
        v = int(round(float(w) / s))
        if v > 127:
            v = 127
        elif v < -127:
            v = -127
        q[i] = v & 0xFF
    return q, scales
def dequant_int8(q: bytes | bytearray, scales: Sequence[float], blocksize: int = 64) -> List[float]:
    out = [0.0] * len(q)
    for i, b in enumerate(q):
        v = b if b < 128 else b - 256
        out[i] = float(v) * float(scales[i // blocksize])
    return out
def i8_signed(b: int) -> int:
    return b if b < 128 else b - 256
def rmse(a: Iterable[float], b: Iterable[float]) -> float:
    n = 0
    acc = 0.0
    for x, y in zip(a, b):
        d = float(x) - float(y)
        acc += d * d
        n += 1
    return math.sqrt(acc / n) if n else 0.0
def max_abs_err(a: Iterable[float], b: Iterable[float]) -> float:
    m = 0.0
    for x, y in zip(a, b):
        d = abs(float(x) - float(y))
        if d > m:
            m = d
    return m
def pack_f32(vals: Sequence[float]) -> bytes:
    return struct.pack("<" + "f" * len(vals), *[float(v) for v in vals])
def unpack_f32(buf: bytes) -> List[float]:
    n = len(buf) // 4
    return list(struct.unpack("<" + "f" * n, buf[: n * 4]))
def decode_absmax_double(
    absmax_u8: bytes | bytearray,
    nested_quant_map: Sequence[float],
    nested_absmax: Sequence[float],
    nested_offset: float,
    nested_blocksize: int = 256,
) -> List[float]:
    """L1 absmax from bnb double-quant (R09 D1).
    absmax_fp32[i] = nested_quant_map[absmax_u8[i]] * nested_absmax[i // B] + offset
    """
    if nested_blocksize <= 0:
        raise ValueError("nested_blocksize must be > 0")
    if len(nested_quant_map) < 256:
        raise ValueError("nested_quant_map must have 256 levels")
    out = [0.0] * len(absmax_u8)
    for i, code in enumerate(absmax_u8):
        nid = i // nested_blocksize
        if nid >= len(nested_absmax):
            raise ValueError(f"nested_absmax short at block {nid}")
        out[i] = float(nested_quant_map[code]) * float(nested_absmax[nid]) + float(nested_offset)
    return out
def decode_absmax(
    absmax_raw: bytes | bytearray,
    *,
    absmax_is_u8: bool,
    nested_quant_map: Optional[Sequence[float]] = None,
    nested_absmax: Optional[Sequence[float]] = None,
    nested_offset: float = 0.0,
    nested_blocksize: int = 256,
) -> List[float]:
    if not absmax_is_u8:
        return unpack_f32(bytes(absmax_raw))
    if nested_quant_map is None or nested_absmax is None:
        raise ValueError("double-quant absmax needs nested_quant_map + nested_absmax")
    return decode_absmax_double(
        absmax_raw, nested_quant_map, nested_absmax, nested_offset, nested_blocksize
    )
