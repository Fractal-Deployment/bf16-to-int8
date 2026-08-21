"""Optional CUDA INT8 block-quant. Falls back to Python loops if .so missing."""
from __future__ import annotations

import ctypes
import math
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

_LIB = None
_LIB_TRIED = False
_SO = Path(__file__).resolve().parent / "cuda" / "libquant_i8.so"


def gpu_quant_available() -> bool:
    global _LIB, _LIB_TRIED
    if _LIB is not None:
        return True
    if _LIB_TRIED:
        return False
    _LIB_TRIED = True
    if not _SO.is_file():
        return False
    try:
        lib = ctypes.CDLL(str(_SO))
        lib.quant_bf16_i8_block.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.quant_bf16_i8_block.restype = ctypes.c_int
        lib.quant_f32_i8_block.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.quant_f32_i8_block.restype = ctypes.c_int
        lib.compare_bf16_i8_block.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.compare_bf16_i8_block.restype = ctypes.c_int
        _LIB = lib
        return True
    except OSError:
        return False


def quant_bf16_i8(raw: bytes, blocksize: int = 64) -> Optional[Tuple[bytes, List[float]]]:
    if blocksize != 64 or not gpu_quant_available() or _LIB is None:
        return None
    n = len(raw) // 2
    if n < 1 or len(raw) != n * 2:
        return None
    nblk = (n + blocksize - 1) // blocksize
    q = (ctypes.c_int8 * n)()
    sc = (ctypes.c_float * nblk)()
    buf = ctypes.create_string_buffer(raw, len(raw))
    rc = _LIB.quant_bf16_i8_block(ctypes.addressof(buf), n, blocksize, q, sc)
    if rc != 0:
        return None
    return bytes(q), [float(sc[i]) for i in range(nblk)]


def compare_bf16_i8(
    raw_bf16: bytes,
    q8: bytes,
    scales: List[float],
    blocksize: int = 64,
) -> Optional[Dict[str, Any]]:
    """RMSE / maxabs of i8*scale vs original BF16. GPU. None if .so missing."""
    if blocksize != 64 or not gpu_quant_available() or _LIB is None:
        return None
    n = len(raw_bf16) // 2
    if n < 1 or len(raw_bf16) != n * 2 or len(q8) != n:
        return None
    nblk = (n + blocksize - 1) // blocksize
    if len(scales) != nblk:
        return None
    q = (ctypes.c_int8 * n).from_buffer_copy(q8)
    sc = (ctypes.c_float * nblk)(*[float(x) for x in scales])
    buf = ctypes.create_string_buffer(raw_bf16, len(raw_bf16))
    ss = ctypes.c_double(0.0)
    mx = ctypes.c_float(0.0)
    ov = ctypes.c_ulonglong(0)
    rc = _LIB.compare_bf16_i8_block(
        ctypes.addressof(buf),
        ctypes.addressof(q),
        ctypes.addressof(sc),
        n,
        blocksize,
        ctypes.byref(ss),
        ctypes.byref(mx),
        ctypes.byref(ov),
    )
    if rc != 0:
        return None
    rmse = math.sqrt(ss.value / float(n)) if n else 0.0
    return {
        "n": n,
        "rmse": rmse,
        "max_abs": float(mx.value),
        "sum_sq": float(ss.value),
        "n_over_half_scale": int(ov.value),
    }
