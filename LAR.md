# LAR — Lattice + Async Residual

Not QLoRA. Not W4A8-only. One INT4 **lattice** (pad → INT8 MMA is lossless for the codes) plus the **residual plane** \(R = W_{\mathrm{BF16}} - \hat W_4\) kept in **host RAM** and software-pipelined like a third cache.

## Why a third plane

| | VRAM | Compute | Residual |
|---|---|---|---|
| QLoRA | NF4 codebook | dequant → BF16 GEMM | thrown away; LoRA patches function |
| W4A8 | INT4 pack | pad → INT8 MMA, A8 | thrown away |
| SpQR / LQER | 4-bit + outliers/UV **on GPU** | mixed | in VRAM |
| **LAR** | INT4 lattice only | pad-INT8 MMA **or** dequant-BF16 | **host**, async tiles |

Same lattice serves **train unfold** (INT4→BF16, NF4-style) and **infer MMA** (INT4→INT8). NF4 cannot pad into `s8` MMA. That is the pivot.

## 3060 hierarchy

Ampere has `cp.async` (no TMA, no wgmma). PCIe 3.0 x16 ~16 GB/s.

- Stream 0: `cp.async` lattice tile GMEM→SMEM → `prmt` pad → `mma.sync … s8.s8.s32`
- Stream 1: `cudaMemcpyAsync` residual tile (UV or sparse COO) host→small VRAM ring
- Epilogue: \(Y \mathrel{+}= U(Vx)\) and/or sparse SAXPY

Rank-16 UV per 4k×4k matrix is ~256 KiB. Whole-model UV is tens of MB — PCIe is not the wall. 5% sparse BF16 residual on 7B is ~700 MB/step if naively streamed — only works **tiled** and overlapped.

MLIR/NVVM (exploratory, >30%): `nvvm.cp.async` + `nvvm.mma.sync` + `nvvm.mbarrier` on sm_86; residual producer warp vs MMA consumer warp. Mapped pinned residual is a maybe (BAR1 is small on GA106) — memcpy ring first.

## Codec sketch

```
python3 lar.py   # RMSE bakeoff, stdlib
```

Gaussian 64×64 toy (not a net):

| reconstruct | RMSE |
|---|---|
| INT8 | 1.1e-4 |
| NF4 | 1.85e-3 |
| INT4 lattice | 1.93e-3 |
| INT4 + rank-4 UV | 1.69e-3 |
| INT4 + rank-4 + 2% sparse | **1.56e-3** (beats NF4 here) |

Pad INT4→INT8 is bit-identity on the codes.

Not product. Not Phi-4. Not better-than-QLoRA on a task until a bakeoff.
