# Keystone speed — CUDA, H-TILE mapped, no Mojo
**Target:** RTX 3060 sm_86 CUDA only map onto existing H-TILE S3 `cp.async` D=2 + S5 `mma.sync`. 
## Roofline (why “RAM is faster than dequant” is half true)
| vs | Inner-loop GMEM | ALU | Quality |
|---|---|---|---|
| Unfused QLoRA (dequant **whole** model, then GEMM) | terrible | codebook | 4-bit |
| **Fused NF4 H-TILE** (what orch actually is) | **4 bit/w** | codebook + scale | 4-bit |
| Stream full BF16 layer (TensorRT-style) | 16 bit/w | none | exact |
| **Keystone fast** | 4 bit hole + **12 bit plug of current layer only** | `<<` / `\|` | **exact BF16** |
PCIe 3.0 x16 ~16 GB/s. 3060 VRAM ~360 GB/s. **Never memcpy per K-tile** (launch overhead > transfer). Pull the plug **one layer ahead**, same idea as H-TILE hiding GMEM behind MMA.
Dequant-unfold in H-TILE is already cheap. You do **not** beat it by hitting PCIe in the K-loop. You beat **unfused** unfold, and you beat **full BF16 streaming** by 25% PCIe (4 bits already in VRAM as the hole).
## Fastest config (the one to build)
```
HOST RAM cold plugs (12 LSbs / w, optional zstd)
STREAM 1 cudaMemcpyAsync plug[L+1] → VRAM ring (pinned)
STREAM 0 H-TILE kernel on layer L:
  cp.async D=2 sh_hole ← gmem_hole // same pack as sh_qpack
  cp.async D=2 sh_plug ← gmem_plug[L] // already resident
  wait_group
  bf16 frag = (hole_nibble << 12) | plug12 // NOT codebook
  ldmatrix + mma.sync.bf16.bf16.f32 // S5, Ampere legal
```
- Hole = H-TILE `qpack` (4-bit, all layers, VRAM).
- Plug = missing 12 bits, **one layer** in VRAM (~tens of MB), rest on host.
- Inflate is bitwise. No NF4 table, no absmax mul.
- INT8 MMA: only after inflate→requant, or skip — **BF16 TC is the 3060 exact path**.
Optional: inflate a whole panel, then **cuBLASLt BF16** (orch U2). Faster than a first-party MMA for fat panels; fused H-TILE wins when M is tiny (decode).
## H-TILE map (do not invent a new tile lattice)
| Today NF4 | Keystone |
|---|---|
| `sh_qpack` NF4 nibbles | `sh_hole` 4 MSBs |
| `sh_absmax` scales | **gone** (bits are the scale) |
| codebook[nibble]*absmax | `(hi4<<12)\|lo12` → bf16 bits |
| S3 `cp.async` D=2 | keep, add second `cp.async` for plug |
| S5 `mma.sync f16` | `mma.sync bf16` (sm_86 has it) |
| Mojo / MLIR | **no.** CUDA `.cu` next to `dequant_gemm_ffi.cu` |
## What not to do
- `cudaMemcpy` inside `TILE_K`
- Mapped BAR1 zero-copy as default
- Mojo “because NVVM”
- Claiming this is faster than fused NF4 H-TILE on **bandwidth**. It isn’t. It is faster than materialize-then-GEMM, cheaper ALU than codebook, exact BF16, 25% less PCIe than streaming BF16.
## Build order on the 3060
1. Host: split safetensors BF16 → hole.safetensors (VRAM) + plug.bin (pinned host). `keystone.py` already bit-exact.
2. CUDA: pin plug, double-buffer `cudaMemcpyAsync`, dummy kernel `inflate_or`.
3. Wire inflate into H-TILE S3 load path; keep S3 FMA golden vs `inflate` host.
4. Flip S5 `mma.sync` **bf16** once goldens hold.
5. Board: ms/step hole-only vs complete vs NF4 H-TILE. No ``.
