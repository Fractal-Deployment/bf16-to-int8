# Keystone — hole + plug (not a lattice)
VRAM holds a **4-bit-sized holey bitplane** of BF16. Host RAM holds the **plug** (the missing bits, entropy-coded). Inflate is **bitwise complete**, not codebook lookup.
Mojo is **not** the only NVVM path. See § languages.
## Inflate
```
BF16 bits: s eeeeeeee mmmmmmm (1+8+7)
Hole (VRAM, ~4b/w packed): coarse bitplanes you choose (e.g. s + 3 exp bits, or 4 MSBs)
Plug (pinned RAM): remaining bitplanes, Huffman/ANS/DFloat11-style
Inflate: hole | (plug << k) → full BF16 (bit-exact if all planes)
                         or round(inflate) → INT8 pin
```
If the plug is late, GEMM still runs on the hole (blurry). When the plug lands, a **correction** (XOR the new bits / SAXPY the delta) — Tensor Cores stay fed.
## Languages that reach NVVM / the same PTX
NVVM is NVIDIA's LLVM IR. **CUDA C++ `nvcc` already emits NVVM.** MLIR's `nvvm` dialect is one frontend.
| Stack | Hits `nvvm.cp.async` / `mma.sync`? | sm_86 |
|---|---|---|
| **CUDA C++ / PTX inline / CUTLASS** | Yes. Native. Orch already here. | Yes |
| **MLIR nvvm / nvgpu dialects** | Yes. Language-agnostic. | Yes (`cp.async`, `mma.sync`; no TMA/wgmma) |
| **IREE / Torch-MLIR / OpenXLA** | Yes, via gpu+nvvm | Partial |
| **Triton** | Own PTX path, **not** the nvvm dialect | Yes |
| **Mojo / MAX** | MLIR-native, `LayoutTensor` | Yes, extra glue |
| **CuTeDSL / ThunderKittens** | CUDA, not MLIR | Yes |
**Verdict:** you do **not** need Mojo to use NVVM features. For a 3060 host→device puzzle bus, **CUDA streams + `cp.async` is the sure path.** Mojo is optional if MAX graph fusion is the wall. Orch already has H-TILE `cp.async` D=2 charge.
## Every similar cache / prefetch (inventory)
### GPU on-chip (keep MMA fed)
1. Registers / fragment 
2. SMEM tiles + **double/triple buffer** 
3. **`cp.async` Ampere** GMEM→SMEM (bypass RF) — orch H-TILE S3 
4. L1 / L2; Hopper+ `cp.async.bulk.prefetch.L2` (**not** 3060) 
5. `ldmatrix` → MMA fragments 
6. Warp specialization (producer copy vs consumer MMA) 
7. `nvvm.mbarrier` / `cp.async.wait_group` so INT8 MMA never waits on an empty fragment 
8. CUDA Graphs (orch GRAPH stays 0 until R7)
### Host DRAM → GPU (the puzzle bus)
9. **Pinned `cudaMallocHost` + `cudaMemcpyAsync`** on a side stream — primary 3060 tool 
10. **Mapped zero-copy** (`cudaHostAllocMapped`) — BAR1 tiny on GA106; last resort 
11. **UVM + `cudaMemPrefetchAsync`** — easy, slower than pinned 
12. **TensorRT weight streaming** — host weights, stream at exec 
13. **IREE Stream dialect** — loads vs gathers, unified-memory map 
14. DeepSpeed ZeRO-Infinity / FlexGen CPU offload 
15. llama.cpp mmap + layer GPU 
16. Persistent kernel + CPU enqueue next tile 
### Compress in transit
17. **nvCOMP** (ANS/Bitcomp) decompress on GPU; DFloat11 **beats** it for BF16 exponents 
18. ZipNN / DFloat11 lossless ~1.5:1 on BF16 — plug is this leftover 
19. Invariant Bit Packing (IBP) for CPU→GPU 
### Not this (KV, not weights)
20. vLLM paged attention, KV L2 prefetch papers 
**3060 rule:** PCIe 3.0 x16 ~16 GB/s. Plug per tile must be **≪ GEMM time**. Bitplanes + entropy, not a second full BF16.
## Why hole+plug ≠ lattice
Lattice = 16 reconstruction values, residual discarded or UV-approximated. 
Hole+plug = **the original bits, split**. Complete plug ⇒ **bit-exact BF16**. Incomplete plug ⇒ progressive quality, still a legal GEMM.
INT8 path: inflate BF16 then quantize, **or** treat hole as INT8-aligned MSBs and plug as extra mantissa for a correction MMA. Prefer **inflate to BF16** (user: “this different”). INT8 MMA only if you then requant the completed tile (two-step, keep TCs busy on tile N while plugging N+1).
