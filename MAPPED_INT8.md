# Mapped INT8 — notes, expectations, future

Stamp: 2026-09-02  
Repo: [Fractal-Deployment/bf16-to-int8](https://github.com/Fractal-Deployment/bf16-to-int8) (hangover clone: `Jadon-Fox/nf4-to-int8`)  
Seals: `train_ok=false` · `measured_omega=false` · this pin is **inference placement**, not a train dest

This file is the operator note for the hole+plug VRAM mapper. It is not a green claim.

---

## 1. What changed (this drop)

Wired the Keystone / nested-NF8 hole+plug **into the real BF16 converter** as dest `--to mapped-int8`, and added a budget-aware **VRAM mapper**.

| Path | Role |
|---|---|
| [`vram_map.py`](vram_map.py) | **new.** Profiles `phi4-mini` / `llama-8b` / `qwen-7b`. Hole, host plug, 1-layer ring, KV, decode/prefill acts vs `--vram-gib`. Schema `vram_map_v1`. |
| [`keystone.py`](keystone.py) | inflate-then-INT8 + `compare_paths` (bitplane vs nested vs uniform INT8 vs NF4). |
| [`pin_convert.py`](pin_convert.py) | dest `mapped-int8` → schema `keystone_int8_pin_v1`. Writes hole, plug, INT8 scales, `vram_map.json`. |
| [`nf4_to_int8.py`](nf4_to_int8.py) / [`bf16_to_int8.py`](bf16_to_int8.py) | `--to mapped-int8 --plug nested\|bitplane --vram-gib --ctx --profile`. New `map` subcommand (map only). |
| [`PIN_ABI.md`](PIN_ABI.md) | `keystone_int8_pin_v1` tensor list. |
| tests | bit-exact inflate, Phi-4-mini 12 GiB FITS, nested/bitplane pin write, **NF4 source HARD_BLOCK even with `--allow-requant`**. |

CLI (preferred name):

```bash
python3 bf16_to_int8.py pin --src /path/to/bf16 --out ./mapped-pin \
  --to mapped-int8 --plug nested --vram-gib 12 --ctx 4096 --profile phi4-mini
```

Hangover name `nf4_to_int8.py` is the same entry. Do not read that filename as “convert from NF4”.

### Phi-4-mini sit (nested, 12 GiB, ctx 4096)

- Linears: 128, `n_elem = 3_221_225_472`
- Hole ≈ 1.50 GiB (4 bit/w)
- Host plug ≈ 1.45 GiB (ring holds 1 layer)
- Resident VRAM ≈ **4.71 GiB — FITS**

---

## 2. Expectations (contract)

### Source (fidelity)

The hop that **translates fidelity** is **dense BF16 / F16 / FP32 → pin**. Microsoft ships Phi-4-mini as BF16. That is the only legal source for `mapped-int8` and `nested-nf8`.

**NF4 → INT8 is not a valid fidelity path.** The 4-bit codes already threw away the within-cell bits. Requant of the dequant cannot restore Microsoft’s weights.

| dest | from an NF4 pin | from BF16/F16/FP32 |
|---|---|---|
| `mapped-int8` | **HARD_BLOCK always** (even `--allow-requant`) | yes |
| `nested-nf8` | **HARD_BLOCK always** | yes |
| dense `int8` | refused unless `--allow-requant` (lossy, not a restore) | yes — orch `bf16_to_int8_pin_v1` |
| `nf4` | copy, do not requant | yes — fit-case *destination* |

`--plug nested` uses NF4’s 16 Gaussian **cells as geometry** on the BF16 tensor. That is not reading an Unsloth/bnb-4bit pin.

### Runtime

```
VRAM:  4-bit hole + tiny scales + embed/norms + KV + 1-layer plug ring
RAM:   the rest of the plug
step:  GEMM on layer L (inflated) while CUDA stream copies plug[L+1]
GEMM:  inflate(hole, plug[L]) → INT8   (bitplane inflate is bit-exact BF16 first)
```

Hole-only (plug late) is a **legal BF16** with LSBs zero. Attention / routing can still run blurry. Full fidelity needs the plug.

Do **not** inner-loop PCIe inside `TILE_K`.

### Quality (toy Gaussian, n=256, this repo’s tests)

| path | what |
|---|---|
| bitplane inflate | **bit-exact BF16** |
| bitplane then INT8 | same INT8 bytes as direct BF16→INT8 |
| nested NF8 (4+4) | RMSE in INT8-class (~1.3e-4), better than NF4 (~1.9e-3) |
| uniform INT8 | ~1.1e-4 |

Nested is the 12 GB default. Bitplane is exact and heavier on host (12 bits/w).

### What this drop does **not** claim

- `train_ok` stays **false**. No LoRA / TDC / product step.
- Orch product GEMM is still Unsloth NF4 H-TILE (or the dense INT8 pin loader). It does **not** load `keystone_int8_pin_v1` yet.
- `keystone_htile.cu` is a **sketch**. No fused Ampere INT8 MMA in this repo on this drop.
- Layer skip `{1,3,30}` is Unsloth’s NF4 hybrid leftover. It is **not** part of this pin.

---

## 3. Future development

Priority is **run on 12 GB VRAM** (INT8 performance, NF4-sized resident). Training on this pin is interesting later, not this drop.

### Next (GPU host, RTX 3060 / sm_86)

1. **Fused inflate → INT8 fragment → `mma.sync` INT8** in `keystone_htile.cu`. Goldens vs host inflate. `cp.async` hole + plug, prefetch plug[L+1] on stream 1. No inner-loop PCIe.
2. **Orch loader L0** for `schema=keystone_int8_pin_v1`. Opt-in env (suggested `ORCH_MAPPED_INT8_PIN=`). Unset = today’s path, bit-identical. Do not invent `ORCH_BASE_PACK`. Do not flip the product default.
3. Board: ms/step hole-only vs complete vs NF4 H-TILE vs dense INT8 pin. **No `train_ok` from a board.**

### Later

- Prefill vs decode envelopes in the mapper (prefill already estimated, not fused).
- Embed on host if 8B-class + long ctx blows 12 GiB (`embed_in_vram=false`).
- Streaming plug compression (Huffman/ANS on bitplane LSBs) if host RAM is the wall, not VRAM.
- Training: LoRA on frozen mapped base. Only after the run kernel matches host inflate goldens. `train_ok` is still a measured seal, not a filename.

### Explicitly out

- Requant from Unsloth NF4 / GGUF / GPTQ / AWQ as a “restore”.
- Claiming INT8 Tensor-Core MMA from Python host code.
- Renaming hangover files (`nf4_to_int8.py`) in this drop — `bf16_to_int8.py` is the preferred entry; the hangover name stays as an alias.

---

## 4. How to check

```bash
python3 test_nf4_to_int8.py
python3 bf16_to_int8.py map --profile phi4-mini --plug nested --vram-gib 12
```

Expect `TEST_NF4_TO_INT8_GREEN … mapped_int8 keystone vram_map … NOT_train_ok`.
