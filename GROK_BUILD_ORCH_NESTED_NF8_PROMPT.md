# Paste this into Grok Build CLI (training_orchestrator)
You are in **Fractal-Deployment/training_orchestrator** (C-native, RTX 3060, H-TILE NF4).
## Job
Load **nested NF8** pins from https://github.com/Fractal-Deployment/bf16-to-int8 (`schema: nested_nf8_pin_v1`). Nested dest is **not** `bf16_to_int8_pin_v1`.
This is **not** uniform INT8. **Not** IEEE bitplanes. Hole = existing NF4 nibble (the 16 Gaussian cells). Plug = 4-bit sub-index **inside that cell**. Reconstruction:
```
w[i] = NF8_CELLS[hole_nibble][plug_nibble] * absmax[i / 64]
```
Copy `include/nf8_cells.h` and `include/nested_nf8.h` from that repo (or regenerate: `python3 nested_nf.py --header include/nf8_cells.h`). Do not hand-edit the 16×16 table.
Read first: that repo `PIN_ABI.md` nested_nf8 section, `artifacts/build_20260719/include/tensor_meta.h`, H-TILE S3 launch (`launch_nf4_dequant_gemm_htile1_s3_f32`).
## L0 — load (this prompt — stop here)
1. Detect dir with `pin.json` `schema=nested_nf8_pin_v1` + `model.safetensors`. Fail-closed otherwise.
2. Per linear stem:
   - `{stem}.weight` U8 packed hole (same nibble walk as NF4, lo-then-hi)
   - `{stem}.weight.nf8_plug` U8 packed plug
   - `{stem}.weight.absmax` F32
   - `_nf8_cells` F32 [16,16] (optional; prefer compiled `NF8_CELLS`)
   - norms / embeds passthrough
3. **Expand once at load** (Mode A, same idea as NF4 nested-absmax expand): dequant nested_nf8 → f32 into existing registry slots.
4. Opt-in: `ORCH_NESTED_NF8_PIN=/abs/path`. Unset = today’s NF4 pin. Do not invent `ORCH_BASE_PACK`. Do not change product default.
6. Tiny synthetic pin from converter tests for CI. No 3 GiB blob.
## L1 — not this prompt (after L0 green)
Keep hole+plug **resident**. H-TILE S3: second `cp.async` of plug next to `sh_qpack`. Replace `codebook[nibble]*absmax` with `NF8_CELLS[nibble][plug]*absmax`. Prefetch plug[L+1] on CUDA stream 1 (pinned host) **per layer**, never inside `TILE_K`. Then S5 `mma.sync` on the inflated f16/bf16 fragment.
## Do not
- Uniform INT8 MMA as this path
- Claim nested NF8 is bit-exact BF16
- Mojo required (CUDA only)
## Done when
