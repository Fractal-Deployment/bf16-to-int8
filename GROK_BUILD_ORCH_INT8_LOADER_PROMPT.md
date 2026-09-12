# Paste this into Grok Build CLI (training_orchestrator)
You are working in **Fractal-Deployment/training_orchestrator** (C-native orch, RTX 3060, Unsloth Phi-4-mini NF4 pin).
## Job
Add an **INT8 pin loader** — load a static pin written by https://github.com/Fractal-Deployment/bf16-to-int8 (`schema: bf16_to_int8_pin_v1`; legacy `nf4_to_int8_pin_v1` still loads). This is **not** a dest-pack flip and **not** a product GEMM change.
Read first (do not skim):
- https://github.com/Fractal-Deployment/bf16-to-int8/blob/main/PIN_ABI.md
- `artifacts/build_20260719/include/tensor_meta.h` (`DTYPE_NF4=5`, `Nf4Meta`)
- Existing NF4 FULL_VRAM / Candle upload path (`llmve_nf4_vram_*`, Mode A expand-at-load)
- `docs/designs/2026-08-18-STEP-SPEED-MEMO-VS-DISK.md` (INT8 is not product; `ORCH_BASE_PACK` does not exist)
## Slice L0 (this prompt — stop here)
1. Detect pin: dir contains `pin.json` with `schema=bf16_to_int8_pin_v1` (or legacy `nf4_to_int8_pin_v1`) and `model.safetensors`. Refuse anything else fail-closed.
2. Parse each `{stem}.weight` (I8 `[out,in]`), `{stem}.weight.int8_scale` (F32, `amax/127`), `{stem}.weight.int8_state` (UTF-8 JSON).
3. Dequant **once at load** (same idea as NF4 Mode A nested-absmax expand):
   `w[i] = signed_i8(b[i]) * scale[i / blocksize]`, `zp=0`, `qmin=-127`.
4. Upload the expanded f32 (or f16) into the existing registry slots those stems already use. Passthrough BF16 tensors copy unchanged.
5. Board honesty, every load:
   - `base_dtype=int8_pin` or `int8_pin_expanded`
   - `int8_gemm=false`
   - `nf4_product_path` unchanged unless this pin is actually selected
6. Opt-in only. Suggested env: `ORCH_INT8_PIN=/abs/path/to/pin` (dir). Default unset = today’s NF4 pin. **Do not** invent `ORCH_BASE_PACK`.
7. Unit test with a **tiny** synthetic pin (the converter tests already write one). Do not require the 3 GiB Phi-4 blob in CI.
## Do not do in this slice
- INT8 Tensor-Core / cublasLt MMA (that is a later kernel, not the loader)
- Per-step requant
- Changing product default off NF4
- Claiming step/s or Unsloth-beat
- Touching boxing TUI
## Done when
- `ORCH_INT8_PIN=...` loads the pin, registry has expanded weights, a 1-step smoke runs on the existing f32/f16 GEMM
- Unset env = NF4 path bit-identical to before
- Test + board fields as above
- Push to the orch repo
L1 (not this prompt): keep I8+scales **resident** and add an INT8 GEMM. Only after L0 is green.
