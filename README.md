# convert pin — BF16 → INT8 (or NF4)

**Download BF16 (or F16 / FP32). Convert once.** Do not download GGUF/GPTQ/AWQ/Unsloth-4bit and requant.

Repo name hangover was `nf4-to-int8`. Canonical dest schema is `bf16_to_int8_pin_v1`. **NF4→INT8 is not a fidelity path** — it is a lossy second hop (`--allow-requant` only, and never for `mapped-int8` / `nested-nf8`).

See [MAPPED_INT8.md](MAPPED_INT8.md) for the VRAM mapper (hole in VRAM, plug on RAM, inflate then INT8).

```bash
# default: BF16/F16 safetensors → INT8 pin (orch loader ABI)
python3 bf16_to_int8.py pin --src /path/to/phi-4-mini-bf16 --out /path/to/int8-pin --to int8
# hangover CLI name (same entry):
python3 nf4_to_int8.py pin --src /path/to/phi-4-mini-bf16 --out /path/to/int8-pin --to int8

# VRAM mapper: 4-bit hole in VRAM, plug on RAM, INT8 after inflate
python3 bf16_to_int8.py pin --src /path/to/bf16 --out /path/to/mapped-pin \
  --to mapped-int8 --plug nested --vram-gib 12 --ctx 4096 --profile phi4-mini

# map only (no convert)
python3 bf16_to_int8.py map --profile phi4-mini --plug nested --vram-gib 12

# fit case: same source → NF4 pin
python3 nf4_to_int8.py pin --src /path/to/phi-4-mini-bf16 --out /path/to/nf4-pin --to nf4

# nested NF8: keep the 16 Gaussian cells, plug = sub-index (orch H-TILE L0/L1)
python3 nf4_to_int8.py pin --src /path/to/phi-4-mini-bf16 --out /path/to/nested-nf8 --to nested-nf8

python3 nf4_to_int8.py pin --src model.safetensors --out /tmp/pin --dry-run
# HF snapshots with model.safetensors.index.json (2+ shards) are read as one catalog.
```

Already-quantized NF4 is **refused** unless `--allow-requant` (lossy second hop; cannot restore BF16). `mapped-int8` and `nested-nf8` **never** accept an NF4 source — no within-cell bits to recover.

| flag | default | |
|------|---------|--|
| `--to` | `int8` | dest: `int8` `mapped-int8` `nf4` `nested-nf8` |
| `--plug` | `nested` | mapped-int8: `nested` (INT8-class RMSE) or `bitplane` (exact BF16 then INT8) |
| `--vram-gib` | `12` | mapper budget |
| `--dense` | `quantize` | 16-bit linears → dest. `copy` to leave BF16 |
| `--embed` | `copy` | embeddings / lm_head stay BF16 |
| (norms) | always copy | RMSNorm / bias / rotary never quantized |
| `--allow-requant` | off | dense `int8` dest only, and only if the only copy you have is already NF4 |

GPTQ, AWQ, GGUF: refuse. Get the HuggingFace **BF16/F16** tree.

## Destinations

| dest | when |
|------|------|
| **INT8** | better unfold from BF16, ~8 bit, orch `bf16_to_int8_pin_v1` loader |
| **mapped-int8** | run on 12 GB: 4-bit hole in VRAM, plug on host, inflate then INT8. Schema `keystone_int8_pin_v1` |
| **NF4** | need to **fit** on 12 GB as a 4-bit *destination*. Schema `bf16_to_nf4_pin_v1` |
| **nested-nf8** | NF4 parent cells + 4-bit sub-index. Schema `nested_nf8_pin_v1` |

Ampere still GEMMs in f16/f32 until the mapped INT8 kernel lands on the GPU host. Storage + placement in this repo. `train_ok=false`. Meaning Version: 0.3.4.

INT4 dest is not offered (worse unfold than NF4 at the same size).
