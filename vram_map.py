"""VRAM mapper — NF4-sized hole resident, plug on host, INT8 after inflate.

Inference placement, not a train path. train_ok=false.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

GIB = 1024**3
MIB = 1024**2

# Phi-3/4 fused linears. Sizes match the INT8-from-BF16 sit (n_elem_linears=3,221,225,472).
PROFILES: Dict[str, Dict[str, Any]] = {
    "phi4-mini": {
        "name": "Phi-4-mini",
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 32,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 200064,
        "fused_qkv": True,
        "fused_gate_up": True,
        "note": "Phi-4-mini instruct. 3.8B. Product sit model.",
    },
    "llama-8b": {
        "name": "Llama-class 8B",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
        "fused_qkv": False,
        "fused_gate_up": False,
        "note": "8B-class. Tight on 12 GiB once KV and embed land.",
    },
    "qwen-7b": {
        "name": "Qwen-class 7B",
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 152064,
        "fused_qkv": False,
        "fused_gate_up": False,
        "note": "7B-class. Nested plug is the 12 GiB default.",
    },
}


@dataclass
class TensorSite:
    name: str
    n_elem: int
    kind: str  # linear | embed | norm
    layer: Optional[int] = None

    @property
    def hole_bytes(self) -> int:
        if self.kind != "linear":
            return 0
        return (self.n_elem + 1) // 2

    def plug_bytes(self, plug: str) -> int:
        if self.kind != "linear":
            return 0
        if plug == "bitplane":
            return self.n_elem * 2  # u16 low-12
        return (self.n_elem + 1) // 2  # nested 4-bit

    def absmax_bytes(self, blocksize: int = 64) -> int:
        if self.kind != "linear":
            return 0
        n_blocks = (self.n_elem + blocksize - 1) // blocksize
        return n_blocks * 4


def inventory(profile: Dict[str, Any]) -> List[TensorSite]:
    d = int(profile["hidden_size"])
    i = int(profile["intermediate_size"])
    l = int(profile["num_hidden_layers"])
    kv = int(profile["num_key_value_heads"])
    hd = int(profile["head_dim"])
    v = int(profile["vocab_size"])
    fused_qkv = bool(profile.get("fused_qkv"))
    fused_gu = bool(profile.get("fused_gate_up"))
    out: List[TensorSite] = []
    out.append(TensorSite("model.embed_tokens.weight", v * d, "embed"))
    out.append(TensorSite("model.norm.weight", d, "norm"))
    for li in range(l):
        pre = f"model.layers.{li}"
        out.append(TensorSite(f"{pre}.input_layernorm.weight", d, "norm", li))
        out.append(TensorSite(f"{pre}.post_attention_layernorm.weight", d, "norm", li))
        if fused_qkv:
            out.append(TensorSite(f"{pre}.self_attn.qkv_proj.weight", (d + 2 * kv * hd) * d, "linear", li))
        else:
            n_q = int(profile["num_attention_heads"]) * hd
            out.append(TensorSite(f"{pre}.self_attn.q_proj.weight", n_q * d, "linear", li))
            out.append(TensorSite(f"{pre}.self_attn.k_proj.weight", kv * hd * d, "linear", li))
            out.append(TensorSite(f"{pre}.self_attn.v_proj.weight", kv * hd * d, "linear", li))
        out.append(TensorSite(f"{pre}.self_attn.o_proj.weight", d * d, "linear", li))
        if fused_gu:
            out.append(TensorSite(f"{pre}.mlp.gate_up_proj.weight", (2 * i) * d, "linear", li))
        else:
            out.append(TensorSite(f"{pre}.mlp.gate_proj.weight", i * d, "linear", li))
            out.append(TensorSite(f"{pre}.mlp.up_proj.weight", i * d, "linear", li))
        out.append(TensorSite(f"{pre}.mlp.down_proj.weight", d * i, "linear", li))
    out.append(TensorSite("lm_head.weight", v * d, "embed"))
    return out


def kv_bytes(profile: Dict[str, Any], ctx: int, batch: int) -> int:
    l = int(profile["num_hidden_layers"])
    kv = int(profile["num_key_value_heads"])
    hd = int(profile["head_dim"])
    # K+V, fp16
    return batch * ctx * l * kv * hd * 2 * 2


def act_bytes(profile: Dict[str, Any], ctx: int, batch: int) -> int:
    """Working activations. Decode (ctx used as 1 token) vs prefill."""
    d = int(profile["hidden_size"])
    i = int(profile["intermediate_size"])
    # decode-oriented: last-token residual + qkv + mlp scratch, fp16, ~8 buffers
    seq = max(1, min(ctx, 8)) if ctx <= 8 else 1
    # Prefer decode envelope; prefill is called out separately.
    return batch * max(seq, 1) * (d * 4 + i * 2) * 2


def prefill_act_bytes(profile: Dict[str, Any], ctx: int, batch: int) -> int:
    d = int(profile["hidden_size"])
    i = int(profile["intermediate_size"])
    return batch * ctx * (d * 4 + i * 2) * 2


def map_profile(
    profile_id: str,
    *,
    plug: str = "nested",
    vram_gib: float = 12.0,
    ctx: int = 4096,
    batch: int = 1,
    ring_layers: int = 1,
    embed_in_vram: bool = True,
    blocksize: int = 64,
    compute: str = "inflate_then_int8",
) -> Dict[str, Any]:
    if profile_id not in PROFILES:
        raise ValueError(f"unknown profile {profile_id}")
    if plug not in ("nested", "bitplane"):
        raise ValueError("plug must be nested or bitplane")
    p = PROFILES[profile_id]
    sites = inventory(p)
    linears = [s for s in sites if s.kind == "linear"]
    embeds = [s for s in sites if s.kind == "embed"]
    norms = [s for s in sites if s.kind == "norm"]

    hole = sum(s.hole_bytes for s in linears)
    plug_all = sum(s.plug_bytes(plug) for s in linears)
    absmax = sum(s.absmax_bytes(blocksize) for s in linears)
    # nested needs absmax; bitplane inflate has no scale table
    if plug == "bitplane":
        absmax = 0
    int8_scale = sum(s.absmax_bytes(blocksize) for s in linears)  # always: INT8 GEMM scales
    embed_b = sum(s.n_elem * 2 for s in embeds)  # BF16 copy
    norm_b = sum(s.n_elem * 2 for s in norms)
    n_layers = int(p["num_hidden_layers"])
    per_layer_plug = plug_all / max(n_layers, 1)
    ring = int(round(per_layer_plug * ring_layers))
    kv = kv_bytes(p, ctx, batch)
    acts = act_bytes(p, ctx, batch)
    prefill = prefill_act_bytes(p, ctx, batch)

    resident = hole + absmax + int8_scale + (embed_b if embed_in_vram else 0) + norm_b + ring + kv + acts
    budget = int(vram_gib * GIB)
    headroom = budget - resident

    layers: List[Dict[str, Any]] = []
    for li in range(n_layers):
        ls = [s for s in linears if s.layer == li]
        layers.append(
            {
                "index": li,
                "n_elem": sum(s.n_elem for s in ls),
                "hole_bytes": sum(s.hole_bytes for s in ls),
                "plug_bytes": sum(s.plug_bytes(plug) for s in ls),
                "sites": [s.name.split(".")[-2] + "." + s.name.split(".")[-1] for s in ls],
            }
        )

    full_int8 = sum(s.n_elem for s in linears) + int8_scale  # I8 weights + scales
    full_int8 += (embed_b if embed_in_vram else 0) + norm_b + kv + acts
    nf4_only = hole + absmax + (embed_b if embed_in_vram else 0) + norm_b + kv + acts

    return {
        "schema": "vram_map_v1",
        "profile": profile_id,
        "profile_name": p["name"],
        "plug": plug,
        "plug_bits": 12 if plug == "bitplane" else 4,
        "hole_bits": 4,
        "compute": compute,
        "gpu": {"name": "budget", "vram_gib": vram_gib},
        "ctx": ctx,
        "batch": batch,
        "ring_layers": ring_layers,
        "embed_in_vram": embed_in_vram,
        "blocksize": blocksize,
        "n_linears": len(linears),
        "n_elem_linears": sum(s.n_elem for s in linears),
        "n_elem_embed": sum(s.n_elem for s in embeds),
        "budget": {
            "hole_bytes": hole,
            "plug_host_bytes": plug_all - ring,
            "plug_ring_bytes": ring,
            "plug_all_bytes": plug_all,
            "absmax_bytes": absmax,
            "int8_scale_bytes": int8_scale,
            "embed_bytes": embed_b if embed_in_vram else 0,
            "embed_host_bytes": 0 if embed_in_vram else embed_b,
            "norm_bytes": norm_b,
            "kv_bytes": kv,
            "act_decode_bytes": acts,
            "act_prefill_bytes": prefill,
            "resident_vram_bytes": resident,
            "budget_bytes": budget,
            "headroom_bytes": headroom,
            "fits": headroom >= 0,
            "fits_prefill": (budget - (resident - acts + prefill)) >= 0,
        },
        "compare": {
            "full_int8_resident_bytes": full_int8,
            "nf4_hole_only_resident_bytes": nf4_only,
            "mapped_saves_vs_int8_bytes": full_int8 - resident,
        },
        "ring": {
            "layers_ahead": ring_layers,
            "stream": "cudaMemcpyAsync pinned host → VRAM ring",
            "inner_loop_pcie": False,
        },
        "layers": layers,
        "note": (
            "VRAM keeps the 4-bit hole + one-layer plug ring. "
            "Host holds the rest of the plug. Inflate then INT8 GEMM. "
            "nested ≈ INT8 RMSE; bitplane = bit-exact BF16 then INT8. "
            "Not train_ok."
        ),
        "train_ok": False,
        "measured_omega": False,
    }


def map_from_modules(
    modules: Sequence[Dict[str, Any]],
    *,
    plug: str,
    vram_gib: float = 12.0,
    profile_id: str = "from-pin",
    ctx: int = 4096,
    batch: int = 1,
    ring_layers: int = 1,
    blocksize: int = 64,
) -> Dict[str, Any]:
    """Build a map from converter module reports (actual n_elem)."""
    sites = [
        TensorSite(str(m.get("src") or m.get("name") or f"m{i}"), int(m["n_elem"]), "linear")
        for i, m in enumerate(modules)
        if int(m.get("n_elem") or 0) > 0
    ]
    hole = sum(s.hole_bytes for s in sites)
    plug_all = sum(s.plug_bytes(plug) for s in sites)
    absmax = 0 if plug == "bitplane" else sum(s.absmax_bytes(blocksize) for s in sites)
    int8_scale = sum(s.absmax_bytes(blocksize) for s in sites)
    n_layers = max(1, len({_layer_of(s.name) for s in sites if _layer_of(s.name) is not None}))
    ring = int(round((plug_all / n_layers) * ring_layers))
    resident = hole + absmax + int8_scale + ring
    budget = int(vram_gib * GIB)
    return {
        "schema": "vram_map_v1",
        "profile": profile_id,
        "plug": plug,
        "n_converted": len(sites),
        "n_elem_linears": sum(s.n_elem for s in sites),
        "budget": {
            "hole_bytes": hole,
            "plug_host_bytes": max(0, plug_all - ring),
            "plug_ring_bytes": ring,
            "plug_all_bytes": plug_all,
            "absmax_bytes": absmax,
            "int8_scale_bytes": int8_scale,
            "resident_weights_bytes": resident,
            "budget_bytes": budget,
            "headroom_bytes": budget - resident,
            "fits_weights": (budget - resident) >= 0,
        },
        "train_ok": False,
    }


def _layer_of(name: str) -> Optional[int]:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def gib(n: int) -> float:
    return n / GIB
