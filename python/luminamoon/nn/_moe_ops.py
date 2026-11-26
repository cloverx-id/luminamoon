"""Internal routing and small-batch decode helpers for RaMoE."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from ._compat import torch
from .activation import ra_glu, ra_swiglu

# Small-batch token count threshold for zero-sync batched GEMM decode (B=1..4, L=1).
SMALL_BATCH_DECODE_MAX_TOKENS = 4


def _can_use_grouped_mm(x: Any, w_gate_up: Any, w_down: Any) -> bool:
    if not (hasattr(torch, "_grouped_mm") and x.is_cuda):
        return False
    if x.dtype not in (torch.bfloat16, torch.float16):
        return False
    if w_gate_up.dtype != x.dtype or w_down.dtype != x.dtype:
        return False
    if (x.shape[-1] % 8 != 0) or (w_gate_up.shape[-1] % 8 != 0) or (w_down.shape[-1] % 8 != 0):
        return False
    cap = torch.cuda.get_device_capability(x.device)
    # PyTorch native CUTLASS _grouped_mm requires Hopper (sm_90+); sm_80..89 & ROCm use Triton grouped GEMM.
    return cap[0] >= 9


def _sort_routing_indices(top_indices: Any, top_weights: Any, num_experts: int) -> Tuple[Any, Any, Any, Any]:
    flat_experts = top_indices.reshape(-1)
    perm = torch.argsort(flat_experts, stable=True)
    token_indices = torch.div(perm, top_indices.shape[-1], rounding_mode="floor")
    sorted_weights = top_weights.reshape(-1).index_select(0, perm)
    sorted_experts = flat_experts.index_select(0, perm)
    counts = torch.bincount(flat_experts.to(torch.int64), minlength=int(num_experts))
    return token_indices, sorted_weights, counts, sorted_experts


def _small_batch_routed_forward(
    x_2d: Any,
    top_indices: Any,
    top_weights: Any,
    w_gu: Any,
    w_dn: Any,
    b_gu: Optional[Any],
    b_dn: Optional[Any],
    *,
    hidden_features: int,
    expert_type: str,
    b: float,
) -> Any:
    """Zero-sync batched GEMM path for small token counts (e.g. B=1..4 autoregressive decode).

    Avoids argsort, bincount, cumsum, host `.tolist()` syncs, and 64-row grouped-GEMM tile
    padding by gathering the selected top-k expert weights directly into two `torch.bmm` calls,
    keeping tensor shapes static for `torch.compile(fullgraph=True, mode="reduce-overhead")`.
    """
    num_tokens, top_k = top_indices.shape
    flat_idx = top_indices.reshape(-1)
    mk = num_tokens * top_k
    x_sel = x_2d.unsqueeze(1).expand(num_tokens, top_k, -1).reshape(mk, 1, x_2d.shape[-1])
    w_gu_sel = w_gu.index_select(0, flat_idx)
    if w_gu_sel.dtype != x_2d.dtype:
        w_gu_sel = w_gu_sel.to(x_2d.dtype)
    gu = torch.bmm(x_sel, w_gu_sel).squeeze(1)
    if b_gu is not None:
        b_gu_sel = b_gu.index_select(0, flat_idx)
        gu = gu + (b_gu_sel if b_gu_sel.dtype == x_2d.dtype else b_gu_sel.to(x_2d.dtype))

    gate, up = gu.split(hidden_features, dim=-1)
    act_fn = ra_swiglu if expert_type == "ra_swiglu" else ra_glu
    h = act_fn(gate.contiguous(), up.contiguous(), b=b)

    w_dn_sel = w_dn.index_select(0, flat_idx)
    if w_dn_sel.dtype != h.dtype:
        w_dn_sel = w_dn_sel.to(h.dtype)
    out_k = torch.bmm(h.unsqueeze(1), w_dn_sel).squeeze(1)
    if b_dn is not None:
        b_dn_sel = b_dn.index_select(0, flat_idx)
        out_k = out_k + (b_dn_sel if b_dn_sel.dtype == out_k.dtype else b_dn_sel.to(out_k.dtype))

    weights = top_weights.to(out_k.dtype).reshape(num_tokens, top_k, 1)
    return (out_k.reshape(num_tokens, top_k, -1) * weights).sum(dim=1)
