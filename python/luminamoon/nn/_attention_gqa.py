from __future__ import annotations

import threading
from typing import Any

from ._compat import _is_torch_compiling, torch

_CAUSAL_MASK_LOCK = threading.Lock()
_CAUSAL_MASK_CACHE: dict[tuple[int, int, int, Any], Any] = {}


def _get_causal_mask(q_start: int, q_end: int, k_limit: int, device: Any) -> Any:
    if _is_torch_compiling():
        q_idx = torch.arange(q_start, q_end, device=device).unsqueeze(1)
        k_idx = torch.arange(0, k_limit, device=device).unsqueeze(0)
        return q_idx < k_idx
    key = (q_start, q_end, k_limit, device)
    with _CAUSAL_MASK_LOCK:
        mask = _CAUSAL_MASK_CACHE.get(key)
    if mask is None:
        q_idx = torch.arange(q_start, q_end, device=device).unsqueeze(1)
        k_idx = torch.arange(0, k_limit, device=device).unsqueeze(0)
        mask = q_idx < k_idx
        with _CAUSAL_MASK_LOCK:
            if len(_CAUSAL_MASK_CACHE) >= 64:
                _CAUSAL_MASK_CACHE.clear()
            _CAUSAL_MASK_CACHE[key] = mask
    return mask


def _gqa_matmul_qk(q: Any, k: Any) -> Any:
    h_q, h_kv = q.shape[1], k.shape[1]
    k_match = k.to(q.dtype) if hasattr(k, "dtype") and k.dtype != q.dtype else k
    if h_q == h_kv:
        return q @ k_match.transpose(-2, -1)
    B, _, N_Q, D = q.shape
    g = h_q // h_kv
    return (q.reshape(B, h_kv, g * N_Q, D) @ k_match.transpose(-2, -1)).reshape(B, h_q, N_Q, k.shape[2])


def _gqa_matmul_pv(p: Any, v: Any) -> Any:
    h_q, h_kv = p.shape[1], v.shape[1]
    v_match = v.to(p.dtype) if hasattr(v, "dtype") and v.dtype != p.dtype else v
    if h_q == h_kv:
        return p @ v_match
    B, _, N_Q, N_K = p.shape
    g = h_q // h_kv
    return (p.reshape(B, h_kv, g * N_Q, N_K) @ v_match).reshape(B, h_q, N_Q, v.shape[-1])


def _gqa_matmul_do_vt(do: Any, v: Any) -> Any:
    h_q, h_kv = do.shape[1], v.shape[1]
    v_match = v.to(do.dtype) if hasattr(v, "dtype") and v.dtype != do.dtype else v
    if h_q == h_kv:
        return do @ v_match.transpose(-2, -1)
    B, _, N_Q, D_V = do.shape
    g = h_q // h_kv
    return (do.reshape(B, h_kv, g * N_Q, D_V) @ v_match.transpose(-2, -1)).reshape(B, h_q, N_Q, v.shape[2])


def _gqa_matmul_pt_do(p: Any, do: Any, h_kv: int) -> Any:
    h_q = p.shape[1]
    do_match = do.to(p.dtype) if hasattr(do, "dtype") and do.dtype != p.dtype else do
    if h_q == h_kv:
        return p.transpose(-2, -1) @ do_match
    B, _, N_Q, N_K = p.shape
    g = h_q // h_kv
    return p.reshape(B, h_kv, g * N_Q, N_K).transpose(-2, -1) @ do_match.reshape(B, h_kv, g * N_Q, do.shape[-1])


def _gqa_matmul_ds_k(ds: Any, k: Any) -> Any:
    h_q, h_kv = ds.shape[1], k.shape[1]
    k_match = k.to(ds.dtype) if hasattr(k, "dtype") and k.dtype != ds.dtype else k
    if h_q == h_kv:
        return ds @ k_match
    B, _, N_Q, N_K = ds.shape
    g = h_q // h_kv
    return (ds.reshape(B, h_kv, g * N_Q, N_K) @ k_match).reshape(B, h_q, N_Q, k.shape[-1])


def _gqa_matmul_dst_q(ds: Any, q: Any, h_kv: int) -> Any:
    h_q = ds.shape[1]
    q_match = q.to(ds.dtype) if hasattr(q, "dtype") and q.dtype != ds.dtype else q
    if h_q == h_kv:
        return ds.transpose(-2, -1) @ q_match
    B, _, N_Q, N_K = ds.shape
    g = h_q // h_kv
    return ds.reshape(B, h_kv, g * N_Q, N_K).transpose(-2, -1) @ q_match.reshape(B, h_kv, g * N_Q, q.shape[-1])


def _slice_attn_mask(attn_mask: Any, q_start: int, q_end: int, k_limit: int) -> Any:
    if attn_mask.ndim == 1:
        m_view = attn_mask.view(1, 1, 1, -1)
    elif attn_mask.ndim == 2:
        m_view = attn_mask.unsqueeze(0).unsqueeze(0)
    elif attn_mask.ndim == 3:
        m_view = attn_mask.unsqueeze(1)
    else:
        m_view = attn_mask
    q_sl = slice(q_start, q_end) if m_view.shape[-2] > 1 else slice(0, 1)
    k_sl = slice(0, k_limit) if m_view.shape[-1] > 1 else slice(0, 1)
    return m_view[:, :, q_sl, k_sl]


def _accum_attn_mask_grad(
    d_mask: Any,
    ds_direct: Any,
    q_start: int,
    q_end: int,
    k_limit: int,
) -> None:
    if d_mask.ndim == 1:
        d_view = d_mask.view(1, 1, 1, -1)
    elif d_mask.ndim == 2:
        d_view = d_mask.unsqueeze(0).unsqueeze(0)
    elif d_mask.ndim == 3:
        d_view = d_mask.unsqueeze(1)
    else:
        d_view = d_mask
    reduce_dims = tuple(ax for ax in range(4) if d_view.shape[ax] == 1)
    grad_slice = ds_direct.sum(dim=reduce_dims, keepdim=True) if reduce_dims else ds_direct
    q_sl = slice(q_start, q_end) if d_view.shape[-2] > 1 else slice(0, 1)
    k_sl = slice(0, k_limit) if d_view.shape[-1] > 1 else slice(0, 1)
    d_view[:, :, q_sl, k_sl] += grad_slice
