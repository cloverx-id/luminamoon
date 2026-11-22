from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

from ._base import HAS_TRITON, _HAS_TORCH, tl, torch, triton

_SM_COUNT_CACHE: dict[int, int] = {}

if TYPE_CHECKING:
    _moe_m_grouped_gemm_kernel: Any = None
    _moe_k_grouped_dw_kernel: Any = None
    _ra_gated_packed_fwd_kernel: Any = None
    _ra_gated_packed_bwd_kernel: Any = None
elif HAS_TRITON:
    _M_GROUPED_CONFIGS = [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8, "CTAS_PER_SM": 2},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8, "CTAS_PER_SM": 2},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8, "CTAS_PER_SM": 2},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8, "CTAS_PER_SM": 2},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8, "CTAS_PER_SM": 2},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8, "CTAS_PER_SM": 1},
            num_warps=8,
            num_stages=3,
        ),
    ]

    _K_GROUPED_DW_CONFIGS = [
        triton.Config({"BLOCK_K": 64, "BLOCK_N": 64, "BLOCK_M": 64, "GROUP_SIZE_K": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K": 64, "BLOCK_N": 128, "BLOCK_M": 64, "GROUP_SIZE_K": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K": 128, "BLOCK_N": 64, "BLOCK_M": 64, "GROUP_SIZE_K": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_K": 64, "BLOCK_N": 64, "BLOCK_M": 32, "GROUP_SIZE_K": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K": 128, "BLOCK_N": 128, "BLOCK_M": 32, "GROUP_SIZE_K": 8}, num_warps=8, num_stages=3),
    ]

    @triton.autotune(configs=_M_GROUPED_CONFIGS, key=["K", "N", "E"])
    @triton.jit
    def _moe_m_grouped_gemm_kernel(
        a_ptr,
        w_ptr,
        c_ptr,
        offs_ptr,
        gather_idx_ptr,
        K,
        N,
        E,
        stride_am,
        stride_ak,
        stride_we,
        stride_wk,
        stride_wn,
        stride_cm,
        stride_cn,
        HAS_GATHER: tl.constexpr,
        EVEN_K: tl.constexpr,
        DOT_PRECISION: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        CTAS_PER_SM: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_progs = tl.num_programs(0)
        num_n_tiles = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_M * num_n_tiles

        tile_id = pid
        tile_base = 0
        prev_end = 0
        for e in range(0, E):
            end_e = tl.load(offs_ptr + e).to(tl.int32)
            m_e = end_e - prev_end
            if m_e > 0:
                num_m_tiles = tl.cdiv(m_e, BLOCK_M)
                tile_end = tile_base + num_m_tiles * num_n_tiles
                w_base = w_ptr + tl.cast(e, tl.int64) * stride_we
                while tile_id < tile_end:
                    local_tile = tile_id - tile_base
                    group_id = local_tile // num_pid_in_group
                    first_pid_m = group_id * GROUP_SIZE_M
                    group_size_m = tl.minimum(num_m_tiles - first_pid_m, GROUP_SIZE_M)
                    rem = local_tile % num_pid_in_group
                    pid_m = first_pid_m + (rem % group_size_m)
                    pid_n = rem // group_size_m

                    offs_m = prev_end + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                    mask_m = offs_m < end_e
                    mask_n = offs_n < N
                    if HAS_GATHER:
                        a_rows = tl.load(gather_idx_ptr + offs_m, mask=mask_m, other=0).to(tl.int64)
                    else:
                        a_rows = offs_m.to(tl.int64)

                    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                    for k0 in range(0, K, BLOCK_K):
                        offs_k = k0 + tl.arange(0, BLOCK_K)
                        a_ptrs = a_ptr + a_rows[:, None] * stride_am + offs_k[None, :].to(tl.int64) * stride_ak
                        w_ptrs = w_base + offs_k[:, None].to(tl.int64) * stride_wk + offs_n[None, :].to(tl.int64) * stride_wn
                        if EVEN_K:
                            a_blk = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
                            w_blk = tl.load(w_ptrs, mask=mask_n[None, :], other=0.0)
                        else:
                            mask_k = offs_k < K
                            a_blk = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                            w_blk = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                        acc = tl.dot(a_blk, w_blk, acc=acc, input_precision=DOT_PRECISION)

                    c_ptrs = c_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :].to(tl.int64) * stride_cn
                    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])
                    tile_id += num_progs
                tile_base = tile_end
            prev_end = end_e

    @triton.autotune(configs=_K_GROUPED_DW_CONFIGS, key=["K", "N", "E"])
    @triton.jit
    def _moe_k_grouped_dw_kernel(
        x_ptr,
        dy_ptr,
        dw_ptr,
        offs_ptr,
        gather_idx_ptr,
        K,
        N,
        E,
        stride_xm,
        stride_xk,
        stride_dym,
        stride_dyn,
        stride_dwe,
        stride_dwk,
        stride_dwn,
        HAS_GATHER: tl.constexpr,
        DOT_PRECISION: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        GROUP_SIZE_K: tl.constexpr,
    ):
        pid_kn = tl.program_id(0)
        e = tl.program_id(1)

        num_k_tiles = tl.cdiv(K, BLOCK_K)
        num_n_tiles = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_K * num_n_tiles
        group_id = pid_kn // num_pid_in_group
        first_pid_k = group_id * GROUP_SIZE_K
        group_size_k = tl.minimum(num_k_tiles - first_pid_k, GROUP_SIZE_K)
        rem = pid_kn % num_pid_in_group
        pid_k = first_pid_k + (rem % group_size_k)
        pid_n = rem // group_size_k

        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_k = offs_k < K
        mask_n = offs_n < N

        start_e = tl.where(e > 0, tl.load(offs_ptr + tl.maximum(e - 1, 0)).to(tl.int32), 0)
        end_e = tl.load(offs_ptr + e).to(tl.int32)

        acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
        for m0 in range(start_e, end_e, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            mask_m = offs_m < end_e
            if HAS_GATHER:
                x_rows = tl.load(gather_idx_ptr + offs_m, mask=mask_m, other=0).to(tl.int64)
            else:
                x_rows = offs_m.to(tl.int64)
            x_ptrs = x_ptr + x_rows[None, :] * stride_xm + offs_k[:, None].to(tl.int64) * stride_xk
            dy_ptrs = dy_ptr + offs_m[:, None].to(tl.int64) * stride_dym + offs_n[None, :].to(tl.int64) * stride_dyn
            x_blk = tl.load(x_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            dy_blk = tl.load(dy_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(x_blk, dy_blk, acc=acc, input_precision=DOT_PRECISION)

        dw_ptrs = (
            dw_ptr
            + tl.cast(e, tl.int64) * stride_dwe
            + offs_k[:, None].to(tl.int64) * stride_dwk
            + offs_n[None, :].to(tl.int64) * stride_dwn
        )
        tl.store(dw_ptrs, acc.to(dw_ptr.dtype.element_ty), mask=mask_k[:, None] & mask_n[None, :])

    @triton.jit
    def _ra_gated_packed_fwd_kernel(
        gu_ptr,
        out_ptr,
        M,
        H,
        stride_gum,
        stride_om,
        b_val: tl.constexpr,
        IS_SWIGLU: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        col_tile = tl.program_id(1)
        offs_h = col_tile * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_h = (row < M) & (offs_h < H)

        gu_base = gu_ptr + row * stride_gum
        z = tl.load(gu_base + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        u = tl.load(gu_base + H + offs_h, mask=mask_h, other=0.0).to(tl.float32)

        abs_z = tl.abs(z)
        b_term = b_val * (b_val + abs_z)
        inv_d = 1.0 / (b_term + z * z)
        if IS_SWIGLU:
            q = (0.5 * b_val * b_val) * inv_d
            s = z * tl.where(z >= 0.0, 1.0 - q, q)
        else:
            s = (b_term * z) * inv_d
        tl.store(out_ptr + row * stride_om + offs_h, (s * u).to(out_ptr.dtype.element_ty), mask=mask_h)

    @triton.jit
    def _ra_gated_packed_bwd_kernel(
        grad_out_ptr,
        gu_ptr,
        dgu_ptr,
        M,
        H,
        stride_gm,
        stride_gum,
        stride_dgum,
        b_val: tl.constexpr,
        IS_SWIGLU: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        col_tile = tl.program_id(1)
        offs_h = col_tile * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_h = (row < M) & (offs_h < H)

        g = tl.load(grad_out_ptr + row * stride_gm + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        gu_base = gu_ptr + row * stride_gum
        z = tl.load(gu_base + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        u = tl.load(gu_base + H + offs_h, mask=mask_h, other=0.0).to(tl.float32)

        abs_z = tl.abs(z)
        b_term = b_val * (b_val + abs_z)
        inv_d = 1.0 / (b_term + z * z)
        if IS_SWIGLU:
            b2 = b_val * b_val
            q = (0.5 * b2) * inv_d
            is_pos = z >= 0.0
            sigma = tl.where(is_pos, 1.0 - q, q)
            s = z * sigma
            ds_dz = tl.where(is_pos, sigma + z * q * (b_val + 2.0 * abs_z) * inv_d, q * (b2 - z * z) * inv_d)
        else:
            s = (b_term * z) * inv_d
            ds_dz = (b_val * b_val * b_val) * (b_val + 2.0 * abs_z) * (inv_d * inv_d)

        dgu_base = dgu_ptr + row * stride_dgum
        tl.store(dgu_base + offs_h, (g * u * ds_dz).to(dgu_ptr.dtype.element_ty), mask=mask_h)
        tl.store(dgu_base + H + offs_h, (g * s).to(dgu_ptr.dtype.element_ty), mask=mask_h)
else:
    _moe_m_grouped_gemm_kernel = None
    _moe_k_grouped_dw_kernel = None
    _ra_gated_packed_fwd_kernel = None
    _ra_gated_packed_bwd_kernel = None


def _get_sm_count(device: Any) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    sm = _SM_COUNT_CACHE.get(idx)
    if sm is None:
        sm = max(1, int(torch.cuda.get_device_properties(idx).multi_processor_count))
        _SM_COUNT_CACHE[idx] = sm
    return sm


def _launch_m_grouped_gemm(a: Any, w: Any, offs: Any, gather_idx: Optional[Any] = None) -> Any:
    E, K, N = w.shape
    n_tok = gather_idx.shape[0] if gather_idx is not None else a.shape[0]
    out = torch.empty((n_tok, N), device=a.device, dtype=a.dtype)
    if n_tok == 0 or E == 0 or N == 0:
        return out
    if K == 0:
        return out.zero_()
    sm_count = _get_sm_count(a.device)
    grid = lambda meta: (sm_count * meta["CTAS_PER_SM"],)
    _moe_m_grouped_gemm_kernel[grid](
        a,
        w,
        out,
        offs,
        gather_idx if gather_idx is not None else offs,
        K,
        N,
        E,
        a.stride(0),
        a.stride(1),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        out.stride(0),
        out.stride(1),
        HAS_GATHER=gather_idx is not None,
        EVEN_K=(K % 64 == 0),
        DOT_PRECISION="ieee",
    )
    return out


def _launch_k_grouped_dw(
    x: Any,
    dy: Any,
    offs: Any,
    out_dtype: Any,
    gather_idx: Optional[Any] = None,
) -> Any:
    E = offs.shape[0]
    K = x.shape[1]
    N = dy.shape[1]
    dw = torch.empty((E, K, N), device=x.device, dtype=out_dtype)
    if E == 0 or K == 0 or N == 0:
        return dw
    if dy.shape[0] == 0:
        return dw.zero_()
    grid = lambda meta: (triton.cdiv(K, meta["BLOCK_K"]) * triton.cdiv(N, meta["BLOCK_N"]), E)
    _moe_k_grouped_dw_kernel[grid](
        x,
        dy,
        dw,
        offs,
        gather_idx if gather_idx is not None else offs,
        K,
        N,
        E,
        x.stride(0),
        x.stride(1),
        dy.stride(0),
        dy.stride(1),
        dw.stride(0),
        dw.stride(1),
        dw.stride(2),
        HAS_GATHER=gather_idx is not None,
        DOT_PRECISION="ieee",
    )
    return dw


_TritonGroupedMMFunction: Any = None
_TritonRaGatedPackedFunction: Any = None

if _HAS_TORCH and torch is not None:

    class _TritonGroupedMMFunctionImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any, w: Any, offs: Any, gather_idx: Optional[Any]) -> Any:
            x_c = x if x.is_contiguous() else x.contiguous()
            w_c = w if w.is_contiguous() else w.contiguous()
            offs_c = offs if (offs.dtype == torch.int32 and offs.is_contiguous()) else offs.to(torch.int32).contiguous()
            g_c = gather_idx.contiguous() if gather_idx is not None else None
            ctx.has_gather = g_c is not None
            if g_c is not None:
                ctx.save_for_backward(x_c, w_c, offs_c, g_c)
            else:
                ctx.save_for_backward(x_c, w_c, offs_c)
            return _launch_m_grouped_gemm(x_c, w_c, offs_c, gather_idx=g_c)

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Tuple[Optional[Any], Optional[Any], None, None]:  # type: ignore[override]
            x_c, w_c, offs_c = ctx.saved_tensors[0], ctx.saved_tensors[1], ctx.saved_tensors[2]
            g_c = ctx.saved_tensors[3] if ctx.has_gather else None
            dy_in = grad_out if grad_out.dtype == w_c.dtype else grad_out.to(w_c.dtype)
            dy_c = dy_in if dy_in.is_contiguous() else dy_in.contiguous()
            dx = None
            if ctx.needs_input_grad[0]:
                dx_perm = _launch_m_grouped_gemm(dy_c, w_c.transpose(-1, -2), offs_c)
                if g_c is not None:
                    dx = torch.zeros_like(x_c)
                    dx.index_add_(0, g_c, dx_perm)
                else:
                    dx = dx_perm
            dw = _launch_k_grouped_dw(x_c, dy_c, offs_c, w_c.dtype, gather_idx=g_c) if ctx.needs_input_grad[1] else None
            return dx, dw, None, None

    _TritonGroupedMMFunction = _TritonGroupedMMFunctionImpl

    class _TritonRaGatedPackedFunctionImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, gu: Any, b: float, is_swiglu: bool) -> Any:
            gu_c = gu if gu.is_contiguous() else gu.contiguous()
            M, two_h = gu_c.shape
            H = two_h // 2
            out = torch.empty((M, H), device=gu_c.device, dtype=gu_c.dtype)
            if M > 0 and H > 0:
                BLOCK_H = min(1024, max(64, triton.next_power_of_2(H)))
                _ra_gated_packed_fwd_kernel[(M, triton.cdiv(H, BLOCK_H))](
                    gu_c, out, M, H, gu_c.stride(0), out.stride(0), b_val=float(b), IS_SWIGLU=is_swiglu, BLOCK_H=BLOCK_H
                )
            ctx.b = float(b)
            ctx.is_swiglu = bool(is_swiglu)
            ctx.save_for_backward(gu_c)
            return out

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Tuple[Optional[Any], None, None]:  # type: ignore[override]
            (gu_c,) = ctx.saved_tensors
            if not ctx.needs_input_grad[0]:
                return None, None, None
            g_c = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
            M, two_h = gu_c.shape
            H = two_h // 2
            dgu = torch.empty_like(gu_c)
            if M > 0 and H > 0:
                BLOCK_H = min(1024, max(64, triton.next_power_of_2(H)))
                _ra_gated_packed_bwd_kernel[(M, triton.cdiv(H, BLOCK_H))](
                    g_c, gu_c, dgu, M, H, g_c.stride(0), gu_c.stride(0), dgu.stride(0),
                    b_val=ctx.b, IS_SWIGLU=ctx.is_swiglu, BLOCK_H=BLOCK_H,
                )
            return dgu, None, None

    _TritonRaGatedPackedFunction = _TritonRaGatedPackedFunctionImpl


def can_use_triton_grouped_gemm(x: Any, w: Any) -> bool:
    """Returns True when Triton grouped GEMM can safely execute on (x, w)."""
    return bool(
        HAS_TRITON
        and _HAS_TORCH
        and isinstance(x, torch.Tensor)
        and isinstance(w, torch.Tensor)
        and x.is_cuda
        and w.is_cuda
        and x.ndim == 2
        and w.ndim == 3
        and x.dtype == w.dtype
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    )


def triton_grouped_gemm(x: Any, w: Any, offs: Any, gather_idx: Optional[Any] = None) -> Optional[Any]:
    """Dispatches 2-kernel persistent Triton grouped GEMM with optional zero-copy gather."""
    if not can_use_triton_grouped_gemm(x, w):
        return None
    if torch.is_grad_enabled() and (x.requires_grad or w.requires_grad) and _TritonGroupedMMFunction is not None:
        return _TritonGroupedMMFunction.apply(x, w, offs, gather_idx)
    x_c = x if x.is_contiguous() else x.contiguous()
    w_c = w if w.is_contiguous() else w.contiguous()
    offs_c = offs if (offs.dtype == torch.int32 and offs.is_contiguous()) else offs.to(torch.int32).contiguous()
    g_c = gather_idx.contiguous() if gather_idx is not None else None
    return _launch_m_grouped_gemm(x_c, w_c, offs_c, gather_idx=g_c)


def triton_ra_gated_packed(gu: Any, b: float = 1.0, *, is_swiglu: bool = True) -> Optional[Any]:
    """Fused RaSwiGLU/RaGLU directly over packed [M, 2H] gate-up tensor with zero contiguous copies."""
    if not (
        HAS_TRITON
        and _HAS_TORCH
        and isinstance(gu, torch.Tensor)
        and gu.is_cuda
        and gu.ndim == 2
        and (gu.shape[1] % 2 == 0)
        and gu.dtype in (torch.bfloat16, torch.float16, torch.float32)
    ):
        return None
    if torch.is_grad_enabled() and gu.requires_grad and _TritonRaGatedPackedFunction is not None:
        return _TritonRaGatedPackedFunction.apply(gu, float(b), bool(is_swiglu))
    gu_c = gu if gu.is_contiguous() else gu.contiguous()
    M, two_h = gu_c.shape
    H = two_h // 2
    out = torch.empty((M, H), device=gu_c.device, dtype=gu_c.dtype)
    if M > 0 and H > 0:
        BLOCK_H = min(1024, max(64, triton.next_power_of_2(H)))
        _ra_gated_packed_fwd_kernel[(M, triton.cdiv(H, BLOCK_H))](
            gu_c, out, M, H, gu_c.stride(0), out.stride(0), b_val=float(b), IS_SWIGLU=bool(is_swiglu), BLOCK_H=BLOCK_H
        )
    return out


__all__ = ["can_use_triton_grouped_gemm", "triton_grouped_gemm", "triton_ra_gated_packed"]
