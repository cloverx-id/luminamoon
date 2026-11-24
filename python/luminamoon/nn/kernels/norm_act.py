from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

from ._base import HAS_TRITON, _HAS_TORCH, tl, torch, triton

if TYPE_CHECKING:
    _ra_norm_fwd_kernel: Any = None
    _ra_norm_bwd_kernel: Any = None
    _ra_norm_bwd_reduce_kernel: Any = None
    _ra_gated_fwd_kernel: Any = None
    _ra_gated_bwd_kernel: Any = None
    _ra_tanh_fwd_kernel: Any = None
    _ra_tanh_bwd_kernel: Any = None
elif HAS_TRITON:
    def _prune_ra_norm_configs(configs: Any, named_args: Any, **kwargs: Any) -> Any:
        block_size = int(named_args.get("BLOCK_SIZE", kwargs.get("BLOCK_SIZE", 512)))
        kept = [
            c for c in configs
            if not (c.num_warps == 1 and block_size > 1024)
            and not (c.num_warps == 2 and block_size > 2048)
        ]
        return kept if kept else configs[-1:]

    _RA_NORM_FWD_CONFIGS = [
        triton.Config({"ROWS_PER_CTA": 1}, num_warps=1),
        triton.Config({"ROWS_PER_CTA": 1}, num_warps=2),
        triton.Config({"ROWS_PER_CTA": 4}, num_warps=2),
        triton.Config({"ROWS_PER_CTA": 8}, num_warps=2),
        triton.Config({"ROWS_PER_CTA": 4}, num_warps=4),
        triton.Config({"ROWS_PER_CTA": 1}, num_warps=4),
        triton.Config({"ROWS_PER_CTA": 1}, num_warps=8),
    ]
    _RA_NORM_BWD_CONFIGS = [
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ]

    @triton.autotune(
        configs=_RA_NORM_FWD_CONFIGS,
        key=["D"],
        prune_configs_by={"early_config_prune": _prune_ra_norm_configs},
    )
    @triton.jit
    def _ra_norm_fwd_kernel(
        x_ptr,
        y_ptr,
        weight_ptr,
        bias_ptr,
        inv_denom_ptr,
        n_rows,
        D: tl.constexpr,
        eps: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        STORE_INV_DENOM: tl.constexpr,
        EVEN_D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        ROWS_PER_CTA: tl.constexpr,
    ):
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask_c = cols < D
        inv_d_const = 1.0 / D

        # Load weight/bias once per CTA and iterate rows sequentially to avoid 2D register spilling.
        if HAS_WEIGHT:
            w = tl.load(weight_ptr + cols).to(tl.float32) if EVEN_D else tl.load(weight_ptr + cols, mask=mask_c, other=1.0).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(bias_ptr + cols).to(tl.float32) if EVEN_D else tl.load(bias_ptr + cols, mask=mask_c, other=0.0).to(tl.float32)

        row_base = pid * ROWS_PER_CTA
        for r in range(0, ROWS_PER_CTA):
            row_idx = row_base + r
            if row_idx < n_rows:
                # Contiguous constexpr stride D enables 128-bit vectorized loads/stores on aligned hidden dims.
                row_off = row_idx.to(tl.int64) * D
                x = tl.load(x_ptr + row_off + cols).to(tl.float32) if EVEN_D else tl.load(x_ptr + row_off + cols, mask=mask_c, other=0.0).to(tl.float32)
                inv_denom = 1.0 / (tl.sum(tl.abs(x), axis=0) * inv_d_const + eps)
                if STORE_INV_DENOM:
                    tl.store(inv_denom_ptr + row_idx, inv_denom)
                normed = x * inv_denom
                if HAS_WEIGHT:
                    normed = normed * w
                if HAS_BIAS:
                    normed = normed + b
                if EVEN_D:
                    tl.store(y_ptr + row_off + cols, normed)
                else:
                    tl.store(y_ptr + row_off + cols, normed, mask=mask_c)

    @triton.autotune(
        configs=_RA_NORM_BWD_CONFIGS,
        key=["D"],
        prune_configs_by={"early_config_prune": _prune_ra_norm_configs},
    )
    @triton.jit
    def _ra_norm_bwd_kernel(
        grad_y_ptr,
        x_ptr,
        weight_ptr,
        inv_denom_ptr,
        dx_ptr,
        partial_dw_ptr,
        partial_db_ptr,
        n_rows,
        D: tl.constexpr,
        eps: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_INV_DENOM: tl.constexpr,
        EVEN_D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_progs = tl.num_programs(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        inv_d_const = 1.0 / D

        if HAS_WEIGHT:
            w = tl.load(weight_ptr + cols).to(tl.float32) if EVEN_D else tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        else:
            w = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        dw_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        db_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        # Contiguous row-chunking per program preserves L2 locality and enables deterministic 2-stage reduction.
        rows_per_prog = (n_rows + num_progs - 1) // num_progs
        row_start = pid * rows_per_prog
        row_end = tl.minimum(row_start + rows_per_prog, n_rows)

        for row_idx in range(row_start, row_end):
            offset = row_idx.to(tl.int64) * D
            if EVEN_D:
                gy = tl.load(grad_y_ptr + offset + cols).to(tl.float32)
                x = tl.load(x_ptr + offset + cols).to(tl.float32)
            else:
                gy = tl.load(grad_y_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
                x = tl.load(x_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
            if HAS_INV_DENOM:
                inv_denom = tl.load(inv_denom_ptr + row_idx).to(tl.float32)
            else:
                inv_denom = 1.0 / (tl.sum(tl.abs(x), axis=0) * inv_d_const + eps)
            x_hat = x * inv_denom

            if HAS_WEIGHT:
                dy_w = gy * w
                dw_acc += gy * x_hat
            else:
                dy_w = gy

            if HAS_BIAS:
                db_acc += gy

            c = tl.sum(dy_w * x_hat, axis=0) * inv_d_const
            sgn_x = (x > 0.0).to(tl.float32) - (x < 0.0).to(tl.float32)
            dx = inv_denom * (dy_w - sgn_x * c)
            if EVEN_D:
                tl.store(dx_ptr + offset + cols, dx)
            else:
                tl.store(dx_ptr + offset + cols, dx, mask=mask)

        pid_off = pid.to(tl.int64) * D
        if HAS_WEIGHT:
            tl.store(partial_dw_ptr + pid_off + cols, dw_acc, mask=mask)
        if HAS_BIAS:
            tl.store(partial_db_ptr + pid_off + cols, db_acc, mask=mask)

    @triton.jit
    def _ra_norm_bwd_reduce_kernel(
        partial_dw_ptr,
        partial_db_ptr,
        dw_ptr,
        db_ptr,
        num_progs,
        D: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        cols = pid * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_c = cols < D
        offs_p = tl.arange(0, BLOCK_P)
        if HAS_WEIGHT:
            dw_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
            for p0 in range(0, num_progs, BLOCK_P):
                p_idx = p0 + offs_p
                mask_2d = (p_idx[:, None] < num_progs) & mask_c[None, :]
                tile = tl.load(partial_dw_ptr + p_idx[:, None].to(tl.int64) * D + cols[None, :], mask=mask_2d, other=0.0)
                dw_acc += tl.sum(tile, axis=0)
            tl.store(dw_ptr + cols, dw_acc, mask=mask_c)
        if HAS_BIAS:
            db_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
            for p0 in range(0, num_progs, BLOCK_P):
                p_idx = p0 + offs_p
                mask_2d = (p_idx[:, None] < num_progs) & mask_c[None, :]
                tile = tl.load(partial_db_ptr + p_idx[:, None].to(tl.int64) * D + cols[None, :], mask=mask_2d, other=0.0)
                db_acc += tl.sum(tile, axis=0)
            tl.store(db_ptr + cols, db_acc, mask=mask_c)

    @triton.jit
    def _ra_gated_fwd_kernel(
        gate_ptr,
        up_ptr,
        out_ptr,
        n_elements,
        b_val: tl.constexpr,
        IS_SWIGLU: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        z = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        abs_z = tl.abs(z)
        inv_m = 1.0 / tl.maximum(b_val, abs_z)
        ub = b_val * inv_m
        ua = abs_z * inv_m
        uz = z * inv_m
        inv_dn = 1.0 / (ub * (ub + ua) + ua * ua)
        if IS_SWIGLU:
            q = (0.5 * ub * ub) * inv_dn
            s = tl.where(z >= 0.0, z * (1.0 - q), (0.5 * b_val * ub) * uz * inv_dn)
        else:
            s = b_val * (ub + ua) * uz * inv_dn

        tl.store(out_ptr + offsets, s * u, mask=mask)

    @triton.jit
    def _ra_gated_bwd_kernel(
        grad_out_ptr,
        gate_ptr,
        up_ptr,
        dgate_ptr,
        dup_ptr,
        n_elements,
        b_val: tl.constexpr,
        IS_SWIGLU: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        g = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        abs_z = tl.abs(z)
        inv_m = 1.0 / tl.maximum(b_val, abs_z)
        ub = b_val * inv_m
        ua = abs_z * inv_m
        uz = z * inv_m
        inv_dn = 1.0 / (ub * (ub + ua) + ua * ua)
        if IS_SWIGLU:
            q = (0.5 * ub * ub) * inv_dn
            is_pos = z >= 0.0
            sigma = tl.where(is_pos, 1.0 - q, q)
            s = tl.where(is_pos, z * sigma, (0.5 * b_val * ub) * uz * inv_dn)
            ds_dz = tl.where(
                is_pos,
                sigma + q * ua * (ub + 2.0 * ua) * inv_dn,
                q * (ub * ub - ua * ua) * inv_dn,
            )
        else:
            s = b_val * (ub + ua) * uz * inv_dn
            ds_dz = (ub * ub * ub) * (ub + 2.0 * ua) * (inv_dn * inv_dn)

        tl.store(dgate_ptr + offsets, g * u * ds_dz, mask=mask)
        tl.store(dup_ptr + offsets, g * s, mask=mask)

    @triton.jit
    def _ra_tanh_fwd_kernel(
        x_ptr,
        out_ptr,
        n_elements,
        b_val: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        abs_x = tl.abs(x)
        inv_m = 1.0 / tl.maximum(b_val, abs_x)
        ub = b_val * inv_m
        ua = abs_x * inv_m
        uz = x * inv_m
        out = (b_val * (ub + ua) * uz) / (ub * (ub + ua) + ua * ua)
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _ra_tanh_bwd_kernel(
        grad_out_ptr,
        x_ptr,
        dx_ptr,
        n_elements,
        b_val: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        g = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        abs_x = tl.abs(x)
        inv_m = 1.0 / tl.maximum(b_val, abs_x)
        ub = b_val * inv_m
        ua = abs_x * inv_m
        inv_dn = 1.0 / (ub * (ub + ua) + ua * ua)
        deriv = (ub * ub * ub) * (ub + 2.0 * ua) * (inv_dn * inv_dn)
        tl.store(dx_ptr + offsets, g * deriv, mask=mask)
else:
    _ra_norm_fwd_kernel = None
    _ra_norm_bwd_kernel = None
    _ra_norm_bwd_reduce_kernel = None
    _ra_gated_fwd_kernel = None
    _ra_gated_bwd_kernel = None
    _ra_tanh_fwd_kernel = None
    _ra_tanh_bwd_kernel = None


_HW_PROPS_CACHE: dict[int, Tuple[int, int]] = {}


def _get_cuda_hw_props(device: Any) -> Tuple[int, int]:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    cached = _HW_PROPS_CACHE.get(idx)
    if cached is None:
        props = torch.cuda.get_device_properties(idx)
        sm = max(1, int(props.multi_processor_count))
        l2 = max(1_048_576, int(getattr(props, "L2_cache_size", 4_194_304)))
        cached = (sm, l2)
        _HW_PROPS_CACHE[idx] = cached
    return cached


def triton_ra_norm(
    x: Any,
    weight: Optional[Any] = None,
    bias: Optional[Any] = None,
    eps: float = 1e-6,
    return_inv_denom: bool = False,
) -> Any:
    """Dispatches RaNorm execution to Triton GPU kernel when available."""
    if not (
        HAS_TRITON
        and _HAS_TORCH
        and isinstance(x, torch.Tensor)
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return None

    D = x.shape[-1]
    if D == 0 or D > 8192 or (weight is not None and weight.numel() != D) or (bias is not None and bias.numel() != D):
        return None
    BLOCK_SIZE = max(16, triton.next_power_of_2(D))
    EVEN_D = D == BLOCK_SIZE

    if not x.is_contiguous():
        x = x.contiguous()
    if weight is not None and not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    y = torch.empty_like(x)
    n_rows = x.numel() // D
    inv_denom = torch.empty((n_rows,), device=x.device, dtype=torch.float32) if return_inv_denom else y
    if n_rows == 0:
        return (y, inv_denom) if return_inv_denom else y

    grid = lambda meta: (triton.cdiv(n_rows, meta["ROWS_PER_CTA"]),)
    _ra_norm_fwd_kernel[grid](
        x,
        y,
        weight if weight is not None else x,
        bias if bias is not None else x,
        inv_denom,
        n_rows,
        D=D,
        eps=float(eps),
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        STORE_INV_DENOM=bool(return_inv_denom),
        EVEN_D=EVEN_D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return (y, inv_denom) if return_inv_denom else y


def triton_ra_norm_backward(
    grad_y: Any,
    x: Any,
    weight: Optional[Any] = None,
    has_bias: bool = False,
    eps: float = 1e-6,
    inv_denom: Optional[Any] = None,
) -> Optional[Tuple[Any, Optional[Any], Optional[Any]]]:
    """Dispatches RaNorm backward execution to deterministic 2-stage Triton GPU kernels."""
    if not (
        HAS_TRITON
        and _HAS_TORCH
        and isinstance(x, torch.Tensor)
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return None

    D = x.shape[-1]
    if D == 0 or D > 8192 or (weight is not None and weight.numel() != D):
        return None
    BLOCK_SIZE = max(16, triton.next_power_of_2(D))
    EVEN_D = D == BLOCK_SIZE

    gy_c = grad_y if grad_y.is_contiguous() else grad_y.contiguous()
    x_c = x if x.is_contiguous() else x.contiguous()
    w_c = (weight if weight.is_contiguous() else weight.contiguous()) if weight is not None else None

    dx = torch.empty_like(x_c)
    has_w = w_c is not None
    n_rows = x_c.numel() // D
    dw_dtype = w_c.dtype if w_c is not None else x_c.dtype
    if n_rows == 0:
        dw_empty = torch.zeros((D,), device=x.device, dtype=dw_dtype) if has_w else None
        db_empty = torch.zeros((D,), device=x.device, dtype=dw_dtype) if has_bias else None
        return dx, dw_empty, db_empty

    has_inv = isinstance(inv_denom, torch.Tensor) and inv_denom.numel() == n_rows
    inv_c = (inv_denom if inv_denom.is_contiguous() else inv_denom.contiguous()) if has_inv else dx

    if has_w or has_bias:
        sm_count, l2_bytes = _get_cuda_hw_props(x.device)
        # Scale persistent CTAs to 8/SM within 1/8 L2 cache so occupancy stays high across sm_75..sm_120.
        bytes_per_prog = D * 4 * (2 if (has_w and has_bias) else 1)
        max_l2_progs = max(sm_count, (l2_bytes // 8) // max(1, bytes_per_prog))
        aligned_l2 = (max_l2_progs // sm_count) * sm_count or sm_count
        num_progs = min(n_rows, max(1, min(sm_count * 8, aligned_l2)))
    else:
        num_progs = n_rows
    partial_dw = torch.empty((num_progs, D), device=x.device, dtype=torch.float32) if has_w else dx
    partial_db = torch.empty((num_progs, D), device=x.device, dtype=torch.float32) if has_bias else dx

    _ra_norm_bwd_kernel[(max(1, num_progs),)](
        gy_c,
        x_c,
        w_c if w_c is not None else x_c,
        inv_c,
        dx,
        partial_dw,
        partial_db,
        n_rows,
        D=D,
        eps=float(eps),
        HAS_WEIGHT=has_w,
        HAS_BIAS=bool(has_bias),
        HAS_INV_DENOM=has_inv,
        EVEN_D=EVEN_D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    dw = torch.empty((D,), device=x.device, dtype=dw_dtype) if has_w else None
    db = torch.empty((D,), device=x.device, dtype=dw_dtype) if has_bias else None
    if has_w or has_bias:
        BLOCK_D = 64 if D >= 64 else 32
        _ra_norm_bwd_reduce_kernel[(triton.cdiv(D, BLOCK_D),)](
            partial_dw,
            partial_db,
            dw if dw is not None else dx,
            db if db is not None else dx,
            num_progs,
            D=D,
            HAS_WEIGHT=has_w,
            HAS_BIAS=bool(has_bias),
            BLOCK_P=32,
            BLOCK_D=BLOCK_D,
            num_warps=4,
        )
    return dx, dw, db


def _triton_ra_gated(gate: Any, up: Any, b: float, is_swiglu: bool) -> Any:
    if not (
        HAS_TRITON and _HAS_TORCH and isinstance(gate, torch.Tensor) and gate.is_cuda
        and gate.dtype in (torch.float16, torch.bfloat16, torch.float32) and up.dtype == gate.dtype
    ):
        return None
    if gate.numel() == 0:
        return torch.empty_like(gate)
    if not gate.is_contiguous():
        gate = gate.contiguous()
    if not up.is_contiguous():
        up = up.contiguous()
    n_elements = gate.numel()
    out = torch.empty_like(gate)
    BLOCK_SIZE = 1024
    _ra_gated_fwd_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        gate, up, out, n_elements, b_val=float(b), IS_SWIGLU=is_swiglu, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def _triton_ra_gated_backward(grad_out: Any, gate: Any, up: Any, b: float, is_swiglu: bool) -> Optional[Tuple[Any, Any]]:
    if not (
        HAS_TRITON and _HAS_TORCH and isinstance(gate, torch.Tensor) and gate.is_cuda
        and gate.dtype in (torch.float16, torch.bfloat16, torch.float32) and up.dtype == gate.dtype
    ):
        return None
    if gate.numel() == 0:
        return torch.empty_like(gate), torch.empty_like(up)
    g_c = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
    gate_c = gate if gate.is_contiguous() else gate.contiguous()
    up_c = up if up.is_contiguous() else up.contiguous()
    n_elements = gate_c.numel()
    dgate = torch.empty_like(gate_c)
    dup = torch.empty_like(up_c)
    BLOCK_SIZE = 1024
    _ra_gated_bwd_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        g_c, gate_c, up_c, dgate, dup, n_elements, b_val=float(b), IS_SWIGLU=is_swiglu, BLOCK_SIZE=BLOCK_SIZE
    )
    return dgate, dup


def triton_ra_swiglu(gate: Any, up: Any, b: float = 1.0) -> Any:
    """Dispatches RaSwiGLU forward execution to Triton GPU kernel."""
    return _triton_ra_gated(gate, up, b, is_swiglu=True)


def triton_ra_swiglu_backward(grad_out: Any, gate: Any, up: Any, b: float = 1.0) -> Optional[Tuple[Any, Any]]:
    """Dispatches RaSwiGLU backward execution to Triton GPU kernel."""
    return _triton_ra_gated_backward(grad_out, gate, up, b, is_swiglu=True)


def triton_ra_glu(gate: Any, up: Any, b: float = 1.0) -> Any:
    """Dispatches RaGLU forward execution to Triton GPU kernel."""
    return _triton_ra_gated(gate, up, b, is_swiglu=False)


def triton_ra_glu_backward(grad_out: Any, gate: Any, up: Any, b: float = 1.0) -> Optional[Tuple[Any, Any]]:
    """Dispatches RaGLU backward execution to Triton GPU kernel."""
    return _triton_ra_gated_backward(grad_out, gate, up, b, is_swiglu=False)


def triton_ra_tanh(x: Any, b: float = 1.0) -> Any:
    """Dispatches RaTanh forward execution to Triton GPU kernel."""
    if not (
        HAS_TRITON and _HAS_TORCH and isinstance(x, torch.Tensor) and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return None
    if x.numel() == 0:
        return torch.empty_like(x)
    x_c = x if x.is_contiguous() else x.contiguous()
    n_elements = x_c.numel()
    out = torch.empty_like(x_c)
    BLOCK_SIZE = 1024
    _ra_tanh_fwd_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        x_c, out, n_elements, b_val=float(b), BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def triton_ra_tanh_backward(grad_out: Any, x: Any, b: float = 1.0) -> Any:
    """Dispatches RaTanh backward execution to Triton GPU kernel."""
    if not (
        HAS_TRITON and _HAS_TORCH and isinstance(x, torch.Tensor) and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return None
    if x.numel() == 0:
        return torch.empty_like(x)
    g_c = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
    x_c = x if x.is_contiguous() else x.contiguous()
    n_elements = x_c.numel()
    dx = torch.empty_like(x_c)
    BLOCK_SIZE = 1024
    _ra_tanh_bwd_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        g_c, x_c, dx, n_elements, b_val=float(b), BLOCK_SIZE=BLOCK_SIZE
    )
    return dx
