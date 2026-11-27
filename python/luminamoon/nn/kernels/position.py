"""Fused Triton forward and backward kernel for QRoPE (Spin(4) double-quaternion rotation)."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from ._base import HAS_TRITON, _dev_idx, _int_spec, _launch_ck_direct, _ptr_aligned, _store_ck, tl, torch, triton

_QROPE_CK_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()

if TYPE_CHECKING:
    _qrope_spin4_kernel: Any = None
    _qrope_spin4_qk_kernel: Any = None
elif HAS_TRITON:

    @triton.jit
    def _qrope_spin4_kernel(
        x_ptr,
        out_ptr,
        cos_l_ptr,
        sin_l_ptr,
        cos_r_ptr,
        sin_r_ptr,
        n_rows,
        heads,
        seq_len,
        num_quaternions,
        stride_xb,
        stride_xh,
        stride_xs,
        stride_out_row,
        stride_trig,
        SIGN: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        ROWS_PER_CTA: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        row_offs = pid * ROWS_PER_CTA + tl.arange(0, ROWS_PER_CTA)
        row_mask = row_offs < n_rows

        pos_idx = row_offs % seq_len
        bh_idx = row_offs // seq_len
        h_idx = bh_idx % heads
        b_idx = bh_idx // heads

        quat_offs = tl.arange(0, BLOCK_Q)
        mask = row_mask[:, None] & (quat_offs[None, :] < num_quaternions)

        trig_offs = pos_idx[:, None] * stride_trig + quat_offs[None, :]
        cl = tl.load(cos_l_ptr + trig_offs, mask=mask, other=1.0).to(tl.float32)
        sl = tl.load(sin_l_ptr + trig_offs, mask=mask, other=0.0).to(tl.float32) * SIGN
        cr = tl.load(cos_r_ptr + trig_offs, mask=mask, other=1.0).to(tl.float32)
        sr = tl.load(sin_r_ptr + trig_offs, mask=mask, other=0.0).to(tl.float32) * SIGN

        in_base = (
            b_idx[:, None].to(tl.int64) * stride_xb
            + h_idx[:, None].to(tl.int64) * stride_xh
            + pos_idx[:, None].to(tl.int64) * stride_xs
            + quat_offs[None, :] * 4
        )
        w = tl.load(x_ptr + in_base + 0, mask=mask, other=0.0).to(tl.float32)
        i = tl.load(x_ptr + in_base + 1, mask=mask, other=0.0).to(tl.float32)
        j = tl.load(x_ptr + in_base + 2, mask=mask, other=0.0).to(tl.float32)
        k = tl.load(x_ptr + in_base + 3, mask=mask, other=0.0).to(tl.float32)

        # Left quaternion multiply: y = q_L * x with axis u_L = (1, 1, 1) / sqrt(3)
        yw = cl * w - sl * (i + j + k)
        yi = cl * i + sl * (w + k - j)
        yj = cl * j + sl * (w + i - k)
        yk = cl * k + sl * (w + j - i)

        # Right quaternion multiply: out = y * q_R with axis u_R = (1, -1, 0) / sqrt(2)
        ow = cr * yw - sr * (yi - yj)
        oi = cr * yi + sr * (yw + yk)
        oj = cr * yj + sr * (yk - yw)
        ok = cr * yk - sr * (yi + yj)

        out_base = row_offs[:, None].to(tl.int64) * stride_out_row + quat_offs[None, :] * 4
        tl.store(out_ptr + out_base + 0, ow, mask=mask)
        tl.store(out_ptr + out_base + 1, oi, mask=mask)
        tl.store(out_ptr + out_base + 2, oj, mask=mask)
        tl.store(out_ptr + out_base + 3, ok, mask=mask)

    @triton.jit
    def _qrope_spin4_qk_kernel(
        q_ptr, k_ptr, oq_ptr, ok_ptr,
        cos_l_ptr, sin_l_ptr, cos_r_ptr, sin_r_ptr,
        n_rows_q, n_rows_k, heads_q, heads_k, seq_len, num_quaternions,
        sqb, sqh, sqs, skb, skh, sks,
        soq_row, sok_row, stride_trig,
        SIGN: tl.constexpr, BLOCK_Q: tl.constexpr, ROWS_PER_CTA: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        q_ctas = tl.cdiv(n_rows_q, ROWS_PER_CTA)
        if pid >= q_ctas:
            row_offs = (pid - q_ctas) * ROWS_PER_CTA + tl.arange(0, ROWS_PER_CTA)
            row_mask = row_offs < n_rows_k
            pos_idx = row_offs % seq_len
            bh_idx = row_offs // seq_len
            h_idx = bh_idx % heads_k
            b_idx = bh_idx // heads_k
            in_ptr, out_ptr, sb, sh, ss, so = k_ptr, ok_ptr, skb, skh, sks, sok_row
        else:
            row_offs = pid * ROWS_PER_CTA + tl.arange(0, ROWS_PER_CTA)
            row_mask = row_offs < n_rows_q
            pos_idx = row_offs % seq_len
            bh_idx = row_offs // seq_len
            h_idx = bh_idx % heads_q
            b_idx = bh_idx // heads_q
            in_ptr, out_ptr, sb, sh, ss, so = q_ptr, oq_ptr, sqb, sqh, sqs, soq_row

        quat_offs = tl.arange(0, BLOCK_Q)
        mask = row_mask[:, None] & (quat_offs[None, :] < num_quaternions)
        trig_offs = pos_idx[:, None] * stride_trig + quat_offs[None, :]
        cl = tl.load(cos_l_ptr + trig_offs, mask=mask, other=1.0).to(tl.float32)
        sl = tl.load(sin_l_ptr + trig_offs, mask=mask, other=0.0).to(tl.float32) * SIGN
        cr = tl.load(cos_r_ptr + trig_offs, mask=mask, other=1.0).to(tl.float32)
        sr = tl.load(sin_r_ptr + trig_offs, mask=mask, other=0.0).to(tl.float32) * SIGN

        in_base = (
            b_idx[:, None].to(tl.int64) * sb
            + h_idx[:, None].to(tl.int64) * sh
            + pos_idx[:, None].to(tl.int64) * ss
            + quat_offs[None, :] * 4
        )
        w = tl.load(in_ptr + in_base + 0, mask=mask, other=0.0).to(tl.float32)
        i = tl.load(in_ptr + in_base + 1, mask=mask, other=0.0).to(tl.float32)
        j = tl.load(in_ptr + in_base + 2, mask=mask, other=0.0).to(tl.float32)
        k = tl.load(in_ptr + in_base + 3, mask=mask, other=0.0).to(tl.float32)

        yw = cl * w - sl * (i + j + k)
        yi = cl * i + sl * (w + k - j)
        yj = cl * j + sl * (w + i - k)
        yk = cl * k + sl * (w + j - i)

        out_base = row_offs[:, None].to(tl.int64) * so + quat_offs[None, :] * 4
        tl.store(out_ptr + out_base + 0, cr * yw - sr * (yi - yj), mask=mask)
        tl.store(out_ptr + out_base + 1, cr * yi + sr * (yw + yk), mask=mask)
        tl.store(out_ptr + out_base + 2, cr * yj + sr * (yk - yw), mask=mask)
        tl.store(out_ptr + out_base + 3, cr * yk - sr * (yi + yj), mask=mask)

else:
    _qrope_spin4_kernel: Any = None
    _qrope_spin4_qk_kernel: Any = None


def _launch_qrope(
    x: Any,
    cos_l: Any,
    sin_l: Any,
    cos_r: Any,
    sin_r: Any,
    sign: float,
) -> Any:
    if not HAS_TRITON or _qrope_spin4_kernel is None:
        raise RuntimeError("Triton is not available for QRoPE kernel launch.")
    if not (isinstance(x, torch.Tensor) and x.is_cuda and x.ndim == 4 and x.shape[-1] > 0 and x.shape[-1] % 4 == 0):
        raise ValueError(f"triton_qrope expects 4D CUDA tensor with head_dim % 4 == 0, got {getattr(x, 'shape', type(x))}.")
    batch, heads, seq_len, d_head = x.shape
    num_quaternions = d_head // 4
    expected = (seq_len, num_quaternions)
    trig_norm = []
    for name, trig in (("cos_l", cos_l), ("sin_l", sin_l), ("cos_r", cos_r), ("sin_r", sin_r)):
        if not isinstance(trig, torch.Tensor) or trig.device != x.device:
            raise ValueError(f"{name} must be a CUDA Tensor on the same device as x ({x.device}).")
        if tuple(trig.shape) == expected:
            trig_norm.append(trig if trig.is_contiguous() else trig.contiguous())
        elif trig.numel() == seq_len * num_quaternions and trig.shape[-2:] == expected:
            trig_norm.append(trig.reshape(expected).contiguous())
        else:
            raise ValueError(f"{name} must have shape {expected} (or broadcast-singleton leading dims), got {tuple(trig.shape)}.")
    cl_c, sl_c, cr_c, sr_c = trig_norm
    # Accept strided (B, H, S, D) views when head_dim is unit-strided to avoid HBM copies.
    x_in = x if x.stride(-1) == 1 else x.contiguous()
    n_rows = batch * heads * seq_len
    out = torch.empty((batch, heads, seq_len, d_head), device=x_in.device, dtype=x_in.dtype)
    if n_rows == 0 or num_quaternions == 0:
        return out

    block_q = triton.next_power_of_2(num_quaternions)
    rows_per_cta = 16 if (block_q <= 32 and n_rows >= 16) else (8 if (block_q <= 64 and n_rows >= 8) else 1)
    num_warps = 4 if rows_per_cta >= 8 else (2 if block_q <= 32 else (4 if block_q <= 128 else 8))
    _qrope_spin4_kernel[(triton.cdiv(n_rows, rows_per_cta),)](
        x_in,
        out,
        cl_c,
        sl_c,
        cr_c,
        sr_c,
        n_rows,
        heads,
        seq_len,
        num_quaternions,
        x_in.stride(0),
        x_in.stride(1),
        x_in.stride(2),
        d_head,
        cl_c.stride(0),
        SIGN=sign,
        BLOCK_Q=block_q,
        ROWS_PER_CTA=rows_per_cta,
        num_warps=num_warps,
        enable_fp_fusion=True,
    )
    return out


def _launch_qrope_qk(
    q: Any, k: Any, cos_l: Any, sin_l: Any, cos_r: Any, sin_r: Any, sign: float,
) -> tuple[Any, Any]:
    if not HAS_TRITON or _qrope_spin4_qk_kernel is None:
        raise RuntimeError("Triton is not available for QRoPE QK kernel launch.")
    if q.ndim != 4 or k.ndim != 4 or q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] or q.shape[3] != k.shape[3]:
        raise ValueError(f"triton_qrope_qk requires matching (B, *, S, D), got {tuple(q.shape)} and {tuple(k.shape)}.")
    if q.shape[3] <= 0 or (q.shape[3] % 4) != 0 or q.device != k.device or q.dtype != k.dtype:
        raise ValueError("triton_qrope_qk requires matching device/dtype and head_dim divisible by 4.")
    batch, heads_q, seq_len, d_head = q.shape
    heads_k = k.shape[1]
    num_quaternions = d_head // 4
    expected = (seq_len, num_quaternions)
    cl_c = cos_l if (tuple(cos_l.shape) == expected and cos_l.is_contiguous()) else cos_l.reshape(expected).contiguous()
    sl_c = sin_l if (tuple(sin_l.shape) == expected and sin_l.is_contiguous()) else sin_l.reshape(expected).contiguous()
    cr_c = cos_r if (tuple(cos_r.shape) == expected and cos_r.is_contiguous()) else cos_r.reshape(expected).contiguous()
    sr_c = sin_r if (tuple(sin_r.shape) == expected and sin_r.is_contiguous()) else sin_r.reshape(expected).contiguous()
    q_in = q if q.stride(-1) == 1 else q.contiguous()
    k_in = k if k.stride(-1) == 1 else k.contiguous()
    oq = torch.empty(q_in.shape, device=q_in.device, dtype=q_in.dtype)
    ok = torch.empty(k_in.shape, device=k_in.device, dtype=k_in.dtype)
    n_rows_q, n_rows_k = batch * heads_q * seq_len, batch * heads_k * seq_len
    if (n_rows_q == 0 and n_rows_k == 0) or num_quaternions == 0:
        return oq, ok
    block_q = triton.next_power_of_2(num_quaternions)
    rows_per_cta = 16 if (block_q <= 32 and max(n_rows_q, n_rows_k) >= 16) else (8 if (block_q <= 64 and max(n_rows_q, n_rows_k) >= 8) else 1)
    num_warps = 4 if rows_per_cta >= 8 else (2 if block_q <= 32 else (4 if block_q <= 128 else 8))
    grid = (triton.cdiv(n_rows_q, rows_per_cta) + triton.cdiv(n_rows_k, rows_per_cta), 1, 1)
    qk_args = (
        q_in, k_in, oq, ok, cl_c, sl_c, cr_c, sr_c,
        n_rows_q, n_rows_k, heads_q, heads_k, seq_len, num_quaternions,
        q_in.stride(0), q_in.stride(1), q_in.stride(2),
        k_in.stride(0), k_in.stride(1), k_in.stride(2),
        d_head, d_head, cl_c.stride(0),
    )
    ck_key = (
        "qk", _dev_idx(q_in.device), q_in.dtype, k_in.dtype, cl_c.dtype, sl_c.dtype, cr_c.dtype, sr_c.dtype,
        float(sign), block_q, rows_per_cta, _ptr_aligned(q_in, k_in, oq, ok, cl_c, sl_c, cr_c, sr_c),
        tuple(_int_spec(v) for v in qk_args[8:]),
    )
    if not _launch_ck_direct(_QROPE_CK_CACHE, ck_key, q_in.device, grid, qk_args):
        with torch.cuda.device(q_in.device):
            ck = _qrope_spin4_qk_kernel[(grid[0],)](
                *qk_args, SIGN=sign, BLOCK_Q=block_q, ROWS_PER_CTA=rows_per_cta, num_warps=num_warps, enable_fp_fusion=True,
            )
        if hasattr(ck, "__getitem__"):
            _store_ck(_QROPE_CK_CACHE, ck_key, ck)
    return oq, ok


def triton_qrope_forward(x: Any, cos_l: Any, sin_l: Any, cos_r: Any, sin_r: Any) -> Any:
    return _launch_qrope(x, cos_l, sin_l, cos_r, sin_r, sign=1.0)


def triton_qrope_backward(grad_out: Any, cos_l: Any, sin_l: Any, cos_r: Any, sin_r: Any) -> Any:
    # Spin(4) double-quaternion multiplication is orthogonal, so transpose equals inverse rotation.
    return _launch_qrope(grad_out, cos_l, sin_l, cos_r, sin_r, sign=-1.0)


def triton_qrope_qk_forward(q: Any, k: Any, cos_l: Any, sin_l: Any, cos_r: Any, sin_r: Any) -> tuple[Any, Any]:
    return _launch_qrope_qk(q, k, cos_l, sin_l, cos_r, sin_r, sign=1.0)


def triton_qrope_qk_backward(gq: Any, gk: Any, cos_l: Any, sin_l: Any, cos_r: Any, sin_r: Any) -> tuple[Any, Any]:
    return _launch_qrope_qk(gq, gk, cos_l, sin_l, cos_r, sin_r, sign=-1.0)


__all__ = ["triton_qrope_forward", "triton_qrope_backward", "triton_qrope_qk_forward", "triton_qrope_qk_backward"]
