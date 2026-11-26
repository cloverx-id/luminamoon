"""Fused Triton forward and backward kernel for QRoPE (Spin(4) double-quaternion rotation)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._base import HAS_TRITON, tl, torch, triton

if TYPE_CHECKING:
    _qrope_spin4_kernel: Any = None
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
        seq_len,
        num_quaternions,
        stride_row,
        stride_trig,
        SIGN: tl.constexpr,
        BLOCK_Q: tl.constexpr,
    ):
        row_idx = tl.program_id(axis=0)
        if row_idx >= n_rows:
            return
        pos_idx = row_idx % seq_len
        quat_offs = tl.arange(0, BLOCK_Q)
        mask = quat_offs < num_quaternions

        trig_offs = pos_idx * stride_trig + quat_offs
        cl = tl.load(cos_l_ptr + trig_offs, mask=mask, other=1.0).to(tl.float32)
        sl = tl.load(sin_l_ptr + trig_offs, mask=mask, other=0.0).to(tl.float32) * SIGN
        cr = tl.load(cos_r_ptr + trig_offs, mask=mask, other=1.0).to(tl.float32)
        sr = tl.load(sin_r_ptr + trig_offs, mask=mask, other=0.0).to(tl.float32) * SIGN

        base = row_idx * stride_row + quat_offs * 4
        w = tl.load(x_ptr + base + 0, mask=mask, other=0.0).to(tl.float32)
        i = tl.load(x_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)
        j = tl.load(x_ptr + base + 2, mask=mask, other=0.0).to(tl.float32)
        k = tl.load(x_ptr + base + 3, mask=mask, other=0.0).to(tl.float32)

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

        tl.store(out_ptr + base + 0, ow, mask=mask)
        tl.store(out_ptr + base + 1, oi, mask=mask)
        tl.store(out_ptr + base + 2, oj, mask=mask)
        tl.store(out_ptr + base + 3, ok, mask=mask)

else:
    _qrope_spin4_kernel: Any = None


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
    x_c = x.contiguous()
    cl_c = cos_l.contiguous()
    sl_c = sin_l.contiguous()
    cr_c = cos_r.contiguous()
    sr_c = sin_r.contiguous()

    batch, heads, seq_len, d_head = x_c.shape
    num_quaternions = d_head // 4
    n_rows = batch * heads * seq_len
    out = torch.empty_like(x_c)
    if n_rows == 0 or num_quaternions == 0:
        return out

    block_q = triton.next_power_of_2(num_quaternions)
    num_warps = 2 if block_q <= 32 else (4 if block_q <= 128 else 8)
    _qrope_spin4_kernel[(n_rows,)](
        x_c,
        out,
        cl_c,
        sl_c,
        cr_c,
        sr_c,
        n_rows,
        seq_len,
        num_quaternions,
        d_head,
        cl_c.stride(0),
        SIGN=sign,
        BLOCK_Q=block_q,
        num_warps=num_warps,
        enable_fp_fusion=True,
    )
    return out


def triton_qrope_forward(x: Any, cos_l: Any, sin_l: Any, cos_r: Any, sin_r: Any) -> Any:
    return _launch_qrope(x, cos_l, sin_l, cos_r, sin_r, sign=1.0)


def triton_qrope_backward(grad_out: Any, cos_l: Any, sin_l: Any, cos_r: Any, sin_r: Any) -> Any:
    # Spin(4) double-quaternion multiplication is orthogonal, so transpose equals inverse rotation.
    return _launch_qrope(grad_out, cos_l, sin_l, cos_r, sin_r, sign=-1.0)


__all__ = ["HAS_TRITON", "triton_qrope_forward", "triton_qrope_backward"]
