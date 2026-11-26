"""Positional encoding and distance-bias modules for luminamoon.nn."""

from __future__ import annotations

import math
from typing import Any, Literal, Optional

from ._compat import BaseModule, _HAS_TORCH, _HAS_TRITON, _is_torch_compiling, nn, torch
from .kernels.position import triton_qrope_backward, triton_qrope_forward

_INV_SQRT3: float = 1.0 / math.sqrt(3.0)
_INV_SQRT2: float = 1.0 / math.sqrt(2.0)


def _qrope_eager(
    x: Any,
    cos_l: Any,
    sin_l: Any,
    cos_r: Any,
    sin_r: Any,
    sign: float = 1.0,
) -> Any:
    batch, heads, seq_len, d_head = x.shape
    num_quaternions = d_head // 4
    input_dtype = x.dtype
    compute_dtype = torch.float64 if input_dtype == torch.float64 else torch.float32

    x_4d = x.to(compute_dtype).reshape(batch, heads, seq_len, num_quaternions, 4)
    w, i, j, k = x_4d.unbind(dim=-1)

    cl = cos_l.to(compute_dtype).view(1, 1, seq_len, num_quaternions)
    sl_base = sin_l.to(compute_dtype).view(1, 1, seq_len, num_quaternions)
    sl = sl_base if sign > 0 else -sl_base
    cr = cos_r.to(compute_dtype).view(1, 1, seq_len, num_quaternions)
    sr_base = sin_r.to(compute_dtype).view(1, 1, seq_len, num_quaternions)
    sr = sr_base if sign > 0 else -sr_base

    yw = cl * w - sl * (i + j + k)
    yi = cl * i + sl * (w + k - j)
    yj = cl * j + sl * (w + i - k)
    yk = cl * k + sl * (w + j - i)

    ow = cr * yw - sr * (yi - yj)
    oi = cr * yi + sr * (yw + yk)
    oj = cr * yj + sr * (yk - yw)
    ok = cr * yk - sr * (yi + yj)

    return torch.stack((ow, oi, oj, ok), dim=-1).reshape(batch, heads, seq_len, d_head).to(dtype=input_dtype)


if _HAS_TORCH:

    class _QRoPEFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any, cos_l: Any, sin_l: Any, cos_r: Any, sin_r: Any) -> Any:
            ctx.save_for_backward(cos_l, sin_l, cos_r, sin_r)
            if (
                _HAS_TRITON
                and x.is_cuda
                and cos_l.device == x.device
                and sin_l.device == x.device
                and cos_r.device == x.device
                and sin_r.device == x.device
                and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and not _is_torch_compiling()
                and triton_qrope_forward is not None
            ):
                return triton_qrope_forward(x, cos_l, sin_l, cos_r, sin_r)
            return _qrope_eager(x, cos_l, sin_l, cos_r, sin_r, sign=1.0)

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> tuple[Any, None, None, None, None]:  # type: ignore[override]
            cos_l, sin_l, cos_r, sin_r = ctx.saved_tensors
            if (
                _HAS_TRITON
                and grad_out.is_cuda
                and cos_l.device == grad_out.device
                and sin_l.device == grad_out.device
                and cos_r.device == grad_out.device
                and sin_r.device == grad_out.device
                and grad_out.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and not _is_torch_compiling()
                and triton_qrope_backward is not None
            ):
                grad_x = triton_qrope_backward(grad_out, cos_l, sin_l, cos_r, sin_r)
            else:
                grad_x = _qrope_eager(grad_out, cos_l, sin_l, cos_r, sin_r, sign=-1.0)
            return grad_x, None, None, None, None

else:
    _QRoPEFunction = None  # type: ignore[assignment]


def qrope(x: Any, cos_l: Any, sin_l: Any, cos_r: Any, sin_r: Any) -> Any:
    """Apply Spin(4) left-right double-quaternion rotary embedding to a 4D tensor `(B, H, S, D)`."""
    if not _HAS_TORCH or not isinstance(x, torch.Tensor):
        raise TypeError("qrope requires a torch.Tensor input.")
    if x.ndim != 4 or x.shape[-1] <= 0 or x.shape[-1] % 4 != 0:
        raise ValueError(f"qrope expects 4D tensor with head_dim divisible by 4, got shape {tuple(x.shape)}.")
    expected = (x.shape[2], x.shape[3] // 4)
    for name, trig in (("cos_l", cos_l), ("sin_l", sin_l), ("cos_r", cos_r), ("sin_r", sin_r)):
        if not isinstance(trig, torch.Tensor):
            raise TypeError(f"qrope requires {name} to be a torch.Tensor.")
        if trig.numel() != expected[0] * expected[1] or trig.shape[-2:] != expected:
            raise ValueError(f"qrope expects {name} with trailing shape {expected}, got {tuple(trig.shape)}.")
    if _QRoPEFunction is not None and not _is_torch_compiling():
        return _QRoPEFunction.apply(x, cos_l, sin_l, cos_r, sin_r)
    return _qrope_eager(x, cos_l, sin_l, cos_r, sin_r, sign=1.0)


class QRoPE(BaseModule):
    """Spin(4) left-right double-quaternion rotary positional embedding.

    Applies two commuting unit-quaternion rotations (`q_L * x * q_R`) to every
    4-element block of each head vector, preserving L2 norm and relative-position
    inner products across sequence offsets.
    """

    def __init__(
        self,
        head_dim: int = 32,
        max_seq_len: int = 4096,
        base: float = 10000.0,
        freq_ratio: float = 0.6180339887498949,
    ) -> None:
        super().__init__()
        if not _HAS_TORCH:
            raise ImportError("QRoPE requires PyTorch.")
        if head_dim <= 0 or head_dim % 4 != 0:
            raise ValueError(f"QRoPE requires head_dim divisible by 4, got {head_dim}.")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}.")
        if base <= 1.0:
            raise ValueError(f"base must be > 1.0, got {base}.")

        self.head_dim = int(head_dim)
        self.num_quaternions = self.head_dim // 4
        self.base = float(base)
        self.freq_ratio = float(freq_ratio)

        theta_l = 1.0 / (
            self.base ** (torch.arange(0, self.num_quaternions, dtype=torch.float32) / self.num_quaternions)
        )
        theta_r = theta_l * self.freq_ratio
        self.register_buffer("theta_l", theta_l, persistent=False)
        self.register_buffer("theta_r", theta_r, persistent=False)

        self._cached_len = 0
        self.register_buffer("cos_l", torch.empty(0), persistent=False)
        self.register_buffer("sin_l", torch.empty(0), persistent=False)
        self.register_buffer("cos_r", torch.empty(0), persistent=False)
        self.register_buffer("sin_r", torch.empty(0), persistent=False)
        self._build_cache(max_seq_len, device=theta_l.device)

    def _build_cache(self, seq_len: int, device: Any) -> None:
        req = int(seq_len)
        target_len = max(req, max(1, self._cached_len * 2)) if req > self._cached_len else max(1, self._cached_len)
        idx = torch.arange(target_len, device=device, dtype=torch.float32)
        theta_l_t: Any = self.theta_l
        theta_r_t: Any = self.theta_r
        half_l = torch.outer(idx, theta_l_t.to(device=device)) * 0.5
        half_r = torch.outer(idx, theta_r_t.to(device=device)) * 0.5
        self.cos_l = torch.cos(half_l)
        self.sin_l = torch.sin(half_l) * _INV_SQRT3
        self.cos_r = torch.cos(half_r)
        self.sin_r = torch.sin(half_r) * _INV_SQRT2
        self._cached_len = target_len

    def forward(self, x: Any, offset: Any = 0) -> Any:
        if x.ndim != 4 or x.shape[-1] != self.head_dim:
            raise ValueError(f"QRoPE expected (B, H, S, {self.head_dim}), got {tuple(x.shape)}.")
        seq_len = x.shape[2]
        if isinstance(offset, torch.Tensor):
            pos = offset.reshape(-1).to(device=x.device, dtype=torch.long)
            if pos.shape[0] != seq_len:
                raise ValueError(
                    f"Tensor offset must have length matching seq_len ({seq_len}), got {tuple(offset.shape)}."
                )
            if self.cos_l.device != x.device:
                self._build_cache(max(self._cached_len, seq_len), device=x.device)
            return qrope(
                x,
                self.cos_l.index_select(0, pos),
                self.sin_l.index_select(0, pos),
                self.cos_r.index_select(0, pos),
                self.sin_r.index_select(0, pos),
            )
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}.")
        needed = offset + seq_len
        if needed > self._cached_len or self.cos_l.device != x.device:
            self._build_cache(needed, device=x.device)
        return qrope(
            x,
            self.cos_l[offset:needed],
            self.sin_l[offset:needed],
            self.cos_r[offset:needed],
            self.sin_r[offset:needed],
        )

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, base={self.base}, freq_ratio={self.freq_ratio:.6f}"


class DistanceDecayBias(BaseModule):
    """Per-head distance-decay attention bias with optional hub-token damping.

    Supports `"linear"` (`-s * d`, ALiBi-style) and `"rational"` (`-log(1 + s * d)`)
    decay modes, optional learnable slopes, and separate damping for prefix hub tokens.
    """

    def __init__(
        self,
        num_heads: int = 8,
        hub_size: int = 0,
        num_hub_heads: Optional[int] = None,
        min_slope: float = 0.01,
        max_slope: float = 0.45,
        mode: Literal["linear", "rational"] = "linear",
        learnable: bool = False,
        min_hub_penalty: float = 0.75,
        max_hub_penalty: float = 2.5,
    ) -> None:
        super().__init__()
        if not _HAS_TORCH:
            raise ImportError("DistanceDecayBias requires PyTorch.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}.")
        if hub_size < 0:
            raise ValueError(f"hub_size must be >= 0, got {hub_size}.")
        mode_l = str(mode).lower()
        if mode_l not in ("linear", "rational"):
            raise ValueError(f"mode must be 'linear' or 'rational', got {mode!r}.")

        self.num_heads = int(num_heads)
        self.hub_size = int(hub_size)
        self.mode = mode_l
        self.learnable = bool(learnable)

        if self.hub_size == 0:
            self.num_hub_heads = 0
            self.num_text_heads = self.num_heads
        elif self.num_heads == 1:
            self.num_hub_heads = 1
            self.num_text_heads = 0
        elif num_hub_heads is not None and num_hub_heads > 0:
            self.num_hub_heads = max(1, min(int(num_hub_heads), self.num_heads - 1))
            self.num_text_heads = self.num_heads - self.num_hub_heads
        else:
            self.num_hub_heads = max(1, self.num_heads // 8)
            self.num_text_heads = self.num_heads - self.num_hub_heads

        slope_parts: list[Any] = []
        if self.num_hub_heads > 0:
            slope_parts.append(torch.full((1, self.num_hub_heads, 1, 1), 0.015, dtype=torch.float32))
        if self.num_text_heads > 1:
            slope_parts.append(
                torch.linspace(float(min_slope), float(max_slope), self.num_text_heads, dtype=torch.float32).view(
                    1, self.num_text_heads, 1, 1
                )
            )
        elif self.num_text_heads == 1:
            slope_parts.append(torch.full((1, 1, 1, 1), 0.5 * (float(min_slope) + float(max_slope)), dtype=torch.float32))
        init_slopes = torch.cat(slope_parts, dim=1)

        if self.learnable:
            self.raw_slopes = nn.Parameter(torch.log(torch.expm1(init_slopes).clamp(min=1e-6)))
        else:
            self.register_buffer("raw_slopes", init_slopes, persistent=False)

        # Direct log-domain hub penalty keeps text heads differentiated even when min_slope is small.
        pen_parts: list[Any] = []
        if self.num_hub_heads > 0:
            pen_parts.append(torch.zeros(1, self.num_hub_heads, 1, 1, dtype=torch.float32))
        if self.num_text_heads > 1:
            pen_parts.append(
                torch.linspace(float(min_hub_penalty), float(max_hub_penalty), self.num_text_heads, dtype=torch.float32).view(
                    1, self.num_text_heads, 1, 1
                )
            )
        elif self.num_text_heads == 1:
            pen_parts.append(
                torch.full((1, 1, 1, 1), 0.5 * (float(min_hub_penalty) + float(max_hub_penalty)), dtype=torch.float32)
            )
        hub_penalty = torch.cat(pen_parts, dim=1) if pen_parts else torch.zeros(1, self.num_heads, 1, 1, dtype=torch.float32)
        self.register_buffer("hub_penalty", hub_penalty, persistent=False)
        self._bias_cache: dict[tuple[Any, ...], Any] = {}

    @property
    def slopes(self) -> Any:
        if self.learnable:
            return torch.nn.functional.softplus(self.raw_slopes) + 1e-4
        return self.raw_slopes

    def forward(
        self,
        seq_len: int,
        num_all: int,
        device: Any,
        dtype: Any,
        is_dense_with_hub: bool = True,
        has_past_kv: bool = False,
    ) -> Any:
        text_k_len = num_all - self.hub_size
        if text_k_len < 0:
            raise ValueError(f"num_all ({num_all}) must be >= hub_size ({self.hub_size}).")

        use_cache = not self.learnable and not _is_torch_compiling()
        cache_key: Optional[tuple[Any, ...]] = None
        if use_cache:
            cache_key = (
                int(seq_len),
                int(num_all),
                str(device),
                dtype,
                bool(is_dense_with_hub),
                bool(has_past_kv),
                int(self.raw_slopes._version),
                int(self.hub_penalty._version),
            )
            cached = self._bias_cache.get(cache_key)
            if cached is not None:
                return cached

        hub_pos = torch.arange(-self.hub_size, 0, device=device, dtype=torch.float32)
        text_pos_k = torch.arange(0, text_k_len, device=device, dtype=torch.float32)
        pos_k = torch.cat((hub_pos, text_pos_k), dim=0).unsqueeze(0)

        if is_dense_with_hub and not has_past_kv:
            text_q_len = seq_len - self.hub_size
            if text_q_len < 0:
                raise ValueError(f"seq_len ({seq_len}) must be >= hub_size ({self.hub_size}) when is_dense_with_hub=True.")
            text_pos_q = torch.arange(0, text_q_len, device=device, dtype=torch.float32)
            pos_q = torch.cat((hub_pos, text_pos_q), dim=0).unsqueeze(1)
        else:
            pos_q = torch.arange(text_k_len - seq_len, text_k_len, device=device, dtype=torch.float32).unsqueeze(1)

        raw_dist = (pos_q - pos_k).clamp(min=0.0).view(1, 1, seq_len, num_all)
        active_slopes = self.slopes.to(device=device, dtype=torch.float32)

        if self.mode == "rational":
            log_bias = -torch.log((1.0 + active_slopes * raw_dist).clamp(min=1e-8))
        else:
            log_bias = -active_slopes * raw_dist

        if self.hub_size > 0:
            hub_pen: Any = self.hub_penalty
            hub_bias = -hub_pen.to(device=device, dtype=torch.float32).expand(1, self.num_heads, seq_len, self.hub_size)
            log_bias = torch.cat((hub_bias, log_bias[:, :, :, self.hub_size :]), dim=-1)

        out = log_bias.clamp(min=-10.0, max=0.0).to(dtype=dtype)
        if use_cache and cache_key is not None:
            if len(self._bias_cache) >= 16:
                self._bias_cache.clear()
            self._bias_cache[cache_key] = out
        return out

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, hub_size={self.hub_size}, "
            f"mode={self.mode!r}, learnable={self.learnable}"
        )


class Sinusoidal3DEmbedding(BaseModule):
    """Sinusoidal positional embedding for 3D discrete or continuous coordinates `(z, y, x)`."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if not _HAS_TORCH:
            raise ImportError("Sinusoidal3DEmbedding requires PyTorch.")
        if dim < 6 or dim % 2 != 0:
            raise ValueError(f"Sinusoidal3DEmbedding requires an even dim >= 6, got {dim}.")
        if theta <= 1.0:
            raise ValueError(f"theta must be > 1.0, got {theta}.")

        self.dim = int(dim)
        self.theta = float(theta)
        self.dim_z = max(2, 2 * (self.dim // 8))
        rem = self.dim - self.dim_z
        self.dim_y = max(2, 2 * (rem // 4))
        self.dim_x = self.dim - self.dim_z - self.dim_y
        if self.dim_x < 2:
            self.dim_y = max(2, self.dim_y - 2)
            self.dim_x = self.dim - self.dim_z - self.dim_y

        self.register_buffer(
            "inv_freq_z",
            1.0 / (self.theta ** (torch.arange(0, self.dim_z, 2, dtype=torch.float32) / self.dim_z)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_y",
            1.0 / (self.theta ** (torch.arange(0, self.dim_y, 2, dtype=torch.float32) / self.dim_y)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_x",
            1.0 / (self.theta ** (torch.arange(0, self.dim_x, 2, dtype=torch.float32) / self.dim_x)),
            persistent=False,
        )

    def forward(
        self,
        p_z: Any,
        p_y: Any,
        p_x: Any,
        dtype: Optional[Any] = None,
    ) -> Any:
        device = p_z.device
        inv_z = self.inv_freq_z.to(device=device)
        inv_y = self.inv_freq_y.to(device=device)
        inv_x = self.inv_freq_x.to(device=device)

        omega_z = p_z.to(device=device, dtype=torch.float32).unsqueeze(-1) * inv_z
        omega_y = p_y.to(device=device, dtype=torch.float32).unsqueeze(-1) * inv_y
        omega_x = p_x.to(device=device, dtype=torch.float32).unsqueeze(-1) * inv_x
        omega = torch.cat((omega_z, omega_y, omega_x), dim=-1)
        out = torch.cat((torch.cos(omega), torch.sin(omega)), dim=-1)
        if dtype is not None:
            return out.to(dtype=dtype)
        return out

    def extra_repr(self) -> str:
        return f"dim={self.dim}, splits=({self.dim_z}, {self.dim_y}, {self.dim_x}), theta={self.theta}"


__all__ = ["qrope", "QRoPE", "DistanceDecayBias", "Sinusoidal3DEmbedding"]
