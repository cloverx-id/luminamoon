from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch

ROOT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(ROOT_PYTHON) not in sys.path:
    sys.path.insert(0, str(ROOT_PYTHON))

from luminamoon.nn import RaBatchNorm2d, RaGELU, ra_gelu
from luminamoon.nn.kernels import (
    HAS_TRITON,
    triton_ra_batch_norm2d,
    triton_ra_batch_norm2d_backward,
    triton_ra_gelu,
    triton_ra_gelu_backward,
)

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def _eager_ra_gelu_ref(x: torch.Tensor) -> torch.Tensor:
    xf = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    z = _SQRT_2_OVER_PI * xf
    az = z.abs()
    inv_m = 1.0 / torch.clamp(az, min=1.0)
    ub, ua = inv_m, az * inv_m
    q = 0.5 * (ub * ub) / (ub * (ub + ua) + ua * ua)
    gate = torch.where(z >= 0.0, 1.0 - q, q)
    return (xf * gate).to(x.dtype)


def _eager_ra_bn2d_ref(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    running_mean: torch.Tensor | None,
    running_mad: torch.Tensor | None,
    *,
    training: bool,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xf = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    if training:
        mean = xf.mean(dim=(0, 2, 3), keepdim=True)
        mad = (xf - mean).abs().mean(dim=(0, 2, 3), keepdim=True)
    else:
        assert running_mean is not None and running_mad is not None
        mean = running_mean.view(1, -1, 1, 1).to(xf.dtype)
        mad = running_mad.view(1, -1, 1, 1).to(xf.dtype)
    y = (xf - mean) / (mad + eps)
    if weight is not None and bias is not None:
        y = y * weight.view(1, -1, 1, 1).to(xf.dtype) + bias.view(1, -1, 1, 1).to(xf.dtype)
    return y.to(x.dtype), mean.reshape(-1), mad.reshape(-1)


class TestRaGELUAndRaBatchNorm2d:
    """CPU gradcheck, compile, edge-case, and CUDA Triton parity tests for RaGELU and RaBatchNorm2d."""

    def test_cpu_triton_guards_return_none(self) -> None:
        x = torch.randn(2, 4, 8, 8)
        assert triton_ra_gelu(x) is None
        assert triton_ra_gelu_backward(x, x) is None
        assert triton_ra_batch_norm2d(x, None, None, None, None, training=True) is None
        assert triton_ra_batch_norm2d_backward(
            x, x, torch.zeros(4), torch.ones(4), None, training=True, need_dx=True, need_dw=False, need_db=False
        ) is None

    def test_ra_gelu_gradcheck_and_extreme_values(self) -> None:
        x = torch.tensor([-4.5, -1.2, -0.15, 0.2, 1.3, 4.8], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(ra_gelu, (x,), eps=1e-6, atol=1e-5)

        for dt in (torch.float32, torch.bfloat16):
            xe = torch.tensor([-1e20, -1e6, -1.0, 0.0, 1.0, 1e6, 1e20], dtype=dt, requires_grad=True)
            ye = RaGELU()(xe)
            assert torch.isfinite(ye).all()
            ye.sum().backward()
            assert xe.grad is not None and torch.isfinite(xe.grad).all()

    @pytest.mark.parametrize("training", [True, False])
    @pytest.mark.parametrize("affine", [True, False])
    def test_ra_batch_norm2d_gradcheck_and_single_channel(self, training: bool, affine: bool) -> None:
        torch.manual_seed(11)
        bn = RaBatchNorm2d(3, affine=affine, eps=1e-5).double()
        if not training:
            bn.eval()
        else:
            bn.train()
        # Keep inputs away from exact channel mean so |x - mean| is smooth for finite differences.
        x = (torch.randn(2, 3, 2, 2, dtype=torch.float64) * 1.5 + 0.37).requires_grad_(True)
        assert torch.autograd.gradcheck(lambda t: bn(t), (x,), eps=1e-6, atol=1e-4)

        bn1 = RaBatchNorm2d(1)
        y1 = bn1(torch.randn(2, 1, 4, 4, requires_grad=True))
        assert y1.shape == (2, 1, 4, 4)
        assert bn1.running_mean.shape == (1,) and bn1.running_mad.shape == (1,)

        with pytest.raises(ValueError, match="channels"):
            bn1(torch.randn(2, 2, 4, 4))

    def test_ra_batch_norm2d_autograd_eager_fallback_when_triton_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import luminamoon.nn.norm as norm_mod

        torch.manual_seed(66)
        x = torch.randn(2, 4, 6, 6, dtype=torch.float32, requires_grad=True)
        bn = RaBatchNorm2d(4, affine=True, eps=1e-5)
        monkeypatch.setattr(norm_mod, "triton_ra_batch_norm2d", lambda *a, **kw: None)
        monkeypatch.setattr(norm_mod, "triton_ra_batch_norm2d_backward", lambda *a, **kw: None)

        y = norm_mod._RaBatchNorm2dFunction.apply(
            x, bn.weight, bn.bias, bn.running_mean, bn.running_mad,
            bn.num_batches_tracked, True, True, bn.momentum, bn.eps,
        )
        y.sum().backward()

        x_ref = x.detach().clone().requires_grad_(True)
        w_ref = bn.weight.detach().clone().requires_grad_(True)
        b_ref = bn.bias.detach().clone().requires_grad_(True)
        y_ref, _, _ = _eager_ra_bn2d_ref(x_ref, w_ref, b_ref, None, None, training=True, eps=1e-5)
        y_ref.sum().backward()
        assert torch.allclose(y, y_ref, atol=1e-5)
        assert x.grad is not None and x_ref.grad is not None and torch.allclose(x.grad, x_ref.grad, atol=1e-5)
        assert bn.weight.grad is not None and w_ref.grad is not None and torch.allclose(bn.weight.grad, w_ref.grad, atol=1e-5)
        assert bn.bias.grad is not None and b_ref.grad is not None and torch.allclose(bn.bias.grad, b_ref.grad, atol=1e-5)

    def test_torch_compile_fullgraph_vision_layers(self) -> None:
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile unavailable")
        torch.manual_seed(33)
        seq = torch.nn.Sequential(RaBatchNorm2d(8), RaGELU())
        c_seq = torch.compile(seq, backend="aot_eager", fullgraph=True)
        x1 = torch.randn(2, 8, 6, 6, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        y1 = seq(x1)
        # Reset running stats so eager & compiled compare from identical initial state.
        seq[0].running_mean.zero_()
        seq[0].running_mad.fill_(1.0)
        seq[0].num_batches_tracked.zero_()
        y2 = c_seq(x2)
        assert torch.allclose(y1, y2, atol=1e-5)
        y1.sum().backward()
        y2.sum().backward()
        assert x1.grad is not None and x2.grad is not None
        assert torch.allclose(x1.grad, x2.grad, atol=1e-5)

    @pytest.mark.skipif(not (torch.cuda.is_available() and HAS_TRITON), reason="Requires CUDA + Triton")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_cuda_triton_ra_gelu_parity(self, dtype: torch.dtype) -> None:
        torch.manual_seed(44)
        x_raw = torch.randn(8, 32, 16, 16, device="cuda", dtype=dtype) * 2.5
        dy = torch.randn_like(x_raw)
        x1 = x_raw.clone().requires_grad_(True)
        x2 = x_raw.clone().requires_grad_(True)
        y1 = ra_gelu(x1)
        y2 = _eager_ra_gelu_ref(x2)
        atol = 1e-5 if dtype == torch.float32 else 2e-2
        assert torch.allclose(y1, y2, atol=atol, rtol=atol)
        y1.backward(dy)
        y2.backward(dy)
        assert x1.grad is not None and x2.grad is not None
        assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=atol)

    @pytest.mark.skipif(not (torch.cuda.is_available() and HAS_TRITON), reason="Requires CUDA + Triton")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("training", [True, False])
    @pytest.mark.parametrize("affine", [True, False])
    def test_cuda_triton_ra_batch_norm2d_parity(self, dtype: torch.dtype, training: bool, affine: bool) -> None:
        torch.manual_seed(55)
        # Exercises multi-split reduction (C=64, M=16*64*64=65536).
        x_raw = torch.randn(16, 64, 64, 64, device="cuda", dtype=dtype) * 1.7 + 0.4
        dy = torch.randn_like(x_raw)

        bn = RaBatchNorm2d(64, affine=affine, eps=1e-5, device="cuda", dtype=torch.float32)
        if affine:
            with torch.no_grad():
                bn.weight.copy_(torch.randn(64, device="cuda") * 0.5 + 1.0)
                bn.bias.copy_(torch.randn(64, device="cuda") * 0.2)
        with torch.no_grad():
            bn.running_mean.copy_(torch.randn(64, device="cuda") * 0.3)
            bn.running_mad.copy_(torch.rand(64, device="cuda") + 0.5)
        bn.train(training)

        rm0 = bn.running_mean.clone()
        rmad0 = bn.running_mad.clone()
        x1 = x_raw.clone().requires_grad_(True)
        y1 = bn(x1)
        y1.backward(dy)

        x2 = x_raw.clone().requires_grad_(True)
        w2 = bn.weight.detach().clone().requires_grad_(True) if affine else None
        b2 = bn.bias.detach().clone().requires_grad_(True) if affine else None
        y2, m_ref, mad_ref = _eager_ra_bn2d_ref(x2, w2, b2, rm0, rmad0, training=training, eps=1e-5)
        y2.backward(dy)

        atol = 1e-4 if dtype == torch.float32 else 3e-2
        assert torch.allclose(y1, y2, atol=atol, rtol=atol)
        assert x1.grad is not None and x2.grad is not None
        assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=atol)
        if affine:
            assert bn.weight.grad is not None and w2 is not None and w2.grad is not None
            assert bn.bias.grad is not None and b2 is not None and b2.grad is not None
            g_atol = 1e-2 if dtype == torch.float32 else 5e-1
            assert torch.allclose(bn.weight.grad, w2.grad, atol=g_atol, rtol=1e-2)
            assert torch.allclose(bn.bias.grad, b2.grad, atol=g_atol, rtol=1e-2)
        if training:
            assert torch.allclose(bn.running_mean, rm0 * 0.9 + m_ref * 0.1, atol=1e-4)
            assert torch.allclose(bn.running_mad, rmad0 * 0.9 + mad_ref * 0.1, atol=1e-4)
