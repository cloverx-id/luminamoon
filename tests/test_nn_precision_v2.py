from __future__ import annotations

import math
import pytest
import torch

from luminamoon.nn import (
    RaMoE,
    linear_cross_entropy,
    pos_ra_gate,
    ra_gelu,
    ra_glu,
    ra_swiglu,
    ra_tanh,
)


def test_v2_precision_hardening_fp16_lce_bf16_gelu_extreme_fp32_and_moe_bound() -> None:
    # 1. RaGELU BF16 negative-tail cancellation-free verification
    x_tail_f64 = torch.tensor([-10.0, -50.0, -5000.0], dtype=torch.float64)
    x_tail_bf16 = x_tail_f64.to(torch.bfloat16)
    ref_gelu = (x_tail_f64 * pos_ra_gate(math.sqrt(2.0 / math.pi) * x_tail_f64, b=1.0)).to(torch.bfloat16)
    out_gelu = ra_gelu(x_tail_bf16)
    assert (out_gelu != 0.0).all()
    assert torch.equal(out_gelu, ref_gelu)

    # 2. Extreme FP32 (|z| = 1e20..1e30) NaN-free forward & backward across RaTanh, RaSwiGLU, RaGLU, pos_ra_gate
    devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    for dev in devices:
        z_ext = torch.tensor(
            [-1.0e30, -1.0e20, -2.0, 0.0, 2.0, 1.0e20, 1.0e30],
            device=dev,
            dtype=torch.float32,
            requires_grad=True,
        )
        u_ext = torch.ones_like(z_ext, requires_grad=True)
        g_gate = pos_ra_gate(z_ext, b=1.5)
        assert torch.isfinite(g_gate).all()
        y_tanh = ra_tanh(z_ext, b=1.5)
        y_tanh.sum().backward()
        assert z_ext.grad is not None and torch.isfinite(y_tanh).all() and torch.isfinite(z_ext.grad).all()

        z_ext.grad = None
        y_swi = ra_swiglu(z_ext, u_ext, b=1.5)
        y_swi.sum().backward()
        assert z_ext.grad is not None and u_ext.grad is not None
        assert torch.isfinite(y_swi).all() and torch.isfinite(z_ext.grad).all() and torch.isfinite(u_ext.grad).all()

        z_ext.grad = None
        u_ext.grad = None
        y_glu = ra_glu(z_ext, u_ext, b=1.5)
        y_glu.sum().backward()
        assert z_ext.grad is not None and u_ext.grad is not None
        assert torch.isfinite(y_glu).all() and torch.isfinite(z_ext.grad).all() and torch.isfinite(u_ext.grad).all()

        # 3. LinearCrossEntropy FP16 large-vocab non-target gradient preservation
        torch.manual_seed(42)
        M, D, V = 256, 32, 32000
        x_f32 = (torch.randn(M, D, device=dev, dtype=torch.float32) * 0.1).to(torch.float16).to(torch.float32).requires_grad_(True)
        w_f32 = (torch.randn(V, D, device=dev, dtype=torch.float32) * 0.02).to(torch.float16).to(torch.float32).requires_grad_(True)
        b_f32 = torch.zeros(V, device=dev, dtype=torch.float32, requires_grad=True)
        tgt = torch.randint(0, V, (M,), device=dev, dtype=torch.long)

        x_f16 = x_f32.detach().to(torch.float16).requires_grad_(True)
        w_f16 = w_f32.detach().to(torch.float16).requires_grad_(True)
        b_f16 = b_f32.detach().to(torch.float16).requires_grad_(True)

        loss_ref = linear_cross_entropy(x_f32, w_f32, tgt, bias=b_f32, reduction="mean")
        loss_ref.backward()
        loss_f16 = linear_cross_entropy(x_f16, w_f16, tgt, bias=b_f16, reduction="mean")
        loss_f16.backward()

        assert x_f16.grad is not None and w_f16.grad is not None and b_f16.grad is not None
        assert abs(float(b_f16.grad.float().sum().item())) < 2e-3
        assert torch.allclose(x_f16.grad.float(), x_f32.grad, atol=1e-3, rtol=2e-2)

    # 4. RaMoE optional router_logit_bound dead-expert safeguard
    with pytest.raises(ValueError):
        RaMoE(8, num_experts=3, top_k=1, router_logit_bound=0.0)
    moe_b = RaMoE(8, 16, 8, num_experts=3, top_k=1, router_logit_bound=6.0, return_aux_loss=True)
    with torch.no_grad():
        moe_b.router.weight.zero_()
        moe_b.router.weight[0, 0] = 50.0
        moe_b.router.weight[2, 0] = -50.0
    xb = torch.ones(4, 8, dtype=torch.float32, requires_grad=True)
    _, aux_b = moe_b(xb)
    aux_b.backward()
    assert moe_b.router.weight.grad is not None
    assert float(moe_b.router.weight.grad[2].abs().sum().item()) > 0.0
