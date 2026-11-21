from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
import torch

ROOT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(ROOT_PYTHON) not in sys.path:
    sys.path.insert(0, str(ROOT_PYTHON))

from luminamoon.nn import RaAttention, RaNorm, RaSwiGLU, ra_attention, ra_softmax


class TestRaAttention:
    """Tests for RaAttention multi-head attention module."""

    def test_attention_dimensions_and_output_shape(self) -> None:
        """Verify RaAttention preserves (B, N, C) shape across heads."""
        dim = 32
        num_heads = 4
        attn = RaAttention(dim=dim, num_heads=num_heads)
        x = torch.randn(2, 8, dim)
        out = attn(x)
        assert out.shape == (2, 8, dim)

    def test_causal_masking(self) -> None:
        """Verify causal masking runs and preserves output dimensionality."""
        dim = 16
        attn_causal = RaAttention(dim=dim, num_heads=2, is_causal=True)
        x = torch.randn(2, 6, dim)
        out = attn_causal(x)
        assert out.shape == (2, 6, dim)
        assert not torch.isnan(out).any()

    def test_qk_norm_integration(self) -> None:
        """Verify QK-Norm integration with RaNorm stabilizes attention."""
        dim = 16
        attn = RaAttention(dim=dim, num_heads=2, qk_norm=True)
        assert attn.q_norm is not None
        assert attn.k_norm is not None
        assert isinstance(attn.q_norm, RaNorm)

        x = torch.randn(2, 4, dim)
        out = attn(x)
        assert out.shape == (2, 4, dim)

    def test_removed_causal_kwarg_raises(self) -> None:
        """Verify removed v1 causal keyword is rejected in v2 API."""
        with pytest.raises(TypeError):
            RaAttention(dim=16, num_heads=2, causal=True)  # type: ignore[call-arg]

    def test_invalid_dimension_raises(self) -> None:
        """Verify ValueError when dim is not divisible by num_heads."""
        with pytest.raises(ValueError):
            RaAttention(dim=15, num_heads=4)



def _eager_ra_softmax_reference(scores: torch.Tensor, dim: int = -1, scale: float = 0.0625) -> torch.Tensor:
    max_scores = scores.max(dim=dim, keepdim=True)[0]
    delta = torch.where(scores == max_scores, 0.0, max_scores - scores)
    u = delta * scale
    num = torch.clamp(1.0 - u, min=0.0)
    den = 1.0 + u
    q = num / den
    q2 = q * q
    q4 = q2 * q2
    phi = q4 * q4
    sum_phi = phi.sum(dim=dim, keepdim=True)
    return phi / sum_phi


def _materialized_ra_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    scale: float | None = None,
    ra_scale: float = 0.0625,
) -> torch.Tensor:
    D = q.shape[-1]
    sm_scale = float(scale) if scale is not None else (1.0 / math.sqrt(D))
    scores = (q @ k.transpose(-2, -1)) * sm_scale
    if is_causal:
        N_Q, N_K = q.shape[-2], k.shape[-2]
        causal_mask = torch.triu(torch.full((N_Q, N_K), float("-inf"), device=q.device, dtype=q.dtype), diagonal=1)
        scores = scores + causal_mask
    if attn_mask is not None:
        scores = scores + attn_mask
    weights = _eager_ra_softmax_reference(scores, dim=-1, scale=ra_scale)
    return weights @ v


class TestFlashRaAttention:
    """Tests for fused tiled Rational Attention (ra_attention / Flash-RaAttention)."""

    @pytest.mark.parametrize("is_causal", [False, True])
    @pytest.mark.parametrize("seq_len", [16, 37, 70])
    def test_tiled_ra_attention_matches_materialized_reference(self, is_causal: bool, seq_len: int) -> None:
        from luminamoon.nn import ra_attention

        torch.manual_seed(2026)
        B, H, D = 2, 3, 16
        q_raw = torch.randn(B, H, seq_len, D, dtype=torch.float64) * 2.0
        k_raw = torch.randn(B, H, seq_len, D, dtype=torch.float64) * 2.0
        v_raw = torch.randn(B, H, seq_len, D, dtype=torch.float64)
        do = torch.randn_like(q_raw)

        q1 = q_raw.clone().requires_grad_(True)
        k1 = k_raw.clone().requires_grad_(True)
        v1 = v_raw.clone().requires_grad_(True)

        q2 = q_raw.clone().requires_grad_(True)
        k2 = k_raw.clone().requires_grad_(True)
        v2 = v_raw.clone().requires_grad_(True)

        out1 = ra_attention(q1, k1, v1, is_causal=is_causal, q_chunk_size=16)
        out2 = _materialized_ra_attention_reference(q2, k2, v2, is_causal=is_causal)

        assert torch.allclose(out1, out2, atol=1e-11, rtol=1e-10)

        out1.backward(do)
        out2.backward(do)

        assert q1.grad is not None and q2.grad is not None
        assert k1.grad is not None and k2.grad is not None
        assert v1.grad is not None and v2.grad is not None
        assert torch.allclose(q1.grad, q2.grad, atol=1e-10, rtol=1e-9)
        assert torch.allclose(k1.grad, k2.grad, atol=1e-10, rtol=1e-9)
        assert torch.allclose(v1.grad, v2.grad, atol=1e-10, rtol=1e-9)

    def test_no_n_squared_matrix_saved_in_autograd(self) -> None:
        """Verify that tiled ra_attention never saves an O(N^2) matrix in the autograd graph."""
        from luminamoon.nn import ra_attention

        B, H, N, D = 2, 4, 128, 16
        q = torch.randn(B, H, N, D, requires_grad=True)
        k = torch.randn(B, H, N, D, requires_grad=True)
        v = torch.randn(B, H, N, D, requires_grad=True)

        out = ra_attention(q, k, v, is_causal=True, q_chunk_size=32)
        for saved_t in out.grad_fn.saved_tensors:
            # No saved tensor should have shape (..., N, N)
            assert saved_t.shape[-2:] != (N, N), f"Unexpected O(N^2) saved tensor of shape {saved_t.shape}"

    def test_attn_mask_support_and_mask_gradients(self) -> None:
        from luminamoon.nn import ra_attention

        torch.manual_seed(3030)
        B, H, N, D = 2, 2, 35, 8
        q_raw = torch.randn(B, H, N, D, dtype=torch.float64)
        k_raw = torch.randn(B, H, N, D, dtype=torch.float64)
        v_raw = torch.randn(B, H, N, D, dtype=torch.float64)
        mask_raw = torch.randn(B, H, N, N, dtype=torch.float64) * 0.5
        do = torch.randn_like(q_raw)

        q1, k1, v1, m1 = (t.clone().requires_grad_(True) for t in (q_raw, k_raw, v_raw, mask_raw))
        q2, k2, v2, m2 = (t.clone().requires_grad_(True) for t in (q_raw, k_raw, v_raw, mask_raw))

        out1 = ra_attention(q1, k1, v1, attn_mask=m1, q_chunk_size=16)
        out2 = _materialized_ra_attention_reference(q2, k2, v2, attn_mask=m2)

        assert torch.allclose(out1, out2, atol=1e-11)
        out1.backward(do)
        out2.backward(do)

        assert torch.allclose(q1.grad, q2.grad, atol=1e-10)
        assert torch.allclose(k1.grad, k2.grad, atol=1e-10)
        assert torch.allclose(v1.grad, v2.grad, atol=1e-10)
        assert m1.grad is not None and m2.grad is not None
        assert torch.allclose(m1.grad, m2.grad, atol=1e-10)

    def test_low_precision_bf16_and_fp16_execution(self) -> None:
        for dt in (torch.bfloat16, torch.float16):
            attn = RaAttention(dim=32, num_heads=4, is_causal=True, qk_norm=True, dtype=dt)
            x = torch.randn(2, 40, 32, dtype=dt, requires_grad=True)
            out = attn(x)
            assert out.dtype == dt
            assert out.shape == (2, 40, 32)
            assert not torch.isnan(out).any()
            out.sum().backward()
            assert x.grad is not None
            assert not torch.isnan(x.grad).any()

    def test_flash_ra_attention_gradcheck_and_cpu_triton_guard(self) -> None:
        from luminamoon.nn.kernels import (
            triton_flash_ra_attention,
            triton_flash_ra_attention_backward,
            triton_ra_softmax_backward,
        )

        x_cpu = torch.randn(2, 2, 8, 8)
        assert triton_ra_softmax_backward(x_cpu, x_cpu, x_cpu) is None
        assert triton_flash_ra_attention(x_cpu, x_cpu, x_cpu) is None
        assert triton_flash_ra_attention_backward(x_cpu, x_cpu, x_cpu, x_cpu, x_cpu, x_cpu, x_cpu, x_cpu) is None

        torch.manual_seed(4040)
        q = (torch.randn(1, 2, 5, 4, dtype=torch.float64) * 1.5).requires_grad_(True)
        k = (torch.randn(1, 2, 5, 4, dtype=torch.float64) * 1.5).requires_grad_(True)
        v = torch.randn(1, 2, 5, 4, dtype=torch.float64).requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda q_, k_, v_: ra_attention(q_, k_, v_, is_causal=True, q_chunk_size=16),
            (q, k, v),
            eps=1e-6,
            atol=1e-4,
        )

    @pytest.mark.parametrize(
        "mask_shape",
        [
            (1, 35),
            (2, 1, 35),
            (2, 1, 1, 35),
            (2, 1, 35, 35),
            (1, 1, 35, 1),
        ],
    )
    def test_broadcast_attn_mask_shapes_and_gradients(self, mask_shape: tuple[int, ...]) -> None:
        from luminamoon.nn import ra_attention

        torch.manual_seed(5050)
        B, H, N_Q, N_K, D = 2, 3, 35, 35, 8
        q_raw = torch.randn(B, H, N_Q, D, dtype=torch.float64)
        k_raw = torch.randn(B, H, N_K, D, dtype=torch.float64)
        v_raw = torch.randn(B, H, N_K, D, dtype=torch.float64)
        mask_raw = torch.randn(*mask_shape, dtype=torch.float64) * 0.5
        do = torch.randn_like(q_raw)

        q1, k1, v1, m1 = (t.clone().requires_grad_(True) for t in (q_raw, k_raw, v_raw, mask_raw))
        q2, k2, v2, m2 = (t.clone().requires_grad_(True) for t in (q_raw, k_raw, v_raw, mask_raw))

        out1 = ra_attention(q1, k1, v1, attn_mask=m1, q_chunk_size=16)
        m2_view = m2.unsqueeze(1) if m2.ndim == 3 else m2
        out2 = _materialized_ra_attention_reference(q2, k2, v2, attn_mask=m2_view)

        assert torch.allclose(out1, out2, atol=1e-11)
        out1.backward(do)
        out2.backward(do)

        assert torch.allclose(q1.grad, q2.grad, atol=1e-10)
        assert torch.allclose(k1.grad, k2.grad, atol=1e-10)
        assert torch.allclose(v1.grad, v2.grad, atol=1e-10)
        assert m1.grad is not None and m2.grad is not None
        assert torch.allclose(m1.grad, m2.grad, atol=1e-10)

    def test_all_neg_inf_row_zero_gradient(self) -> None:
        from luminamoon.nn import ra_attention

        s = torch.tensor([[1.0, 2.0, 3.0], [float("-inf"), float("-inf"), float("-inf")]], dtype=torch.float64, requires_grad=True)
        p = ra_softmax(s, dim=-1)
        assert torch.allclose(p[1], torch.zeros(3, dtype=torch.float64))
        p.backward(torch.ones_like(p))
        assert s.grad is not None
        assert not torch.isnan(s.grad).any()
        assert torch.allclose(s.grad[1], torch.zeros(3, dtype=torch.float64))

        q = torch.randn(1, 1, 4, 8, dtype=torch.float64, requires_grad=True)
        k = torch.randn(1, 1, 4, 8, dtype=torch.float64, requires_grad=True)
        v = torch.randn(1, 1, 4, 8, dtype=torch.float64, requires_grad=True)
        mask = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
        mask[:, :, 2, :] = float("-inf")
        out = ra_attention(q, k, v, attn_mask=mask, q_chunk_size=16)
        assert torch.allclose(out[:, :, 2, :], torch.zeros(1, 1, 8, dtype=torch.float64))
        out.sum().backward()
        assert q.grad is not None
        assert torch.allclose(q.grad[:, :, 2, :], torch.zeros(1, 1, 8, dtype=torch.float64))

    def test_3d_input_with_dropout_and_3d_mask_and_asymmetric_dv(self) -> None:
        from luminamoon.nn import ra_attention

        # 1. 3D (B, N, D) with dropout_p > 0 and 3D attn_mask (B, N, N)
        q3 = torch.randn(2, 10, 8, requires_grad=True)
        k3 = torch.randn(2, 10, 8, requires_grad=True)
        v3 = torch.randn(2, 10, 8, requires_grad=True)
        m3 = torch.zeros(2, 10, 10)
        out3 = ra_attention(q3, k3, v3, attn_mask=m3, dropout_p=0.1, is_causal=True)
        assert out3.shape == (2, 10, 8)

        # 2. NumPy 4D with 3D attn_mask
        q_np = np.ones((2, 3, 6, 4), dtype=np.float32)
        m_np = np.zeros((2, 6, 6), dtype=np.float32)
        out_np = ra_attention(q_np, q_np, q_np, attn_mask=m_np)
        assert out_np.shape == (2, 3, 6, 4)

        # 3. Asymmetric value dimension D_v != D_qk
        B, H, N_Q, N_K, D_QK, D_V = 2, 2, 20, 24, 8, 12
        q_raw = torch.randn(B, H, N_Q, D_QK, dtype=torch.float64)
        k_raw = torch.randn(B, H, N_K, D_QK, dtype=torch.float64)
        v_raw = torch.randn(B, H, N_K, D_V, dtype=torch.float64)
        do = torch.randn(B, H, N_Q, D_V, dtype=torch.float64)

        q1, k1, v1 = (t.clone().requires_grad_(True) for t in (q_raw, k_raw, v_raw))
        q2, k2, v2 = (t.clone().requires_grad_(True) for t in (q_raw, k_raw, v_raw))

        out1 = ra_attention(q1, k1, v1, q_chunk_size=16)
        out2 = _materialized_ra_attention_reference(q2, k2, v2)
        assert out1.shape == (B, H, N_Q, D_V)
        assert torch.allclose(out1, out2, atol=1e-11)

        out1.backward(do)
        out2.backward(do)
        assert torch.allclose(q1.grad, q2.grad, atol=1e-10)
        assert torch.allclose(k1.grad, k2.grad, atol=1e-10)
        assert torch.allclose(v1.grad, v2.grad, atol=1e-10)


    def test_torch_compile_fullgraph_parity(self) -> None:
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile is not available in this PyTorch version")

        from luminamoon.nn import ra_attention

        torch.manual_seed(404)
        for train_mode in (True, False):
            norm = RaNorm(16)
            glu = RaSwiGLU(16, 32)
            attn = RaAttention(16, num_heads=2, qk_norm=True, is_causal=True)
            norm.train(train_mode)
            glu.train(train_mode)
            attn.train(train_mode)

            c_norm = torch.compile(norm, backend="aot_eager", fullgraph=True)
            c_glu = torch.compile(glu, backend="aot_eager", fullgraph=True)
            c_attn = torch.compile(attn, backend="aot_eager", fullgraph=True)

            x_raw = torch.randn(2, 8, 16)
            if train_mode:
                x_ref = x_raw.clone().requires_grad_(True)
                x_cmp = x_raw.clone().requires_grad_(True)
                y_ref = attn(glu(norm(x_ref)))
                y_cmp = c_attn(c_glu(c_norm(x_cmp)))
                assert torch.allclose(y_ref, y_cmp, atol=1e-5)
                y_ref.sum().backward()
                y_cmp.sum().backward()
                assert x_ref.grad is not None and x_cmp.grad is not None
                assert torch.allclose(x_ref.grad, x_cmp.grad, atol=1e-5)
            else:
                with torch.no_grad():
                    y_ref = attn(glu(norm(x_raw)))
                    y_cmp = c_attn(c_glu(c_norm(x_raw)))
                    assert torch.allclose(y_ref, y_cmp, atol=1e-5)

            c_sm = torch.compile(lambda t: ra_softmax(t, dim=-1), backend="aot_eager", fullgraph=True)
            c_fa = torch.compile(lambda t: ra_attention(t, t, t, is_causal=True), backend="aot_eager", fullgraph=True)

            s_raw = torch.randn(2, 2, 8, 8)
            if train_mode:
                s_ref = s_raw.clone().requires_grad_(True)
                s_cmp = s_raw.clone().requires_grad_(True)
                p_ref = ra_softmax(s_ref, dim=-1)
                p_cmp = c_sm(s_cmp)
                assert torch.allclose(p_ref, p_cmp, atol=1e-6)
                p_ref.sum().backward()
                p_cmp.sum().backward()
                assert s_ref.grad is not None and s_cmp.grad is not None
                assert torch.allclose(s_ref.grad, s_cmp.grad, atol=1e-5)

                q_ref = s_raw.clone().requires_grad_(True)
                q_cmp = s_raw.clone().requires_grad_(True)
                o_ref = ra_attention(q_ref, q_ref, q_ref, is_causal=True)
                o_cmp = c_fa(q_cmp)
                assert torch.allclose(o_ref, o_cmp, atol=1e-5)
                o_ref.sum().backward()
                o_cmp.sum().backward()
                assert q_ref.grad is not None and q_cmp.grad is not None
                assert torch.allclose(q_ref.grad, q_cmp.grad, atol=1e-5)
            else:
                with torch.no_grad():
                    assert torch.allclose(ra_softmax(s_raw, dim=-1), c_sm(s_raw), atol=1e-6)
                    assert torch.allclose(ra_attention(s_raw, s_raw, s_raw, is_causal=True), c_fa(s_raw), atol=1e-5)



class TestGroupedQueryRaAttention:
    """Tests for Native Grouped-Query Attention (GQA)."""

    def test_gqa_mixed_precision_dtype_safety(self) -> None:
        attn = RaAttention(dim=32, num_heads=4, num_kv_heads=2, qk_norm=True, is_causal=True).to(torch.float32)
        x = torch.randn(2, 8, 32, dtype=torch.float32)
        out = attn(x)
        assert out.dtype == torch.float32
        loss = out.sum()
        loss.backward()
        assert attn.q_proj.weight.grad is not None

        # Direct functional test with mixed precision tensors (Float32 Q, Float16 V)
        q = torch.randn(2, 4, 8, 8, dtype=torch.float32, requires_grad=True)
        k = torch.randn(2, 2, 8, 8, dtype=torch.float32, requires_grad=True)
        v = torch.randn(2, 2, 8, 8, dtype=torch.float16, requires_grad=True)
        out_fn = ra_attention(q, k, v, is_causal=True)
        loss_fn = out_fn.sum()
        loss_fn.backward()
        assert q.grad is not None and v.grad is not None

    @pytest.mark.parametrize("q_chunk_size", [None, 7])
    @pytest.mark.parametrize("is_causal", [False, True])
    def test_native_gqa_forward_and_backward_parity_against_repeat_interleave(
        self,
        q_chunk_size: Any,
        is_causal: bool,
    ) -> None:
        torch.manual_seed(2027)
        B, H_q, H_kv, N, D = 2, 6, 2, 18, 12
        g = H_q // H_kv

        q_raw = torch.randn(B, H_q, N, D, dtype=torch.float64)
        k_raw = torch.randn(B, H_kv, N, D, dtype=torch.float64)
        v_raw = torch.randn(B, H_kv, N, D, dtype=torch.float64)
        do = torch.randn_like(q_raw)

        q1, k1, v1 = (t.clone().requires_grad_(True) for t in (q_raw, k_raw, v_raw))
        q2, k2, v2 = (t.clone().requires_grad_(True) for t in (q_raw, k_raw, v_raw))

        # Native GQA without repeat_interleave
        out_gqa = ra_attention(q1, k1, v1, is_causal=is_causal, q_chunk_size=q_chunk_size)

        # Reference expanded MHA using repeat_interleave
        k2_rep = k2.repeat_interleave(g, dim=1)
        v2_rep = v2.repeat_interleave(g, dim=1)
        out_ref = _materialized_ra_attention_reference(q2, k2_rep, v2_rep, is_causal=is_causal)

        assert out_gqa.shape == (B, H_q, N, D)
        assert torch.allclose(out_gqa, out_ref, atol=1e-11, rtol=1e-10)

        out_gqa.backward(do)
        out_ref.backward(do)

        assert q1.grad is not None and q2.grad is not None
        assert k1.grad is not None and k2.grad is not None
        assert v1.grad is not None and v2.grad is not None
        assert torch.allclose(q1.grad, q2.grad, atol=1e-10, rtol=1e-9)
        assert torch.allclose(k1.grad, k2.grad, atol=1e-10, rtol=1e-9)
        assert torch.allclose(v1.grad, v2.grad, atol=1e-10, rtol=1e-9)

        # Verify NumPy GQA parity
        out_np = ra_attention(
            q_raw.numpy().astype(np.float32),
            k_raw.numpy().astype(np.float32),
            v_raw.numpy().astype(np.float32),
            is_causal=is_causal,
        )
        np.testing.assert_allclose(out_np, out_ref.detach().numpy().astype(np.float32), atol=1e-5, rtol=1e-5)

    def test_ra_attention_gqa_and_torch_compile(self) -> None:
        torch.manual_seed(8088)
        B, T, D = 2, 12, 16
        x = torch.randn(B, T, D, requires_grad=True)

        attn = RaAttention(dim=D, num_heads=4, num_kv_heads=2, is_causal=True, qk_norm=True)
        assert "num_kv_heads=2" in attn.extra_repr()
        assert attn.k_proj.out_features == 8

        out = attn(x)
        assert out.shape == (B, T, D)
        out.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

        if hasattr(torch, "compile"):
            c_attn = torch.compile(attn, backend="aot_eager", fullgraph=True)
            x1 = x.detach().clone().requires_grad_(True)
            x2 = x.detach().clone().requires_grad_(True)
            o1 = attn(x1)
            o2 = c_attn(x2)
            assert torch.allclose(o1, o2, atol=1e-5)
            o1.sum().backward()
            o2.sum().backward()
            assert x1.grad is not None and x2.grad is not None
            assert torch.allclose(x1.grad, x2.grad, atol=1e-5)


class TestRaGradcheckAndBoundaries:
    """Explicit gradcheck and float32/bfloat16 compact-support boundary tests."""

    def test_ra_softmax_gradcheck_and_shift_invariance(self) -> None:
        scores = torch.tensor(
            [
                [20.0, 19.2, 10.0, 4.5, 3.0, -5.0],
                [5.0, 4.9, 4.0, -11.5, -20.0, 0.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        assert torch.autograd.gradcheck(
            lambda t: ra_softmax(t, dim=-1, scale=0.0625),
            (scores,),
            eps=1e-6,
            atol=1e-4,
        )

        x = torch.tensor([[10.0, 8.5, 2.0, -10.0, -25.0]], dtype=torch.float64, requires_grad=True)
        w = torch.tensor([[0.3, -1.2, 0.7, 2.5, -0.4]], dtype=torch.float64)
        (ra_softmax(x, dim=-1, scale=0.0625) * w).sum().backward()
        assert x.grad is not None
        assert x.grad.sum().abs().item() < 1e-12

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_ra_softmax_compact_support_boundary_finite(self, dtype: torch.dtype) -> None:
        u_edge = float(np.nextafter(np.float32(1.0), np.float32(0.0)))
        deltas = [0.0, 15.999, 15.9999, 16.0 - 1e-6, 16.0 - 1e-7, u_edge * 16.0, 16.0, 16.0001, 16.001, 20.0]
        scores = torch.tensor([[16.0 - d for d in deltas]], dtype=dtype, requires_grad=True)
        probs = ra_softmax(scores, dim=-1, scale=0.0625)
        grad_w = torch.linspace(-1.0, 1.0, len(deltas), dtype=dtype).unsqueeze(0)
        (probs * grad_w).sum().backward()
        assert scores.grad is not None
        assert torch.isfinite(probs).all()
        assert torch.isfinite(scores.grad).all()

    @pytest.mark.parametrize("is_causal", [False, True])
    def test_ra_attention_gradcheck_mha_and_gqa(self, is_causal: bool) -> None:
        torch.manual_seed(150)
        q = (torch.randn(1, 2, 5, 4, dtype=torch.float64) * 1.2).requires_grad_(True)
        k = (torch.randn(1, 2, 5, 4, dtype=torch.float64) * 1.2).requires_grad_(True)
        v = (torch.randn(1, 2, 5, 4, dtype=torch.float64) * 1.2).requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda q_, k_, v_: ra_attention(q_, k_, v_, is_causal=is_causal),
            (q, k, v),
            eps=1e-6,
            atol=1e-4,
        )
        kg = (torch.randn(1, 1, 5, 4, dtype=torch.float64) * 1.2).requires_grad_(True)
        vg = (torch.randn(1, 1, 5, 4, dtype=torch.float64) * 1.2).requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda q_, k_, v_: ra_attention(q_, k_, v_, is_causal=is_causal),
            (q, kg, vg),
            eps=1e-6,
            atol=1e-4,
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_ra_attention_compact_support_boundary_finite(self, dtype: torch.dtype) -> None:
        scale = 1.0 / math.sqrt(2.0)
        q = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=dtype, requires_grad=True)
        k_vals = [16.0 / scale, (16.0 - 15.99999) / scale, 0.0, -0.001 / scale]
        k = torch.tensor([[[[kv, 0.0] for kv in k_vals]]], dtype=dtype, requires_grad=True)
        v = torch.randn(1, 1, 4, 2, dtype=dtype, requires_grad=True)
        out = ra_attention(q, k, v, is_causal=False)
        out.sum().backward()
        assert q.grad is not None and k.grad is not None and v.grad is not None
        assert all(torch.isfinite(t).all() for t in (out, q.grad, k.grad, v.grad))




