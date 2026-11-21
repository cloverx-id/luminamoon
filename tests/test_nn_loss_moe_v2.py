from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
import torch

ROOT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(ROOT_PYTHON) not in sys.path:
    sys.path.insert(0, str(ROOT_PYTHON))

from luminamoon.nn import (
    LinearCrossEntropy,
    RaAttention,
    RaBatchNorm2d,
    RaGLU,
    RaMoE,
    RaNorm,
    RaSwiGLU,
    linear_cross_entropy,
    permute_tokens_by_expert,
    ra_attention,
    ra_glu,
    ra_softmax,
    ra_swiglu,
    ra_tanh,
)


class TestLinearCrossEntropy:
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("has_bias", [False, True])
    @pytest.mark.parametrize("label_smoothing", [0.0, 0.15])
    @pytest.mark.parametrize("logit_scale", [1.0, 0.75])
    def test_forward_backward_exact_parity_against_materialized_pytorch(
        self,
        reduction: str,
        has_bias: bool,
        label_smoothing: float,
        logit_scale: float,
    ) -> None:
        torch.manual_seed(2026)
        B, S, D, V = 3, 7, 24, 53
        ignore_index = -100

        x_base = torch.randn(B, S, D, dtype=torch.float32)
        w_base = torch.randn(V, D, dtype=torch.float32)
        b_base = torch.randn(V, dtype=torch.float32) if has_bias else None
        target = torch.randint(0, V, (B, S), dtype=torch.long)
        target[0, 2] = ignore_index
        target[1, 5] = ignore_index
        target[2, 0] = ignore_index

        x1 = x_base.clone().requires_grad_(True)
        w1 = w_base.clone().requires_grad_(True)
        b1 = b_base.clone().requires_grad_(True) if b_base is not None else None

        x2 = x_base.clone().requires_grad_(True)
        w2 = w_base.clone().requires_grad_(True)
        b2 = b_base.clone().requires_grad_(True) if b_base is not None else None

        # Force multi-chunk execution with chunk_size=5 (< B * S = 21)
        loss_fused = linear_cross_entropy(
            x1,
            w1,
            target,
            bias=b1,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
            logit_scale=logit_scale,
            chunk_size=5,
        )

        logits_ref = torch.nn.functional.linear(x2, w2, b2) * logit_scale
        loss_ref = torch.nn.functional.cross_entropy(
            logits_ref.view(-1, V),
            target.view(-1),
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        if reduction == "none":
            loss_ref = loss_ref.view(B, S)

        assert loss_fused.shape == loss_ref.shape
        assert torch.allclose(loss_fused, loss_ref, atol=1e-5, rtol=1e-5)

        if reduction == "none":
            grad_out = torch.randn_like(loss_ref)
            loss_fused.backward(grad_out)
            loss_ref.backward(grad_out)
        else:
            loss_fused.backward()
            loss_ref.backward()

        assert x1.grad is not None and x2.grad is not None
        assert w1.grad is not None and w2.grad is not None
        assert torch.allclose(x1.grad, x2.grad, atol=1e-5, rtol=1e-5)
        assert torch.allclose(w1.grad, w2.grad, atol=1e-5, rtol=1e-5)
        if has_bias:
            assert b1 is not None and b2 is not None
            assert b1.grad is not None and b2.grad is not None
            assert torch.allclose(b1.grad, b2.grad, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_stability_and_gradients(self, dtype: Any) -> None:
        torch.manual_seed(77)
        B, S, D, V = 2, 6, 32, 64
        x_f32 = torch.randn(B, S, D)
        w_f32 = torch.randn(V, D) * 0.2
        b_f32 = torch.randn(V) * 0.1
        target = torch.randint(0, V, (B, S))
        target[0, 0] = -100

        x_lp = x_f32.to(dtype).detach().requires_grad_(True)
        w_lp = w_f32.to(dtype).detach().requires_grad_(True)
        b_lp = b_f32.to(dtype).detach().requires_grad_(True)

        x_ref = x_lp.detach().to(torch.float32).requires_grad_(True)
        w_ref = w_lp.detach().to(torch.float32).requires_grad_(True)
        b_ref = b_lp.detach().to(torch.float32).requires_grad_(True)

        loss_lp = linear_cross_entropy(
            x_lp, w_lp, target, bias=b_lp, label_smoothing=0.1, chunk_size=4
        )
        logits_ref = torch.nn.functional.linear(x_ref, w_ref, b_ref)
        loss_ref = torch.nn.functional.cross_entropy(
            logits_ref.view(-1, V), target.view(-1), ignore_index=-100, label_smoothing=0.1
        )

        assert loss_lp.dtype == dtype
        tol = 2e-2 if dtype == torch.bfloat16 else 5e-3
        assert torch.allclose(loss_lp.float(), loss_ref, atol=tol, rtol=tol)

        loss_lp.backward()
        loss_ref.backward()
        assert x_lp.grad is not None and w_lp.grad is not None and b_lp.grad is not None
        assert torch.isfinite(x_lp.grad).all()
        assert torch.isfinite(w_lp.grad).all()
        assert torch.isfinite(b_lp.grad).all()
        assert torch.allclose(x_lp.grad.float(), x_ref.grad, atol=tol, rtol=tol)
        assert torch.allclose(w_lp.grad.float(), w_ref.grad, atol=tol, rtol=tol)

    def test_all_ignored_targets_yield_zero_loss_and_zero_grad(self) -> None:
        x = torch.randn(4, 16, requires_grad=True)
        w = torch.randn(32, 16, requires_grad=True)
        b = torch.randn(32, requires_grad=True)
        target = torch.full((4,), -100, dtype=torch.long)

        for red in ("mean", "sum", "none"):
            x.grad = None
            w.grad = None
            b.grad = None
            loss = linear_cross_entropy(x, w, target, bias=b, reduction=red, label_smoothing=0.1)
            assert torch.isfinite(loss).all()
            assert (loss == 0.0).all()
            loss.sum().backward()
            assert x.grad is not None and (x.grad == 0.0).all()
            assert w.grad is not None and (w.grad == 0.0).all()
            assert b.grad is not None and (b.grad == 0.0).all()

    def test_stateful_and_stateless_module_modes(self) -> None:
        import luminamoon as lm

        assert lm.LinearCrossEntropy is LinearCrossEntropy
        assert lm.linear_cross_entropy is linear_cross_entropy

        torch.manual_seed(99)
        x = torch.randn(2, 5, 16, requires_grad=True)
        target = torch.randint(0, 25, (2, 5))

        # 1. Stateful LM Head + Loss Mode
        head = LinearCrossEntropy(16, 25, bias=True, label_smoothing=0.05)
        assert "in_features=16, vocab_size=25" in head.extra_repr()

        # Inference mode (target=None -> returns logits)
        logits = head(x)
        assert logits.shape == (2, 5, 25)

        # Training mode (target provided -> returns fused scalar loss)
        loss_stateful = head(x, target)
        loss_ref = torch.nn.functional.cross_entropy(
            logits.view(-1, 25), target.view(-1), label_smoothing=0.05
        )
        assert torch.allclose(loss_stateful, loss_ref, atol=1e-5)

        # Weight-tying override in stateful mode
        tied_w = torch.randn(25, 16, requires_grad=True)
        loss_tied = head(x, tied_w, target)
        loss_tied.backward()
        assert tied_w.grad is not None

        # 2. Stateless Criterion Mode
        criterion = LinearCrossEntropy(label_smoothing=0.05)
        loss_stateless = criterion(x, head.weight, target, bias=head.bias)
        assert torch.allclose(loss_stateless, loss_ref, atol=1e-5)

    def test_numpy_fallback_and_validation(self) -> None:
        rng = np.random.default_rng(123)
        x_np = rng.standard_normal((3, 4, 12)).astype(np.float32)
        w_np = rng.standard_normal((20, 12)).astype(np.float32)
        b_np = rng.standard_normal((20,)).astype(np.float32)
        t_np = rng.integers(0, 20, size=(3, 4), dtype=np.int64)
        t_np[0, 1] = -100

        for red in ("mean", "sum", "none"):
            loss_np = linear_cross_entropy(
                x_np, w_np, t_np, bias=b_np, reduction=red, label_smoothing=0.1, chunk_size=3
            )
            loss_pt = linear_cross_entropy(
                torch.from_numpy(x_np),
                torch.from_numpy(w_np),
                torch.from_numpy(t_np),
                bias=torch.from_numpy(b_np),
                reduction=red,
                label_smoothing=0.1,
                chunk_size=3,
            )
            np.testing.assert_allclose(loss_np, loss_pt.numpy(), atol=1e-5, rtol=1e-5)

        with pytest.raises(ValueError, match="reduction"):
            linear_cross_entropy(x_np, w_np, t_np, reduction="invalid")
        with pytest.raises(ValueError, match="label_smoothing"):
            linear_cross_entropy(x_np, w_np, t_np, label_smoothing=1.0)
        with pytest.raises(ValueError, match="logit_scale"):
            linear_cross_entropy(x_np, w_np, t_np, logit_scale=0.0)
        with pytest.raises(ValueError, match="Both in_features and vocab_size"):
            LinearCrossEntropy(in_features=16, vocab_size=None)

    def test_torch_compile_linear_cross_entropy(self) -> None:
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile is not available in this PyTorch version")

        torch.manual_seed(512)
        head = LinearCrossEntropy(16, 32, bias=True, label_smoothing=0.05)
        c_head = torch.compile(head, backend="aot_eager", fullgraph=True)
        c_fn = torch.compile(
            lambda x, w, t, b: linear_cross_entropy(x, w, t, bias=b, label_smoothing=0.05),
            backend="aot_eager",
            fullgraph=True,
        )

        x_raw = torch.randn(2, 6, 16)
        target = torch.randint(0, 32, (2, 6))
        target[0, 2] = -100

        x1 = x_raw.clone().requires_grad_(True)
        x2 = x_raw.clone().requires_grad_(True)
        l1 = head(x1, target)
        l2 = c_head(x2, target)
        assert torch.allclose(l1, l2, atol=1e-5)
        l1.backward()
        l2.backward()
        assert x1.grad is not None and x2.grad is not None
        assert torch.allclose(x1.grad, x2.grad, atol=1e-5)

        x3 = x_raw.clone().requires_grad_(True)
        l3 = c_fn(x3, head.weight, target, head.bias)
        assert torch.allclose(l1, l3, atol=1e-5)


class TestRaMoE:
    """Tests for RaMoE and Contiguous Token Permutation Dispatcher."""

    @pytest.mark.parametrize("top_k", [1, 2])
    @pytest.mark.parametrize("expert_type", ["ra_swiglu", "ra_glu"])
    def test_permutation_dispatcher_matches_unpermuted_reference_and_gradients(
        self,
        top_k: int,
        expert_type: str,
    ) -> None:
        torch.manual_seed(1337)
        B, T, D, H = 2, 7, 16, 32
        num_experts = 4

        moe = RaMoE(
            in_features=D,
            hidden_features=H,
            out_features=D,
            num_experts=num_experts,
            top_k=top_k,
            num_shared_experts=1,
            expert_type=expert_type,
            return_aux_loss=True,
        )
        assert "num_experts=4" in moe.extra_repr()

        x_raw = torch.randn(B, T, D)
        grad_out = torch.randn(B, T, D)

        x1 = x_raw.clone().requires_grad_(True)
        out_perm, aux_loss = moe(x1)
        assert out_perm.shape == (B, T, D)
        assert aux_loss.ndim == 0 and aux_loss.item() > 0.0
        assert moe.last_aux_loss is not None

        # Reference unpermuted evaluation using identical parameters
        x2 = x_raw.clone().requires_grad_(True)
        x2_2d = x2.reshape(-1, D)
        r_logits = moe.router(x2_2d)
        t_vals, t_idx = torch.topk(r_logits, k=top_k, dim=-1)
        if top_k == 1:
            f_probs = ra_softmax(r_logits, dim=-1, scale=moe.ra_scale)
            t_weights = f_probs.gather(dim=-1, index=t_idx)
        else:
            t_weights = ra_softmax(t_vals, dim=-1, scale=moe.ra_scale)

        ref_2d = torch.zeros_like(x2_2d)
        for e_idx in range(moe.num_experts):
            mask = t_idx == e_idx
            w_e = torch.where(mask, t_weights, torch.zeros_like(t_weights)).sum(dim=-1, keepdim=True)
            ref_2d = ref_2d + moe.eval_expert(e_idx, x2_2d) * w_e
        if moe.shared_experts is not None:
            for s_exp in moe.shared_experts:
                ref_2d = ref_2d + s_exp(x2_2d)
        out_ref = ref_2d.reshape(B, T, D)

        assert torch.allclose(out_perm, out_ref, atol=1e-5, rtol=1e-5)

        out_perm.backward(grad_out)
        out_ref.backward(grad_out)

        assert x1.grad is not None and x2.grad is not None
        assert torch.allclose(x1.grad, x2.grad, atol=1e-5, rtol=1e-5)
        assert moe.router.weight.grad is not None
        assert torch.isfinite(moe.router.weight.grad).all()

    def test_ra_moe_numpy_fallback_and_torch_compile(self) -> None:
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 6, 12)).astype(np.float32)

        moe_np = RaMoE(
            in_features=12,
            hidden_features=24,
            out_features=12,
            num_experts=4,
            top_k=2,
            num_shared_experts=1,
            return_aux_loss=True,
        )
        # Exercise NumPy execution path directly by passing ndarray
        out_np, aux_np = moe_np(x_np)
        assert out_np.shape == (2, 6, 12)
        assert np.isfinite(out_np).all()
        assert float(aux_np) > 0.0

        with torch.no_grad():
            out_pt, aux_pt = moe_np(torch.from_numpy(x_np))
        assert out_pt.shape == (2, 6, 12)
        np.testing.assert_allclose(out_np, out_pt.detach().numpy(), atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(float(aux_np), float(aux_pt.item()), atol=1e-5, rtol=1e-5)

        if hasattr(torch, "compile"):
            moe_cmp = RaMoE(12, 24, 12, num_experts=3, top_k=2)
            c_moe = torch.compile(moe_cmp, backend="aot_eager", fullgraph=True)
            x_t = torch.randn(2, 5, 12, requires_grad=True)
            x_t2 = x_t.detach().clone().requires_grad_(True)
            y1 = moe_cmp(x_t)
            y2 = c_moe(x_t2)
            assert torch.allclose(y1, y2, atol=1e-5)
            y1.sum().backward()
            y2.sum().backward()
            assert x_t.grad is not None and x_t2.grad is not None
            assert torch.allclose(x_t.grad, x_t2.grad, atol=1e-5)

    def test_moe_rejects_removed_dim_keyword(self) -> None:
        with pytest.raises(TypeError):
            RaMoE(dim=32, hidden_features=64, num_experts=4)  # type: ignore[call-arg]
        moe = RaMoE(32, hidden_features=64, num_experts=4, num_shared_experts=1)
        assert moe.in_features == 32
        x = torch.randn(2, 5, 32)
        y = moe(x)
        assert y.shape == (2, 5, 32)
        assert permute_tokens_by_expert is not None

    def test_ra_attention_dynamic_fast_path_matches_chunked_path(self) -> None:
        torch.manual_seed(909)
        q_raw = torch.randn(2, 2, 24, 16, dtype=torch.float64)
        k_raw = torch.randn(2, 2, 24, 16, dtype=torch.float64)
        v_raw = torch.randn(2, 2, 24, 16, dtype=torch.float64)
        do = torch.randn_like(q_raw)

        q1, k1, v1 = (t.clone().requires_grad_(True) for t in (q_raw, k_raw, v_raw))
        q2, k2, v2 = (t.clone().requires_grad_(True) for t in (q_raw, k_raw, v_raw))

        # q_chunk_size=None triggers the single-shot dynamic fast-path (2*2*24*24 = 2304 <= 4M)
        out_fast = ra_attention(q1, k1, v1, is_causal=True, q_chunk_size=None)
        # q_chunk_size=8 forces multi-chunk execution
        out_chunked = ra_attention(q2, k2, v2, is_causal=True, q_chunk_size=8)

        assert torch.allclose(out_fast, out_chunked, atol=1e-11, rtol=1e-10)
        out_fast.backward(do)
        out_chunked.backward(do)
        assert q1.grad is not None and q2.grad is not None
        assert torch.allclose(q1.grad, q2.grad, atol=1e-10, rtol=1e-9)
        assert torch.allclose(k1.grad, k2.grad, atol=1e-10, rtol=1e-9)
        assert torch.allclose(v1.grad, v2.grad, atol=1e-10, rtol=1e-9)


class TestNNParityAndConsistencyV2:
    """Regression tests for non-unit b gradients, FP16 softmax, NumPy parity, ND Rust SIMD, and API consistency."""

    def test_ra_tanh_gradcheck_non_unit_b(self) -> None:
        from luminamoon.nn.activation import _RaTanhFunction

        x = torch.tensor([0.0, -2.0, -0.5, 0.7, 3.0], dtype=torch.float64, requires_grad=True)
        for b in (0.5, 1.0, 2.5):
            assert torch.autograd.gradcheck(lambda t: ra_tanh(t, b=b), (x,), eps=1e-6, atol=1e-4)
            if _RaTanhFunction is not None:
                assert torch.autograd.gradcheck(
                    lambda t: _RaTanhFunction.apply(t, b), (x,), eps=1e-6, atol=1e-4
                )
                x0 = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
                _RaTanhFunction.apply(x0, b).sum().backward()
                assert x0.grad is not None
                assert abs(float(x0.grad.item()) - 1.0) < 1e-12

    def test_ra_softmax_fp16_all_neg_inf_no_nan(self) -> None:
        s = torch.tensor([[1.0, 2.0, 3.0], [float("-inf"), float("-inf"), float("-inf")]], dtype=torch.float16)
        p = ra_softmax(s, dim=-1)
        assert p.dtype == torch.float16
        assert torch.isfinite(p).all()
        assert torch.allclose(p[1], torch.zeros(3, dtype=torch.float16))

    def test_numpy_module_parity_glu_loss_attention(self) -> None:
        torch.manual_seed(2026)
        x_t = torch.randn(2, 5, 8, dtype=torch.float32)
        x_np = x_t.numpy()

        for mod in (
            RaSwiGLU(8, 16, out_features=4, bias=True, b=2.0),
            RaGLU(8, 16, out_features=4, bias=True, b=2.0),
        ):
            with torch.no_grad():
                y_pt = mod(x_t).numpy()
            y_np = mod(x_np)
            assert y_np.shape == (2, 5, 4)
            np.testing.assert_allclose(y_np, y_pt, atol=1e-5, rtol=1e-5)

        assert "b=2.0" in RaSwiGLU(8, 16, b=2.0).extra_repr()
        assert "b=2.0" in RaGLU(8, 16, b=2.0).extra_repr()
        moe_def = RaMoE(12, num_experts=2, top_k=1, b=2.5)
        assert moe_def.expert_type == "ra_swiglu" and moe_def.hidden_features == int(12 * 8 / 3)
        with pytest.raises(ValueError):
            RaMoE(8, num_experts=2, top_k=1, expert_type="pos_ra_glu")

        _, _, _, counts = permute_tokens_by_expert(
            torch.randn(6, 8), torch.tensor([[0], [1], [0], [1], [0], [1]]), torch.ones(6, 1), 2
        )
        assert isinstance(counts, torch.Tensor) and counts.dtype == torch.int64 and counts.tolist() == [3, 3]

        lce = LinearCrossEntropy(8, 16, bias=True, logit_scale=0.8)
        tgt_t = torch.randint(0, 16, (2, 5), dtype=torch.long)
        with torch.no_grad():
            logits_pt = lce(x_t).numpy()
            loss_pt = float(lce(x_t, tgt_t).item())
        np.testing.assert_allclose(lce(x_np), logits_pt, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(float(lce(x_np, tgt_t.numpy())), loss_pt, atol=1e-5, rtol=1e-5)

        attn = RaAttention(dim=16, num_heads=4, num_kv_heads=2, qk_norm=True, is_causal=True)
        attn.eval()
        xa_t = torch.randn(2, 6, 16, dtype=torch.float32)
        with torch.no_grad():
            oa_pt = attn(xa_t).numpy()
        oa_np = attn(xa_t.numpy())
        assert oa_np.shape == (2, 6, 16)
        np.testing.assert_allclose(oa_np, oa_pt, atol=1e-4, rtol=1e-4)

    def test_cpu_rust_simd_nd_ranorm_and_ra_gated(self) -> None:
        torch.manual_seed(77)
        norm = RaNorm(32, bias=True)
        with torch.no_grad():
            norm.weight.copy_(torch.randn(32))
            norm.bias.copy_(torch.randn(32))

        for shape in ((32,), (6, 32), (2, 8, 32)):
            x = torch.randn(*shape, dtype=torch.float32)
            with torch.no_grad():
                y_simd = norm(x)
                ref = (x / (x.abs().mean(dim=-1, keepdim=True) + norm.eps)) * norm.weight + norm.bias
            assert torch.allclose(y_simd, ref, atol=1e-5, rtol=1e-5)
            y_np = norm(x.numpy())
            np.testing.assert_allclose(y_np, ref.numpy(), atol=1e-5, rtol=1e-5)

        x_leaf = torch.randn(2, 4, 32, dtype=torch.float32, requires_grad=False)
        norm.zero_grad()
        norm(x_leaf).sum().backward()
        assert norm.weight.grad is not None and torch.isfinite(norm.weight.grad).all()

        g = torch.randn(4, 32, dtype=torch.float32)
        u = torch.randn(4, 32, dtype=torch.float32)
        for b in (1.0, 2.5):
            ref_swi = (g * (0.5 * (1.0 + ra_tanh(g, b=b) / b))) * u
            ref_glu = ra_tanh(g, b=b) * u
            with torch.no_grad():
                assert torch.allclose(ra_swiglu(g, u, b=b), ref_swi, atol=1e-5, rtol=1e-5)
                assert torch.allclose(ra_glu(g, u, b=b), ref_glu, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(ra_swiglu(g.numpy(), u.numpy(), b=b), ref_swi.numpy(), atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(ra_glu(g.numpy(), u.numpy(), b=b), ref_glu.numpy(), atol=1e-5, rtol=1e-5)

    def test_ra_batchnorm2d_momentum_none(self) -> None:
        torch.manual_seed(99)
        bn_pt = RaBatchNorm2d(4, momentum=None)
        bn_np = RaBatchNorm2d(4, momentum=None)
        x1 = torch.randn(2, 4, 4, 4)
        x2 = torch.randn(2, 4, 4, 4)
        y1_pt = bn_pt(x1)
        y2_pt = bn_pt(x2)
        y1_np = bn_np(x1.numpy())
        y2_np = bn_np(x2.numpy())
        assert y1_pt.shape == x1.shape and y2_pt.shape == x2.shape
        np.testing.assert_allclose(y1_np, y1_pt.detach().numpy(), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(y2_np, y2_pt.detach().numpy(), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(
            bn_np.running_mean.detach().numpy(), bn_pt.running_mean.detach().numpy(), atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(
            bn_np.running_mad.detach().numpy(), bn_pt.running_mad.detach().numpy(), atol=1e-5, rtol=1e-5
        )

    def test_meta_device_skips_rust_cpu_simd(self) -> None:
        with torch.no_grad():
            x_meta = torch.empty(2, 16, device="meta")
            assert RaNorm(16, device="meta")(x_meta).shape == (2, 16)
            assert ra_tanh(x_meta).shape == (2, 16)
            assert ra_swiglu(x_meta, x_meta).shape == (2, 16)
            assert ra_glu(x_meta, x_meta).shape == (2, 16)
            assert ra_softmax(x_meta).shape == (2, 16)
            assert RaMoE(16, 32, 16, num_experts=2, top_k=1, device="meta")(x_meta).shape == (2, 16)
            q_meta = torch.empty(1, 2, 4, 8, device="meta")
            assert ra_attention(q_meta, q_meta, q_meta).shape == (1, 2, 4, 8)

    def test_empty_moe_numpy_and_3d_numpy_attention(self) -> None:
        moe = RaMoE(16, 32, 8, num_experts=2, top_k=1, return_aux_loss=True)
        moe_out_np, aux_np = moe(np.empty((0, 16), dtype=np.float32))
        assert moe_out_np.shape == (0, 8) and float(aux_np) == 0.0
        with pytest.raises(ValueError):
            moe(np.empty((16,), dtype=np.float32))

        q3 = np.ones((2, 4, 8), dtype=np.float32)
        m4 = np.zeros((2, 1, 4, 4), dtype=np.float32)
        out3_np = ra_attention(q3, q3, q3, attn_mask=m4)
        out3_pt = ra_attention(torch.from_numpy(q3), torch.from_numpy(q3), torch.from_numpy(q3), attn_mask=torch.from_numpy(m4))
        assert out3_np.shape == tuple(out3_pt.shape) == (2, 4, 8)
        np.testing.assert_allclose(out3_np, out3_pt.numpy(), atol=1e-5, rtol=1e-5)
        with pytest.raises(ValueError):
            ra_attention(np.ones((4, 8), dtype=np.float32), np.ones((4, 8), dtype=np.float32), np.ones((4, 8), dtype=np.float32))

    def test_domain_validation_b_and_scale_and_cache_threshold(self) -> None:
        from luminamoon.nn import RaAttention, RaGLU, RaSwiGLU, RaTanh, pos_ra_gate
        from luminamoon.nn._attention_ops import (
            _CUDA_FAST_CACHE_ELEMENTS,
            _CUDA_FAST_ELEMENTS,
            _TRITON_ATTN_MIN_SEQ,
            _should_prefer_triton_attn,
        )

        assert _CUDA_FAST_CACHE_ELEMENTS == 16_777_216
        assert _CUDA_FAST_ELEMENTS == 134_217_728
        assert _TRITON_ATTN_MIN_SEQ == 64
        assert _should_prefer_triton_attn(4, 8, 512, 512, None)
        assert _should_prefer_triton_attn(4, 8, 2048, 2048, None)
        assert _should_prefer_triton_attn(4, 8, 1, 4096, None)
        assert _should_prefer_triton_attn(4, 8, 16, 16, 8)
        assert not _should_prefer_triton_attn(4, 8, 1, 1024, None)
        assert not _should_prefer_triton_attn(4, 8, 32, 32, None)

        x = torch.zeros(2, 4)
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                ra_tanh(x, b=bad)
            with pytest.raises(ValueError):
                RaTanh(b=bad)
            with pytest.raises(ValueError):
                pos_ra_gate(x, b=bad)
            with pytest.raises(ValueError):
                ra_swiglu(x, x, b=bad)
            with pytest.raises(ValueError):
                RaSwiGLU(4, b=bad)
            with pytest.raises(ValueError):
                ra_glu(x, x, b=bad)
            with pytest.raises(ValueError):
                RaGLU(4, b=bad)
            with pytest.raises(ValueError):
                ra_softmax(x, scale=bad)
            with pytest.raises(ValueError):
                ra_attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1), ra_scale=bad)
            with pytest.raises(ValueError):
                RaAttention(8, num_heads=2, ra_scale=bad)
            with pytest.raises(ValueError):
                RaMoE(8, num_experts=2, top_k=1, b=bad)
            with pytest.raises(ValueError):
                RaMoE(8, num_experts=2, top_k=1, ra_scale=bad)

    def test_cubecl_nn_threshold_and_dispatch_gating(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from types import SimpleNamespace
        from luminamoon.nn import RaNorm, _compat, activation, is_cubecl_available, norm, ra_glu, ra_softmax, ra_swiglu, ra_tanh, softmax

        assert isinstance(is_cubecl_available(), bool)
        assert not _compat._can_use_cubecl_nn(_compat._CUBECL_NN_MIN_ELEMENTS - 1, "ra_norm_cubecl_f32_pyo3")

        calls: list[str] = []
        fake = SimpleNamespace(
            is_cubecl_available_pyo3=lambda: True,
            ra_norm_cubecl_f32_pyo3=lambda x, out, w, b, eps, cols: calls.append(f"norm:{cols}"),
            ra_tanh_cubecl_f32_pyo3=lambda x, out, b: calls.append("tanh"),
            ra_swiglu_cubecl_f32_pyo3=lambda g, u, out, b: calls.append("swiglu"),
            ra_glu_cubecl_f32_pyo3=lambda g, u, out, b: calls.append("glu"),
            ra_softmax_cubecl_f32_pyo3=lambda s, out, cols, sc: calls.append(f"softmax:{cols}"),
            ra_norm_f32_pyo3=lambda *a: calls.append("simd_norm"),
            ra_tanh_f32_pyo3=lambda *a: calls.append("simd_tanh"),
            ra_swiglu_f32_pyo3=lambda *a: calls.append("simd_swiglu"),
            ra_glu_f32_pyo3=lambda *a: calls.append("simd_glu"),
            ra_softmax_f32_pyo3=lambda *a: calls.append("simd_softmax"),
        )
        for mod in (_compat, norm, activation, softmax):
            monkeypatch.setattr(mod, "_rs_engine", fake, raising=False)
            monkeypatch.setattr(mod, "_HAS_RUST_SIMD", True, raising=False)
        monkeypatch.setattr(_compat, "_HAS_CUBECL", True)

        big = np.ones((256, 256), dtype=np.float32)
        small = np.ones((4, 16), dtype=np.float32)
        RaNorm(256)(big); ra_tanh(big); ra_swiglu(big, big); ra_glu(big, big); ra_softmax(big); ra_tanh(small)
        assert calls == ["norm:256", "tanh", "swiglu", "glu", "softmax:256", "simd_tanh"]

    def test_moe_grouped_gemm_fallback_skewed_gradcheck_and_ptx(self) -> None:
        from luminamoon.nn.kernels import HAS_TRITON, can_use_triton_grouped_gemm, triton_grouped_gemm, triton_ra_gated_packed
        x_cpu, w_cpu, offs = torch.randn(4, 8), torch.randn(2, 8, 16), torch.tensor([4, 4], dtype=torch.int32)
        assert not can_use_triton_grouped_gemm(x_cpu, w_cpu) and triton_grouped_gemm(x_cpu, w_cpu, offs) is None
        assert triton_ra_gated_packed(torch.randn(4, 16)) is None
        moe = RaMoE(4, 6, 4, num_experts=3, top_k=1, dtype=torch.float64)
        with torch.no_grad():
            moe.router.weight.zero_(); moe.router.weight[0].fill_(10.0)
        x64 = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda t: moe(t), (x64,), eps=1e-6, atol=1e-4)
        if HAS_TRITON:
            from triton.backends.compiler import GPUTarget
            from triton.compiler import ASTSource, compile as tc
            from luminamoon.nn.kernels.moe_grouped_gemm import _moe_k_grouped_dw_kernel, _moe_m_grouped_gemm_kernel, _ra_gated_packed_bwd_kernel, _ra_gated_packed_fwd_kernel
            tgt = GPUTarget("cuda", 89, 32)
            mk = lambda ks: {k: ("*i32" if k == "offs_ptr" else "*i64" if k == "gather_idx_ptr" else "*bf16" if k.endswith("_ptr") else "i64" if k.startswith("stride_") else "i32") for k in ks}
            c1 = tc(ASTSource(fn=_moe_m_grouped_gemm_kernel.fn, signature=mk(("a_ptr", "w_ptr", "c_ptr", "offs_ptr", "gather_idx_ptr", "K", "N", "E", "stride_am", "stride_ak", "stride_we", "stride_wk", "stride_wn", "stride_cm", "stride_cn")), constexprs={"HAS_GATHER": True, "EVEN_K": True, "DOT_PRECISION": "ieee", "BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8, "CTAS_PER_SM": 2}), target=tgt)
            c2 = tc(ASTSource(fn=_moe_k_grouped_dw_kernel.fn, signature=mk(("x_ptr", "dy_ptr", "dw_ptr", "offs_ptr", "gather_idx_ptr", "K", "N", "E", "stride_xm", "stride_xk", "stride_dym", "stride_dyn", "stride_dwe", "stride_dwk", "stride_dwn")), constexprs={"HAS_GATHER": True, "DOT_PRECISION": "ieee", "BLOCK_K": 64, "BLOCK_N": 64, "BLOCK_M": 64, "GROUP_SIZE_K": 8}), target=tgt)
            c3 = tc(ASTSource(fn=_ra_gated_packed_fwd_kernel, signature=mk(("gu_ptr", "out_ptr", "M", "H", "stride_gum", "stride_om")), constexprs={"b_val": 1.0, "IS_SWIGLU": True, "BLOCK_H": 1024}), target=tgt)
            c4 = tc(ASTSource(fn=_ra_gated_packed_bwd_kernel, signature=mk(("grad_out_ptr", "gu_ptr", "dgu_ptr", "M", "H", "stride_gm", "stride_gum", "stride_dgum")), constexprs={"b_val": 1.0, "IS_SWIGLU": True, "BLOCK_H": 1024}), target=tgt)
            assert all(len(c.asm["ptx"]) > 400 for c in (c1, c2, c3, c4))
