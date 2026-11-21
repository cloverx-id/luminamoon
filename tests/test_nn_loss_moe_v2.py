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
    PosRaGLU,
    RaAttention,
    RaBatchNorm2d,
    RaEMA,
    RaGLU,
    RaMoE,
    RaNorm,
    TriGLU,
    linear_cross_entropy,
    permute_tokens_by_expert,
    pos_ra_glu,
    ra_attention,
    ra_softmax,
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
    @pytest.mark.parametrize("expert_type", ["pos_ra_glu", "tri_glu"])
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
        for e_idx, exp in enumerate(moe.experts):
            mask = t_idx == e_idx
            w_e = torch.where(mask, t_weights, torch.zeros_like(t_weights)).sum(dim=-1, keepdim=True)
            ref_2d = ref_2d + exp(x2_2d) * w_e
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

    def test_numpy_module_parity_glu_ema_loss_attention(self) -> None:
        torch.manual_seed(2026)
        x_t = torch.randn(2, 5, 8, dtype=torch.float32)
        x_np = x_t.numpy()

        for mod in (
            PosRaGLU(8, 16, out_features=4, bias=True, b=2.0),
            RaGLU(8, 16, out_features=4, bias=True, b=2.0),
            TriGLU(8, 16, out_features=4, bias=True, b=2.5),
        ):
            with torch.no_grad():
                y_pt = mod(x_t).numpy()
            y_np = mod(x_np)
            assert y_np.shape == (2, 5, 4)
            np.testing.assert_allclose(y_np, y_pt, atol=1e-5, rtol=1e-5)

        assert "b=2.0" in RaGLU(8, 16, b=2.0).extra_repr()
        assert "b=2.5" in TriGLU(8, 16, b=2.5).extra_repr()
        moe_tri = RaMoE(8, 16, num_experts=2, top_k=1, expert_type="tri_glu", b=2.5)
        assert all(exp.b == 2.5 for exp in moe_tri.experts)

        ema_dep = RaEMA(dim=8, input_dependent_decay=True, use_residual_gate=True)
        ema_stat = RaEMA(dim=8, input_dependent_decay=False, use_residual_gate=True)
        with torch.no_grad():
            ema_dep.decay_proj.weight.normal_(0.0, 0.25)
            ema_dep.residual_logit.fill_(-0.4)
            ema_stat.decay_logit.fill_(0.7)
            ema_stat.residual_logit.fill_(-0.4)
            for ema_mod in (ema_dep, ema_stat):
                y_pt, s_pt = ema_mod(x_t, return_state=True)
                y_np, s_np = ema_mod(x_np, return_state=True)
                np.testing.assert_allclose(y_np, y_pt.numpy(), atol=1e-5, rtol=1e-5)
                np.testing.assert_allclose(s_np, s_pt.numpy(), atol=1e-5, rtol=1e-5)

        lce = LinearCrossEntropy(8, 16, bias=True, logit_scale=0.8)
        tgt_t = torch.randint(0, 16, (2, 5), dtype=torch.long)
        with torch.no_grad():
            logits_pt = lce(x_t).numpy()
            loss_pt = float(lce(x_t, tgt_t).item())
        np.testing.assert_allclose(lce(x_np), logits_pt, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(float(lce(x_np, tgt_t.numpy())), loss_pt, atol=1e-5, rtol=1e-5)

        attn = RaAttention(dim=16, num_heads=4, num_kv_heads=2, qk_norm=True, use_ema=True, is_causal=True)
        attn.eval()
        xa_t = torch.randn(2, 6, 16, dtype=torch.float32)
        with torch.no_grad():
            oa_pt = attn(xa_t).numpy()
        oa_np = attn(xa_t.numpy())
        assert oa_np.shape == (2, 6, 16)
        np.testing.assert_allclose(oa_np, oa_pt, atol=1e-4, rtol=1e-4)

    def test_cpu_rust_simd_nd_ranorm_and_pos_ra_glu(self) -> None:
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

        # Ensure autograd is preserved when x.requires_grad is False but weight.requires_grad is True
        x_leaf = torch.randn(2, 4, 32, dtype=torch.float32, requires_grad=False)
        norm.zero_grad()
        norm(x_leaf).sum().backward()
        assert norm.weight.grad is not None and torch.isfinite(norm.weight.grad).all()

        g = torch.randn(4, 32, dtype=torch.float32)
        u = torch.randn(4, 32, dtype=torch.float32)
        for b in (1.0, 2.5):
            ref = u * (0.5 * (1.0 + ra_tanh(g, b=b) / b))
            with torch.no_grad():
                assert torch.allclose(pos_ra_glu(g, u, b=b), ref, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(pos_ra_glu(g.numpy(), u.numpy(), b=b), ref.numpy(), atol=1e-5, rtol=1e-5)

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



