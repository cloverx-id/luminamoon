from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from luminamoon.nn import (
    LinearCrossEntropy,
    RaAttention,
    RaGLU,
    RaMoE,
    RaNorm,
    RaSwiGLU,
    linear_cross_entropy,
    ra_attention,
)
from luminamoon.nn._custom_ops import (
    _HAS_CUSTOM_OP,
    compiled_linear_ce,
    compiled_ra_attention,
)


@pytest.mark.skipif(not _HAS_CUSTOM_OP, reason="torch.library.custom_op requires PyTorch >= 2.4")
class TestCustomOpsAndOptionalCompileV2:
    """Verifies optional torch.compile / CUDA Graphs custom_op integration and Inductor fusion."""

    def test_custom_ops_opcheck_contiguous_and_non_contiguous_cpu(self) -> None:
        torch.manual_seed(101)
        sm = 1.0 / math.sqrt(8.0)

        # Contiguous & transposed non-contiguous Q/K/V for Flash-RaAttention GQA.
        q = torch.randn(2, 4, 16, 8, requires_grad=True)
        k = torch.randn(2, 2, 16, 8, requires_grad=True)
        v = torch.randn(2, 2, 16, 8, requires_grad=True)
        torch.library.opcheck(torch.ops.luminamoon.ra_attention_fwd, (q, k, v, True, sm, 0.0625))
        torch.library.opcheck(torch.ops.luminamoon.ra_attention_fwd, (q, k, v, False, sm, 0.0625))

        q_nc = torch.randn(2, 16, 4, 8).transpose(1, 2).requires_grad_(True)
        k_nc = torch.randn(2, 16, 2, 8).transpose(1, 2).requires_grad_(True)
        v_nc = torch.randn(2, 16, 2, 8).transpose(1, 2).requires_grad_(True)
        assert not q_nc.is_contiguous()
        torch.library.opcheck(torch.ops.luminamoon.ra_attention_fwd, (q_nc, k_nc, v_nc, True, sm, 0.0625))

        # Adaptive Chunked LinearCrossEntropy (mean, none + label_smoothing, mixed BF16 x / FP32 weight).
        lx = torch.randn(2, 8, 16, requires_grad=True)
        lw = torch.randn(32, 16, requires_grad=True)
        lb = torch.randn(32, requires_grad=True)
        lt = torch.randint(0, 32, (2, 8))
        torch.library.opcheck(torch.ops.luminamoon.linear_ce_fwd, (lx, lw, lt, lb, -100, "mean", 0.0, 1.0, 0, 0))
        torch.library.opcheck(torch.ops.luminamoon.linear_ce_fwd, (lx, lw, lt, None, -100, "none", 0.1, 1.0, 8, 0))
        lx_bf16 = torch.randn(2, 8, 16, dtype=torch.bfloat16, requires_grad=True)
        torch.library.opcheck(torch.ops.luminamoon.linear_ce_fwd, (lx_bf16, lw, lt, None, -100, "mean", 0.0, 1.0, 4, 0))

    def test_custom_ops_fullgraph_compile_parity_cpu(self) -> None:
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile unavailable.")

        torch.manual_seed(202)
        # 1. Flash-RaAttention GQA causal
        q1 = torch.randn(2, 4, 16, 8, requires_grad=True)
        k1 = torch.randn(2, 2, 16, 8, requires_grad=True)
        v1 = torch.randn(2, 2, 16, 8, requires_grad=True)
        q2, k2, v2 = (t.detach().clone().requires_grad_(True) for t in (q1, k1, v1))
        sm = 1.0 / math.sqrt(8.0)
        oa1 = ra_attention(q1, k1, v1, is_causal=True, scale=sm, ra_scale=0.0625)
        c_attn = torch.compile(
            lambda q_, k_, v_: compiled_ra_attention(q_, k_, v_, True, sm, 0.0625),
            backend="aot_eager",
            fullgraph=True,
        )
        oa2 = c_attn(q2, k2, v2)
        assert torch.allclose(oa1, oa2, atol=1e-5)
        oa1.sum().backward()
        oa2.sum().backward()
        assert torch.allclose(q1.grad, q2.grad, atol=1e-5)
        assert torch.allclose(k1.grad, k2.grad, atol=1e-5)
        assert torch.allclose(v1.grad, v2.grad, atol=1e-5)

        # 2. LinearCrossEntropy
        lx1 = torch.randn(2, 6, 16, requires_grad=True)
        lw1 = torch.randn(32, 16, requires_grad=True)
        lb1 = torch.randn(32, requires_grad=True)
        lx2, lw2, lb2 = (t.detach().clone().requires_grad_(True) for t in (lx1, lw1, lb1))
        lt = torch.randint(0, 32, (2, 6))
        l1 = linear_cross_entropy(lx1, lw1, lt, lb1, label_smoothing=0.05, chunk_size=4)
        c_lce = torch.compile(
            lambda x_, w_, t_, b_: compiled_linear_ce(x_, w_, t_, b_, -100, "mean", 0.05, 1.0, 4),
            backend="aot_eager",
            fullgraph=True,
        )
        l2 = c_lce(lx2, lw2, lt, lb2)
        assert torch.allclose(l1, l2, atol=1e-5)
        l1.backward()
        l2.backward()
        assert torch.allclose(lx1.grad, lx2.grad, atol=1e-5)
        assert torch.allclose(lw1.grad, lw2.grad, atol=1e-5)
        assert torch.allclose(lb1.grad, lb2.grad, atol=1e-5)

        # 3. Pointwise/Norm layers (RaNorm, RaSwiGLU, RaGLU) remain Inductor-fusable fullgraph
        block = torch.nn.Sequential(RaNorm(16), RaSwiGLU(16, 32, 16), RaGLU(16, 32, 16))
        c_block = torch.compile(block, backend="aot_eager", fullgraph=True)
        x1 = torch.randn(2, 8, 16, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        y1, y2 = block(x1), c_block(x2)
        assert torch.allclose(y1, y2, atol=1e-5)
        y1.sum().backward()
        y2.sum().backward()
        assert torch.allclose(x1.grad, x2.grad, atol=1e-5)

    def test_default_eager_block_remains_default(self) -> None:
        torch.manual_seed(303)
        block = torch.nn.Sequential(
            RaNorm(16),
            RaAttention(16, num_heads=4, num_kv_heads=2, is_causal=True, qk_norm=True),
            RaSwiGLU(16, 32, 16),
            RaGLU(16, 32, 16),
            RaMoE(16, 24, 16, num_experts=4, top_k=2, num_shared_experts=1),
        )
        head = LinearCrossEntropy(16, 64)
        x = torch.randn(2, 12, 16, requires_grad=True)
        t = torch.randint(0, 64, (2, 12))
        loss = head(block(x), t)
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
    def test_linear_cross_entropy_save_logits_option_parity(
        self, reduction: str, label_smoothing: float
    ) -> None:
        torch.manual_seed(505)
        B, T, D, V = 2, 6, 16, 48
        x1 = torch.randn(B, T, D, requires_grad=True)
        w1 = torch.randn(V, D, requires_grad=True)
        b1 = torch.randn(V, requires_grad=True)
        x2, w2, b2 = (t.detach().clone().requires_grad_(True) for t in (x1, w1, b1))
        tgt = torch.randint(0, V, (B, T))
        tgt[0, 0] = -100
        tgt[1, -1] = V + 7  # out-of-range target must be silently ignored in both modes

        l_chunk = linear_cross_entropy(
            x1, w1, tgt, b1, reduction=reduction, label_smoothing=label_smoothing, logit_scale=0.5, save_logits=False
        )
        l_saved = linear_cross_entropy(
            x2, w2, tgt, b2, reduction=reduction, label_smoothing=label_smoothing, logit_scale=0.5, save_logits=True
        )
        assert torch.allclose(l_chunk, l_saved, atol=1e-5, rtol=1e-5)

        g_out = torch.randn_like(l_chunk) if reduction == "none" else torch.tensor(1.0)
        l_chunk.backward(g_out)
        l_saved.backward(g_out)
        assert torch.allclose(x1.grad, x2.grad, atol=1e-4, rtol=1e-4)
        assert torch.allclose(w1.grad, w2.grad, atol=1e-4, rtol=1e-4)
        assert torch.allclose(b1.grad, b2.grad, atol=1e-4, rtol=1e-4)

        # Stateful module extra_repr and all-ignored target safety.
        mod = LinearCrossEntropy(D, V, bias=True, reduction="mean", save_logits=True)
        assert "save_logits=True" in mod.extra_repr()
        all_ign = torch.full((B, T), -100, dtype=torch.long)
        x_ign = torch.randn(B, T, D, requires_grad=True)
        l_ign = mod(x_ign, all_ign)
        assert l_ign.item() == 0.0 and torch.isfinite(l_ign)
        l_ign.backward()
        assert x_ign.grad is not None and torch.all(x_ign.grad == 0.0)

    @pytest.mark.parametrize("compile_mode", ["default", "reduce-overhead"])
    @pytest.mark.parametrize("use_amp_fp32_weights", [False, True])
    def test_cuda_optional_torch_compile_fullgraph_parity(
        self, compile_mode: str, use_amp_fp32_weights: bool
    ) -> None:
        if not (torch.cuda.is_available() and hasattr(torch, "compile")):
            pytest.skip("CUDA + torch.compile required for live GPU compile/CUDA-graph parity test.")

        from luminamoon.nn.kernels import HAS_TRITON

        if not HAS_TRITON:
            pytest.skip("Triton unavailable.")

        torch.manual_seed(404)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        param_dtype = torch.float32 if use_amp_fp32_weights else amp_dtype

        class _MiniBlock(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n1 = RaNorm(64, device="cuda", dtype=param_dtype)
                self.attn = RaAttention(
                    64, num_heads=4, num_kv_heads=2, is_causal=True, qk_norm=True, device="cuda", dtype=param_dtype
                )
                self.n2 = RaNorm(64, device="cuda", dtype=param_dtype)
                self.ffn = RaSwiGLU(64, 128, 64, device="cuda", dtype=param_dtype)
                self.moe = RaMoE(
                    64, 64, 64, num_experts=4, top_k=2, num_shared_experts=1, device="cuda", dtype=param_dtype
                )
                self.head = LinearCrossEntropy(64, 256, device="cuda", dtype=param_dtype)

            def forward(self, x_in: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
                h = x_in + self.attn(self.n1(x_in))
                h = h + self.ffn(self.n2(h)) + self.moe(h)
                return self.head(h, tgt)

        eager_mod = _MiniBlock()
        comp_mod = _MiniBlock()
        comp_mod.load_state_dict(eager_mod.state_dict())
        compiled_fn = torch.compile(comp_mod, mode=compile_mode, fullgraph=True)

        x_base = torch.randn(2, 64, 64, device="cuda", dtype=amp_dtype)
        tgt = torch.randint(0, 256, (2, 64), device="cuda")

        x1 = x_base.clone().requires_grad_(True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp_fp32_weights):
            l1 = eager_mod(x1, tgt)
        l1.backward()

        for _ in range(2):
            comp_mod.zero_grad(set_to_none=True)
            x2 = x_base.clone().requires_grad_(True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp_fp32_weights):
                l2 = compiled_fn(x2, tgt)
            l2.backward()

        assert torch.allclose(l1.float(), l2.float(), atol=5e-2, rtol=5e-2)
        assert x1.grad is not None and x2.grad is not None
        assert torch.isfinite(x2.grad).all()
        assert torch.allclose(x1.grad.float(), x2.grad.float(), atol=1e-1, rtol=1e-1)


def test_nn_numerical_stability_validation_and_edge_cases() -> None:
    import math
    import numpy as np
    from luminamoon.nn import (
        LinearCrossEntropy,
        RaBatchNorm2d,
        RaMoE,
        RaNorm,
        linear_cross_entropy,
        ra_attention,
        ra_gelu,
        ra_glu,
        ra_softmax,
        ra_tanh,
    )

    # [H5] ra_tanh / ra_gelu / ra_glu numerical stability for extreme FP16, FP32, NumPy, and Python float inputs
    x_fp16 = torch.tensor([-1e4, -500.0, 0.0, 500.0, 1e4], dtype=torch.float16)
    y_fp16 = ra_tanh(x_fp16, b=1.5)
    assert y_fp16.dtype == torch.float16 and torch.isfinite(y_fp16).all()
    assert torch.allclose(y_fp16.float(), torch.tensor([-1.5, -1.5, 0.0, 1.5, 1.5]), atol=2e-2)

    x_fp32 = torch.tensor([-1e25, -1e20, 0.0, 1e20, 1e25], dtype=torch.float32, requires_grad=True)
    y_fp32 = ra_tanh(x_fp32, b=1.0)
    assert torch.isfinite(y_fp32).all()
    assert torch.allclose(y_fp32, torch.tensor([-1.0, -1.0, 0.0, 1.0, 1.0]), atol=1e-6)
    y_fp32.sum().backward()
    assert x_fp32.grad is not None and torch.isfinite(x_fp32.grad).all()

    assert torch.isfinite(ra_gelu(torch.tensor([-500.0, 500.0], dtype=torch.float16))).all()
    assert torch.isfinite(
        ra_glu(torch.tensor([500.0], dtype=torch.float16), torch.tensor([2.0], dtype=torch.float16))
    ).all()
    y_np = ra_tanh(np.array([-1e200, 1e200], dtype=np.float64), b=2.0)
    assert np.isfinite(y_np).all() and np.allclose(y_np, [-2.0, 2.0])
    assert math.isfinite(ra_tanh(1e300, b=1.0)) and abs(ra_tanh(1e300, b=1.0) - 1.0) < 1e-12

    # [H4] & [L18] RaNorm eps validation & RaBatchNorm2d FP16/BF16 running stats
    for bad_eps in (0.0, -1e-5, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="eps"):
            RaNorm(8, eps=bad_eps)
    for dt in (torch.float16, torch.bfloat16):
        bn = RaBatchNorm2d(4)
        bn.train()
        inp = torch.randn(2, 4, 3, 3, dtype=dt)
        out = bn(inp)
        assert out.dtype == dt and torch.isfinite(out).all()
        assert torch.isfinite(bn.running_mean).all() and torch.isfinite(bn.running_mad).all()

    # [M11] & [M12] LinearCrossEntropy save_logits empty autograd, mixed dtype, and FP64 precision
    x_empty = torch.empty(0, 8, dtype=torch.float32, requires_grad=True)
    w_ce = torch.randn(16, 8, dtype=torch.float32, requires_grad=True)
    b_ce = torch.randn(16, dtype=torch.float32, requires_grad=True)
    t_empty = torch.empty(0, dtype=torch.long)
    for red in ("mean", "sum", "none"):
        loss_e = linear_cross_entropy(x_empty, w_ce, t_empty, bias=b_ce, reduction=red, save_logits=True)
        loss_e.sum().backward()
    assert x_empty.grad is not None and w_ce.grad is not None and b_ce.grad is not None

    x_mix = torch.randn(4, 8, dtype=torch.float32, requires_grad=True)
    w_bf16 = torch.randn(16, 8, dtype=torch.bfloat16, requires_grad=True)
    t_mix = torch.randint(0, 16, (4,))
    loss_mix = linear_cross_entropy(x_mix, w_bf16, t_mix, save_logits=True)
    loss_mix.backward()
    assert torch.isfinite(loss_mix) and x_mix.grad is not None and w_bf16.grad is not None

    x64 = torch.randn(6, 8, dtype=torch.float64, requires_grad=True)
    w64 = torch.randn(12, 8, dtype=torch.float64, requires_grad=True)
    b64 = torch.randn(12, dtype=torch.float64, requires_grad=True)
    t64 = torch.randint(0, 12, (6,))
    l_chunk64 = linear_cross_entropy(x64, w64, t64, bias=b64, chunk_size=2)
    l_ref64 = torch.nn.functional.cross_entropy(torch.nn.functional.linear(x64, w64, b64), t64)
    assert torch.allclose(l_chunk64, l_ref64, atol=1e-12, rtol=1e-12)
    gx_c, gw_c, gb_c = torch.autograd.grad(l_chunk64, (x64, w64, b64))
    gx_r, gw_r, gb_r = torch.autograd.grad(l_ref64, (x64, w64, b64))
    assert torch.allclose(gx_c, gx_r, atol=1e-11, rtol=1e-11)
    assert torch.allclose(gw_c, gw_r, atol=1e-11, rtol=1e-11)
    assert torch.allclose(gb_c, gb_r, atol=1e-11, rtol=1e-11)

    # [M11], [M13], [M14], [L19] RaMoE dim validation, M=0 autograd, mixed dtype, detached last_aux_loss
    with pytest.raises(ValueError, match="hidden_features"):
        RaMoE(8, hidden_features=0)
    with pytest.raises(ValueError, match="out_features"):
        RaMoE(8, out_features=-2)
    moe = RaMoE(8, hidden_features=16, out_features=8, num_experts=4, top_k=2, bias=True)
    x_m0 = torch.empty(0, 8, dtype=torch.float32, requires_grad=True)
    y_m0 = moe(x_m0)
    y_m0.sum().backward()
    assert x_m0.grad is not None and moe.w_gate_up.grad is not None

    x_bf = torch.randn(2, 4, 8, dtype=torch.bfloat16, requires_grad=True)
    y_bf = moe(x_bf)
    assert y_bf.dtype == torch.bfloat16 and torch.isfinite(y_bf).all()
    assert moe.last_aux_loss is not None and not moe.last_aux_loss.requires_grad
    y_bf2, aux_ret = moe(x_bf, return_aux_loss=True)
    assert aux_ret.requires_grad and moe.last_aux_loss is aux_ret

    # [M15] & [H9] 1D attn_mask support & empty softmax/tanh guards
    q = torch.randn(2, 2, 4, 8, requires_grad=True)
    k = torch.randn(2, 2, 6, 8, requires_grad=True)
    v = torch.randn(2, 2, 6, 8, requires_grad=True)
    m1d = torch.tensor([0.0, 0.0, -1.0, 0.0, -2.0, 0.0], requires_grad=True)
    out_1d = ra_attention(q, k, v, attn_mask=m1d)
    out_4d = ra_attention(q, k, v, attn_mask=m1d.view(1, 1, 1, -1))
    assert torch.allclose(out_1d, out_4d, atol=1e-6, rtol=1e-6)
    out_1d.sum().backward()
    assert m1d.grad is not None and m1d.grad.shape == m1d.shape and torch.isfinite(m1d.grad).all()
    assert ra_softmax(torch.empty(0, 8)).shape == (0, 8)
    assert ra_tanh(torch.empty(0, 8)).shape == (0, 8)
