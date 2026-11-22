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
    ra_glu,
    ra_swiglu,
    ra_tanh,
)
from luminamoon.nn._custom_ops import (
    _HAS_CUSTOM_OP,
    compiled_linear_ce,
    compiled_ra_attention,
    compiled_ra_gated,
    compiled_ra_norm,
    compiled_ra_tanh,
)


@pytest.mark.skipif(not _HAS_CUSTOM_OP, reason="torch.library.custom_op requires PyTorch >= 2.4")
class TestCustomOpsAndOptionalCompileV2:
    """Verifies optional torch.compile / CUDA Graphs custom_op integration without altering default eager mode."""

    def test_custom_ops_opcheck_cpu(self) -> None:
        torch.manual_seed(101)
        x = torch.randn(2, 8, 16, requires_grad=True)
        w = torch.ones(16, requires_grad=True)
        b = torch.zeros(16, requires_grad=True)
        torch.library.opcheck(torch.ops.luminamoon.ra_norm_fwd, (x, w, b, 1e-6))
        torch.library.opcheck(torch.ops.luminamoon.ra_norm_fwd, (x, w, None, 1e-6))

        g = torch.randn(2, 8, 16, requires_grad=True)
        u = torch.randn(2, 8, 16, requires_grad=True)
        torch.library.opcheck(torch.ops.luminamoon.ra_gated_fwd, (g, u, 1.0, True))
        torch.library.opcheck(torch.ops.luminamoon.ra_gated_fwd, (g, u, 1.0, False))
        torch.library.opcheck(torch.ops.luminamoon.ra_tanh_fwd, (x, 1.0))

        q = torch.randn(2, 4, 16, 8, requires_grad=True)
        k = torch.randn(2, 2, 16, 8, requires_grad=True)
        v = torch.randn(2, 2, 16, 8, requires_grad=True)
        sm = 1.0 / math.sqrt(8.0)
        torch.library.opcheck(torch.ops.luminamoon.ra_attention_fwd, (q, k, v, True, sm, 0.0625, True))
        torch.library.opcheck(torch.ops.luminamoon.ra_attention_fwd, (q, k, v, False, sm, 0.0625, False))

        lx = torch.randn(2, 8, 16, requires_grad=True)
        lw = torch.randn(32, 16, requires_grad=True)
        lb = torch.randn(32, requires_grad=True)
        lt = torch.randint(0, 32, (2, 8))
        torch.library.opcheck(torch.ops.luminamoon.linear_ce_fwd, (lx, lw, lt, lb, -100, "mean", 0.0, 1.0, 0))
        torch.library.opcheck(torch.ops.luminamoon.linear_ce_fwd, (lx, lw, lt, None, -100, "none", 0.1, 1.0, 8))

    def test_custom_ops_non_contiguous_and_mixed_dtype_opcheck_cpu(self) -> None:
        torch.manual_seed(151)
        # Non-contiguous QK-Norm layout (B, H, T, D) from transpose(1, 2) + mixed BF16 x / FP32 weight.
        x_nc = torch.randn(2, 16, 4, 8, dtype=torch.bfloat16).transpose(1, 2).requires_grad_(True)
        w_f32 = torch.ones(8, dtype=torch.float32, requires_grad=True)
        b_f32 = torch.zeros(8, dtype=torch.float32, requires_grad=True)
        assert not x_nc.is_contiguous()
        torch.library.opcheck(torch.ops.luminamoon.ra_norm_fwd, (x_nc, w_f32, b_f32, 1e-6))

        # Non-contiguous split gate/up layout for RaSwiGLU / RaGLU.
        gu = torch.randn(2, 8, 32, requires_grad=True)
        g_nc, u_nc = gu.chunk(2, dim=-1)
        assert not g_nc.is_contiguous()
        torch.library.opcheck(torch.ops.luminamoon.ra_gated_fwd, (g_nc, u_nc, 1.0, True))
        torch.library.opcheck(torch.ops.luminamoon.ra_gated_fwd, (g_nc, u_nc, 1.0, False))
        torch.library.opcheck(torch.ops.luminamoon.ra_tanh_fwd, (g_nc, 1.0))

        # Non-contiguous transposed Q/K/V layout (B, H, T, D) from (B, T, H, D).transpose(1, 2).
        q_nc = torch.randn(2, 16, 4, 8).transpose(1, 2).requires_grad_(True)
        k_nc = torch.randn(2, 16, 2, 8).transpose(1, 2).requires_grad_(True)
        v_nc = torch.randn(2, 16, 2, 8).transpose(1, 2).requires_grad_(True)
        assert not q_nc.is_contiguous()
        sm = 1.0 / math.sqrt(8.0)
        torch.library.opcheck(torch.ops.luminamoon.ra_attention_fwd, (q_nc, k_nc, v_nc, True, sm, 0.0625, True))

    def test_custom_ops_fullgraph_compile_parity_cpu(self) -> None:
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile unavailable.")

        torch.manual_seed(202)
        # 1. RaNorm
        x1 = torch.randn(2, 6, 16, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        w1 = torch.randn(16, requires_grad=True)
        w2 = w1.detach().clone().requires_grad_(True)
        b1 = torch.randn(16, requires_grad=True)
        b2 = b1.detach().clone().requires_grad_(True)

        norm = RaNorm(16, bias=True)
        with torch.no_grad():
            norm.weight.copy_(w1)
            norm.bias.copy_(b1)
        y_ref = norm(x1)
        c_norm = torch.compile(lambda a, b_, c_: compiled_ra_norm(a, b_, c_, 1e-6), backend="aot_eager", fullgraph=True)
        y_cmp = c_norm(x2, w2, b2)
        assert torch.allclose(y_ref, y_cmp, atol=1e-5)
        y_ref.sum().backward()
        y_cmp.sum().backward()
        assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
        assert torch.allclose(norm.weight.grad, w2.grad, atol=1e-5)
        assert torch.allclose(norm.bias.grad, b2.grad, atol=1e-5)

        # 2. RaSwiGLU / RaGLU / RaTanh
        for is_swi in (True, False):
            g1 = torch.randn(2, 6, 16, requires_grad=True)
            u1 = torch.randn(2, 6, 16, requires_grad=True)
            g2 = g1.detach().clone().requires_grad_(True)
            u2 = u1.detach().clone().requires_grad_(True)
            o_ref = ra_swiglu(g1, u1, b=1.2) if is_swi else ra_glu(g1, u1, b=1.2)
            c_act = torch.compile(
                lambda g_, u_: compiled_ra_gated(g_, u_, 1.2, is_swi), backend="aot_eager", fullgraph=True
            )
            o_cmp = c_act(g2, u2)
            assert torch.allclose(o_ref, o_cmp, atol=1e-5)
            o_ref.sum().backward()
            o_cmp.sum().backward()
            assert torch.allclose(g1.grad, g2.grad, atol=1e-5)
            assert torch.allclose(u1.grad, u2.grad, atol=1e-5)

        t1 = torch.randn(2, 6, 16, requires_grad=True)
        t2 = t1.detach().clone().requires_grad_(True)
        c_tanh = torch.compile(lambda a: compiled_ra_tanh(a, 1.5), backend="aot_eager", fullgraph=True)
        ot1, ot2 = ra_tanh(t1, b=1.5), c_tanh(t2)
        assert torch.allclose(ot1, ot2, atol=1e-5)
        ot1.sum().backward()
        ot2.sum().backward()
        assert torch.allclose(t1.grad, t2.grad, atol=1e-5)

        # 3. Flash-RaAttention GQA causal
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

        # 4. LinearCrossEntropy
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

    def test_default_eager_block_remains_default(self) -> None:
        torch.manual_seed(303)
        block = torch.nn.Sequential(
            RaNorm(16),
            RaAttention(16, num_heads=4, num_kv_heads=2, is_causal=True),
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

    @pytest.mark.parametrize("compile_mode", ["default", "reduce-overhead"])
    def test_cuda_optional_torch_compile_fullgraph_parity(self, compile_mode: str) -> None:
        if not (torch.cuda.is_available() and hasattr(torch, "compile")):
            pytest.skip("CUDA + torch.compile required for live GPU compile/CUDA-graph parity test.")

        from luminamoon.nn.kernels import HAS_TRITON

        if not HAS_TRITON:
            pytest.skip("Triton unavailable.")

        torch.manual_seed(404)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        class _MiniBlock(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n1 = RaNorm(64, device="cuda", dtype=dtype)
                self.attn = RaAttention(
                    64, num_heads=4, num_kv_heads=2, is_causal=True, qk_norm=True, device="cuda", dtype=dtype
                )
                self.n2 = RaNorm(64, device="cuda", dtype=dtype)
                self.ffn = RaSwiGLU(64, 128, 64, device="cuda", dtype=dtype)
                self.moe = RaMoE(64, 64, 64, num_experts=4, top_k=2, num_shared_experts=1, device="cuda", dtype=dtype)
                self.head = LinearCrossEntropy(64, 256, device="cuda", dtype=dtype)

            def forward(self, x_in: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
                h = x_in + self.attn(self.n1(x_in))
                h = h + self.ffn(self.n2(h)) + self.moe(h)
                return self.head(h, tgt)

        eager_mod = _MiniBlock()
        comp_mod = _MiniBlock()
        comp_mod.load_state_dict(eager_mod.state_dict())
        compiled_fn = torch.compile(comp_mod, mode=compile_mode, fullgraph=True)

        x_base = torch.randn(2, 64, 64, device="cuda", dtype=dtype)
        tgt = torch.randint(0, 256, (2, 64), device="cuda")

        x1 = x_base.clone().requires_grad_(True)
        l1 = eager_mod(x1, tgt)
        l1.backward()

        # Run twice for reduce-overhead so CUDA graph capture & replay both execute.
        for _ in range(2):
            comp_mod.zero_grad(set_to_none=True)
            x2 = x_base.clone().requires_grad_(True)
            l2 = compiled_fn(x2, tgt)
            l2.backward()

        assert torch.allclose(l1.float(), l2.float(), atol=5e-2, rtol=5e-2)
        assert x1.grad is not None and x2.grad is not None
        assert torch.isfinite(x2.grad).all()
        assert torch.allclose(x1.grad.float(), x2.grad.float(), atol=1e-1, rtol=1e-1)
