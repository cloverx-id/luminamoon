"""Regression tests for numerical, FFI, cache, and portability edge cases in luminamoon.nn."""

from __future__ import annotations

import pytest
import torch

from luminamoon.nn._compat import _HAS_RUST_SIMD, _rs_engine
from luminamoon.nn.kernels._base import _int_spec
from luminamoon.nn.kernels.moe_grouped_gemm import _resolve_dot_precision
from luminamoon.nn.attention import LatentKVAttention, SlotCrossAttention
from luminamoon.nn.memory import CausalMemoryTap, DirectiveGate, OrthogonalHub, TwoTierMemory, assert_causal_no_leak
from luminamoon.nn.moe import RaMoE, permute_tokens_by_expert
from luminamoon.nn.norm import RaNorm
from luminamoon.nn.position import DistanceDecayBias, QRoPE


def test_ra_norm_empty_batch_zeroes_parameter_grads() -> None:
    norm = RaNorm(16, eps=1e-6, elementwise_affine=True, bias=True)
    x = torch.empty((0, 16), dtype=torch.float32, requires_grad=True)
    y = norm(x)
    y.sum().backward()
    assert norm.weight.grad is not None and torch.all(norm.weight.grad == 0.0)
    assert norm.bias.grad is not None and torch.all(norm.bias.grad == 0.0)


@pytest.mark.skipif(not (_HAS_RUST_SIMD and hasattr(_rs_engine, "ra_norm_bwd_f32_pyo3")), reason="Rust SIMD unavailable")
def test_ra_norm_bwd_ffi_rejects_db_aliasing_weight() -> None:
    gy = torch.randn(4, 16, dtype=torch.float32)
    x = torch.randn(4, 16, dtype=torch.float32)
    w = torch.randn(16, dtype=torch.float32)
    dx = torch.empty_like(x)
    with pytest.raises(ValueError, match="Aliased"):
        _rs_engine.ra_norm_bwd_f32_pyo3(gy, x, w, dx, None, w, 1e-6, 16)


def test_distance_decay_bias_lru_cache_and_single_token_skip() -> None:
    bias_mod = DistanceDecayBias(num_heads=4, mode="linear", learnable=False)
    b1 = bias_mod(8, 8, device=torch.device("cpu"), dtype=torch.float32)
    b2 = bias_mod(8, 8, device=torch.device("cpu"), dtype=torch.float32)
    assert b1 is b2
    # Single-token decode skips caching so variable KV lengths do not thrash cache.
    _ = bias_mod(1, 32, device=torch.device("cpu"), dtype=torch.float32)
    assert len(bias_mod._bias_cache) == 1


def test_qrope_tensor_offset_autogrows_and_rejects_negative() -> None:
    qr = QRoPE(head_dim=16, max_seq_len=8)
    x = torch.randn(1, 2, 4, 16)
    pos = torch.tensor([10, 11, 12, 13], dtype=torch.long)
    out = qr(x, offset=pos)
    assert out.shape == x.shape and torch.isfinite(out).all()
    with pytest.raises(ValueError, match=">= 0"):
        qr(x, offset=torch.tensor([-1, 0, 1, 2], dtype=torch.long))


def test_moe_permute_and_forward_on_meta_device() -> None:
    x_meta = torch.empty((4, 16), device="meta", dtype=torch.float32)
    top_idx = torch.empty((4, 2), device="meta", dtype=torch.long)
    top_w = torch.empty((4, 2), device="meta", dtype=torch.float32)
    px, ti, sw, counts = permute_tokens_by_expert(x_meta, top_idx, top_w, num_experts=4)
    assert px.device.type == "meta" and counts.shape == (4,)

    moe = RaMoE(in_features=16, hidden_features=32, num_experts=4, top_k=1, return_aux_loss=True).to("meta")
    y, aux = moe(torch.empty((2, 3, 16), device="meta", dtype=torch.float32))
    assert y.shape == (2, 3, 16) and aux.device.type == "meta"


def test_resolve_dot_precision_handles_new_fp32_precision_api() -> None:
    assert _resolve_dot_precision(torch.bfloat16) == "tf32"
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        prev = torch.backends.cuda.matmul.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            assert _resolve_dot_precision(torch.float32) == "tf32"
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            assert _resolve_dot_precision(torch.float32) == "ieee"
        finally:
            torch.backends.cuda.matmul.fp32_precision = prev


def test_triton_int_spec_distinguishes_equal_to_1_and_div16() -> None:
    assert _int_spec(1) == 0
    assert _int_spec(16) == 1
    assert _int_spec(0) == 1
    assert _int_spec(3) == 2
    assert len({_int_spec(1), _int_spec(16), _int_spec(3)}) == 3


def test_assert_causal_no_leak_passes_causal_and_blocks_leaky_model() -> None:
    attn = LatentKVAttention(dim=32, num_heads=4, kv_latent_dim=16, bias_mode="linear", attention_backend="ra")
    x = torch.randn(2, 8, 32)
    diff = assert_causal_no_leak(lambda inp: attn(inp)[0], x)
    assert diff == 0.0

    dgate = DirectiveGate(dim=32)
    chunks = torch.randn(2, 4, 32)
    # Misusing DirectiveGate with target sequence `x` leaks future tokens and must stop with RuntimeError.
    with pytest.raises(RuntimeError, match="Causal future-token leakage detected"):
        assert_causal_no_leak(lambda inp: inp + dgate(chunks, inp).mean(dim=1, keepdim=True), x)

    with pytest.warns(RuntimeWarning, match="Causal future-token leakage detected"):
        leak = assert_causal_no_leak(
            lambda inp: inp + dgate(chunks, inp).mean(dim=1, keepdim=True), x, raise_on_leak=False
        )
    assert leak > 0.0


def test_two_tier_memory_and_directive_gate_causal_defaults_and_cold_start() -> None:
    mem = TwoTierMemory(dim=32, max_l1=2, max_l2=4, num_heads=4, memory_dropout=0.0).eval()
    dgate = DirectiveGate(dim=32).eval()
    past = torch.randn(2, 6, 32)
    x = torch.randn(2, 8, 32)
    # Default query=None / directive_tokens=None uses past chunks only -> zero future leakage.
    l1, l2 = mem(past)
    gated = dgate(torch.cat((l1, l2), dim=1))
    assert gated.shape == (2, 6, 32)
    assert assert_causal_no_leak(lambda inp: inp + dgate(torch.cat(mem(past), dim=1)).mean(dim=1, keepdim=True), x) == 0.0
    # Cold-start chunks=None returns an empty (B, 0, D) tensor without placeholder zeros.
    e1, e2 = mem(None, query=x)
    assert e1.shape == (2, 0, 32) and e2.shape == (2, 0, 32)
    assert dgate(e1).shape == (2, 0, 32)

    tap = CausalMemoryTap(dim=32, num_heads=4, attention_backend="ra").eval()
    with torch.no_grad():
        tap.fuse_proj.weight.fill_(0.05)
    assert assert_causal_no_leak(lambda inp: tap(inp, past), x) == 0.0
    assert torch.equal(tap(x, None), x)
    with pytest.raises(ValueError, match="alias"):
        tap(x, x)

    hub = OrthogonalHub(hub_size=4, dim=32, use_context_gate=True).eval()
    h_empty = hub(2, context=torch.empty(2, 0, 32))
    assert h_empty.shape == (2, 4, 32) and torch.isfinite(h_empty).all()

    slot_ca = SlotCrossAttention(dim=32, num_heads=4, num_terminals=4, slots_per_terminal=2, attention_backend="ra").eval()
    s_empty = slot_ca(torch.randn(2, 4, 32), torch.empty(2, 0, 32))
    assert s_empty.shape == (2, 8, 32) and torch.all(s_empty == 0.0)
