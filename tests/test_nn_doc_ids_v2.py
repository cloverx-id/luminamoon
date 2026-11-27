from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(ROOT_PYTHON) not in sys.path:
    sys.path.insert(0, str(ROOT_PYTHON))

from luminamoon.data import SequencePacker
from luminamoon.nn import RaAttention, ra_attention


def _reference_doc_mask(doc_ids: torch.Tensor, *, is_causal: bool) -> torch.Tensor:
    same_doc = (doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)) & (doc_ids.unsqueeze(2) > 0)
    if is_causal:
        seq_len = doc_ids.shape[1]
        causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=doc_ids.device))
        same_doc = same_doc & causal.unsqueeze(0)
    return same_doc.unsqueeze(1)


def _reference_ra_attention_with_bool_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep_mask: torch.Tensor,
    *,
    ra_scale: float = 0.0625,
) -> torch.Tensor:
    if k.shape[1] != q.shape[1]:
        g = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)
    sm_scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-2, -1)) * sm_scale
    scores = scores.masked_fill(~keep_mask, float("-inf"))
    max_s = scores.max(dim=-1, keepdim=True)[0]
    has_valid = torch.isfinite(max_s)
    safe_max = torch.where(has_valid, max_s, torch.zeros_like(max_s))
    delta = torch.where(scores == safe_max, torch.zeros_like(scores), safe_max - scores)
    u = delta * ra_scale
    num = torch.clamp(1.0 - u, min=0.0)
    q_r = num / (1.0 + u)
    q2 = q_r * q_r
    q4 = q2 * q2
    phi = torch.where(torch.isfinite(scores), q4 * q4, torch.zeros_like(scores))
    sum_phi = phi.sum(dim=-1, keepdim=True)
    weights = torch.where(sum_phi > 0.0, phi / torch.where(sum_phi > 0.0, sum_phi, torch.ones_like(sum_phi)), torch.zeros_like(phi))
    return weights @ v


class TestRaAttentionDocIds:
    """Parity, padding safety, GQA, compile, and NumPy tests for native `doc_ids` isolation."""

    @pytest.mark.parametrize("is_causal", [True, False])
    @pytest.mark.parametrize("num_kv_heads", [4, 2])
    def test_doc_ids_matches_bool_mask_forward_and_backward(self, is_causal: bool, num_kv_heads: int) -> None:
        torch.manual_seed(2026)
        B, H_q, N, D = 2, 4, 24, 16
        doc_ids = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0],
            ],
            dtype=torch.int32,
        )
        keep_mask = _reference_doc_mask(doc_ids, is_causal=is_causal)

        q_raw = torch.randn(B, H_q, N, D, dtype=torch.float64) * 1.5
        k_raw = torch.randn(B, num_kv_heads, N, D, dtype=torch.float64) * 1.5
        v_raw = torch.randn(B, num_kv_heads, N, D, dtype=torch.float64)
        do = torch.randn(B, H_q, N, D, dtype=torch.float64)

        q1, k1, v1 = (t.clone().requires_grad_(True) for t in (q_raw, k_raw, v_raw))
        q2, k2, v2 = (t.clone().requires_grad_(True) for t in (q_raw, k_raw, v_raw))
        q3, k3, v3 = (t.clone().requires_grad_(True) for t in (q_raw, k_raw, v_raw))

        out_doc = ra_attention(q1, k1, v1, is_causal=is_causal, doc_ids=doc_ids, q_chunk_size=8)
        out_mask = ra_attention(q2, k2, v2, attn_mask=keep_mask, is_causal=False, q_chunk_size=8)
        out_ref = _reference_ra_attention_with_bool_mask(q3, k3, v3, keep_mask)

        assert torch.allclose(out_doc, out_ref, atol=1e-11, rtol=1e-10)
        assert torch.allclose(out_mask, out_ref, atol=1e-11, rtol=1e-10)
        # Padding positions (doc_ids == 0) must produce exact zeros.
        assert torch.allclose(out_doc[0, :, 18:, :], torch.zeros_like(out_doc[0, :, 18:, :]))
        assert torch.allclose(out_doc[1, :, 22:, :], torch.zeros_like(out_doc[1, :, 22:, :]))

        out_doc.backward(do)
        out_mask.backward(do)
        out_ref.backward(do)

        for g1, g2, g3 in (
            (q1.grad, q2.grad, q3.grad),
            (k1.grad, k2.grad, k3.grad),
            (v1.grad, v2.grad, v3.grad),
        ):
            assert g1 is not None and g2 is not None and g3 is not None
            assert torch.isfinite(g1).all()
            assert torch.allclose(g1, g3, atol=1e-10, rtol=1e-9)
            assert torch.allclose(g2, g3, atol=1e-10, rtol=1e-9)
        assert torch.allclose(q1.grad[0, :, 18:, :], torch.zeros_like(q1.grad[0, :, 18:, :]))

    def test_sequence_packer_batch_doc_ids_matches_causal_doc_mask(self) -> None:
        torch.manual_seed(77)
        packer = SequencePacker(batch_size=2, seq_len=16, pad_token_id=0, drop_last=False)
        for doc in ([10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24], [30, 31, 32, 33, 34, 35, 36]):
            packer.add_document(doc, source_name="unit", row_idx=0)
        batches = packer.flush()
        assert len(batches) == 1
        batch = batches[0]

        attn = RaAttention(dim=32, num_heads=4, num_kv_heads=2, is_causal=True, qk_norm=True)
        x = torch.randn(2, 16, 32)
        x1 = x.clone().requires_grad_(True)
        x2 = x.clone().requires_grad_(True)

        out_doc = attn(x1, doc_ids=batch.doc_ids)
        out_mask = attn(x2, attn_mask=batch.causal_doc_mask(as_bool=True))
        assert torch.allclose(out_doc, out_mask, atol=1e-5)

        out_doc.sum().backward()
        out_mask.sum().backward()
        assert x1.grad is not None and x2.grad is not None
        assert torch.allclose(x1.grad, x2.grad, atol=1e-5)

    def test_doc_ids_combined_with_attn_mask_and_numpy_and_validation(self) -> None:
        torch.manual_seed(99)
        B, H, N, D = 2, 2, 10, 8
        doc_ids = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2, 0, 0], [1, 1, 1, 2, 2, 2, 3, 3, 3, 0]], dtype=torch.int32)
        bias = torch.randn(B, 1, N, N, dtype=torch.float64) * 0.25
        q = torch.randn(B, H, N, D, dtype=torch.float64, requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=torch.float64, requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=torch.float64, requires_grad=True)

        out_comb = ra_attention(q, k, v, attn_mask=bias, is_causal=True, doc_ids=doc_ids)
        ref_mask = _reference_doc_mask(doc_ids, is_causal=True)
        eff_bias = bias.masked_fill(~ref_mask, float("-inf"))
        out_ref = ra_attention(q.detach(), k.detach(), v.detach(), attn_mask=eff_bias, is_causal=False)
        assert torch.allclose(out_comb, out_ref, atol=1e-11)

        out_np = ra_attention(
            q.detach().numpy().astype(np.float32),
            k.detach().numpy().astype(np.float32),
            v.detach().numpy().astype(np.float32),
            is_causal=True,
            doc_ids=doc_ids.numpy(),
        )
        out_np_ref = ra_attention(
            q.detach().float(),
            k.detach().float(),
            v.detach().float(),
            is_causal=True,
            doc_ids=doc_ids,
        ).numpy()
        np.testing.assert_allclose(out_np, out_np_ref, atol=1e-5, rtol=1e-5)

        with pytest.raises(ValueError, match="doc_ids"):
            ra_attention(q.detach(), k.detach(), v.detach(), doc_ids=torch.ones(B, N + 1, dtype=torch.int32))
        with pytest.raises(ValueError, match="doc_ids"):
            ra_attention(q.detach(), k.detach(), v.detach(), doc_ids=torch.ones(B, 1, N, dtype=torch.int32))
        with pytest.raises(TypeError, match="integer"):
            ra_attention(q.detach(), k.detach(), v.detach(), doc_ids=torch.full((B, N), 1.5, dtype=torch.float32))
        with pytest.raises(TypeError, match="integer"):
            ra_attention(
                q.detach().numpy().astype(np.float32),
                k.detach().numpy().astype(np.float32),
                v.detach().numpy().astype(np.float32),
                doc_ids=np.full((B, N), 1.5, dtype=np.float32),
            )

    @pytest.mark.parametrize("B,H", [(2, 4), (2, 2)])
    def test_doc_ids_with_3d_mask_and_differentiable_attn_mask(self, B: int, H: int) -> None:
        torch.manual_seed(314)
        N, D = 8, 8
        doc_ids = torch.tensor([[1, 1, 1, 1, 2, 2, 0, 0], [1, 1, 2, 2, 2, 2, 3, 0]], dtype=torch.int32)
        q0 = torch.randn(B, H, N, D, dtype=torch.float64)
        k0 = torch.randn(B, H, N, D, dtype=torch.float64)
        v0 = torch.randn(B, H, N, D, dtype=torch.float64)
        do = torch.randn(B, H, N, D, dtype=torch.float64)

        # 3D mask (B, N, N) + doc_ids must broadcast per-batch (B, 1, N, N) without head cross-contamination.
        m3d = (torch.randn(B, N, N, dtype=torch.float64) * 0.2).requires_grad_(True)
        q1, k1, v1 = (t.clone().requires_grad_(True) for t in (q0, k0, v0))
        out1 = ra_attention(q1, k1, v1, attn_mask=m3d, is_causal=True, doc_ids=doc_ids)
        out1.backward(do)

        ref_keep = _reference_doc_mask(doc_ids, is_causal=True)
        m3d_ref = m3d.detach().clone().requires_grad_(True)
        q2, k2, v2 = (t.clone().requires_grad_(True) for t in (q0, k0, v0))
        scores = (q2 @ k2.transpose(-2, -1)) / math.sqrt(D)
        scores = scores.masked_fill(~ref_keep, float("-inf")) + m3d_ref.unsqueeze(1)
        out2 = _reference_ra_attention_with_bool_mask(q2, k2, v2, ref_keep)
        # Exact reference with additive bias via ra_attention(..., attn_mask=eff_bias, is_causal=False)
        eff_bias = m3d_ref.unsqueeze(1) + torch.where(ref_keep, torch.zeros_like(m3d_ref.unsqueeze(1)), torch.full_like(m3d_ref.unsqueeze(1), float("-inf")))
        out_ref = ra_attention(q2, k2, v2, attn_mask=eff_bias, is_causal=False)
        out_ref.backward(do)
        assert out2.shape == out1.shape
        assert torch.allclose(out1, out_ref, atol=1e-10)
        assert m3d.grad is not None and m3d_ref.grad is not None
        assert torch.allclose(m3d.grad, m3d_ref.grad, atol=1e-10)

        # Broadcast differentiable 4D mask (1, 1, N, N) + doc_ids must preserve mask grad & shape.
        m_bcast = (torch.randn(1, 1, N, N, dtype=torch.float64) * 0.2).requires_grad_(True)
        out_b = ra_attention(q0.clone(), k0.clone(), v0.clone(), attn_mask=m_bcast, is_causal=True, doc_ids=doc_ids)
        out_b.backward(do)
        assert m_bcast.grad is not None and tuple(m_bcast.grad.shape) == (1, 1, N, N)
        assert torch.isfinite(m_bcast.grad).all()

    def test_triton_doc_ids_backward_fallback_handles_8_tensors_and_padding_sentinel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import luminamoon.nn._attention_ops as ops_mod

        B, H, N, D = 2, 2, 8, 8
        doc_ids = torch.tensor([[1, 1, 1, 1, 2, 2, 0, 0], [1, 1, 2, 2, 2, 3, 3, 0]], dtype=torch.int32)
        q = torch.randn(B, H, N, D, dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=torch.float32, requires_grad=True)

        # Simulate Triton SRAM forward returning sentinel m_idx == N on padding rows, then Triton bwd returning None.
        ref_keep = _reference_doc_mask(doc_ids, is_causal=True)
        with torch.no_grad():
            s = (q @ k.transpose(-2, -1)) / math.sqrt(D)
            s = s.masked_fill(~ref_keep, float("-inf"))
            m_val, m_idx = s.max(dim=-1)
            m_idx = torch.where(torch.isfinite(m_val), m_idx.to(torch.int32), torch.full_like(m_idx, N, dtype=torch.int32))
            p, _, l_val, _ = ops_mod._ra_attn_weights_fwd(s, 0.0625, q_start=0, is_causal=False)
            out_fake = p @ v

        monkeypatch.setattr(ops_mod, "_HAS_TRITON", True)
        monkeypatch.setattr(ops_mod, "triton_flash_ra_attention", lambda *a, **kw: (out_fake, m_val, l_val.squeeze(-1), m_idx))
        monkeypatch.setattr(ops_mod, "triton_flash_ra_attention_backward", lambda *a, **kw: None)
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
        out = ops_mod._FlashRaAttentionFunction.apply(q, k, v, None, doc_ids, True, 1.0 / math.sqrt(D), 0.0625, None)
        out.sum().backward()
        assert q.grad is not None and k.grad is not None and v.grad is not None
        assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all() and torch.isfinite(v.grad).all()

    def test_doc_ids_torch_compile_fullgraph(self) -> None:
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile is not available")
        torch.manual_seed(123)
        attn = RaAttention(dim=16, num_heads=2, is_causal=True, qk_norm=True)
        c_attn = torch.compile(attn, backend="aot_eager", fullgraph=True)
        doc_ids = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 0], [1, 1, 2, 2, 2, 3, 3, 3]], dtype=torch.int32)
        x1 = torch.randn(2, 8, 16, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)

        y1 = attn(x1, doc_ids=doc_ids)
        y2 = c_attn(x2, doc_ids=doc_ids)
        assert torch.allclose(y1, y2, atol=1e-5)
        y1.sum().backward()
        y2.sum().backward()
        assert x1.grad is not None and x2.grad is not None
        assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
