from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(ROOT_PYTHON) not in sys.path:
    if any((ROOT_PYTHON / "luminamoon" / "optimizer").glob("_luminamoon_optimizer_rs*")):
        sys.path.insert(0, str(ROOT_PYTHON))
    else:
        sys.path.append(str(ROOT_PYTHON))

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

import luminamoon.nn as lnn
from luminamoon.nn import (
    DirectiveGate,
    DistanceDecayBias,
    LatentKVAttention,
    OrthogonalHub,
    QRoPE,
    ResidualRollout,
    ReversibleDKDBlock,
    Sinusoidal3DEmbedding,
    SlotCrossAttention,
    TwoTierMemory,
    qrope,
)


def test_qrope_isometry_shift_invariance_and_backward_parity() -> None:
    torch.manual_seed(0)
    rope = QRoPE(head_dim=32, max_seq_len=32)

    # Dynamic cache growth beyond initial max_seq_len + L2 norm preservation.
    x = torch.randn(2, 4, 48, 32, requires_grad=True)
    y = rope(x, offset=0)
    assert rope._cached_len >= 48
    assert (y.norm(dim=-1) - x.norm(dim=-1)).abs().max().item() < 1e-5

    # Relative-position shift invariance: <R(m)q, R(n)k> == <R(m+s)q, R(n+s)k>.
    q0 = torch.randn(1, 2, 1, 32)
    k0 = torch.randn(1, 2, 1, 32)
    for m, n, s in ((2, 9, 11), (0, 5, 40), (15, 3, 19)):
        d1 = (rope(q0, offset=m) * rope(k0, offset=n)).sum(dim=-1)
        d2 = (rope(q0, offset=m + s) * rope(k0, offset=n + s)).sum(dim=-1)
        assert (d1 - d2).abs().max().item() < 1e-5

    # Backward through custom autograd function matches eager autograd.
    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    assert x.grad is not None

    x_ref = x.detach().clone().requires_grad_(True)
    cl = rope.cos_l[:48]
    sl = rope.sin_l[:48]
    cr = rope.cos_r[:48]
    sr = rope.sin_r[:48]
    y_ref = lnn.position._qrope_eager(x_ref, cl, sl, cr, sr, sign=1.0)
    y_ref.backward(grad_out)
    assert x_ref.grad is not None
    assert (x.grad - x_ref.grad).abs().max().item() < 1e-5

    # BF16/FP16 finite check for QRoPE.
    for dt in (torch.bfloat16, torch.float16):
        xt = torch.randn(1, 2, 12, 16, dtype=dt, requires_grad=True)
        rt = QRoPE(head_dim=16, max_seq_len=32)
        yt = rt(xt, offset=3)
        assert yt.dtype == dt and torch.isfinite(yt).all()
        yt.float().sum().backward()
        assert xt.grad is not None and torch.isfinite(xt.grad).all()

    # head_dim validation for QRoPE and functional qrope.

    with pytest.raises(ValueError, match="divisible by 4"):
        QRoPE(head_dim=10)
    with pytest.raises(ValueError, match="divisible by 4"):
        qrope(torch.randn(1, 2, 4, 6), cl[:4], sl[:4], cr[:4], sr[:4])


def test_distance_decay_bias_modes_hub_damping_and_learnable() -> None:
    torch.manual_seed(1)
    for mode in ("linear", "rational"):
        bias = DistanceDecayBias(num_heads=8, hub_size=8, num_hub_heads=1, mode=mode, learnable=True)
        b_mat = bias(seq_len=24, num_all=24, device=torch.device("cpu"), dtype=torch.float32, is_dense_with_hub=True)
        assert b_mat.shape == (1, 8, 24, 24)
        assert torch.isfinite(b_mat).all()
        # Hub head (0) has zero hub penalty; text heads (1..7) receive negative hub bias.
        assert b_mat[0, 0, :, :8].abs().max().item() == 0.0
        assert b_mat[0, 1, :, :8].max().item() <= -0.7
        assert b_mat[0, 7, :, :8].max().item() < b_mat[0, 1, :, :8].min().item()
        b_mat.sum().backward()
        assert bias.raw_slopes.grad is not None and torch.isfinite(bias.raw_slopes.grad).all()

    # Decode step (has_past_kv=True) matches dense slice for last token.
    bias_lin = DistanceDecayBias(num_heads=4, hub_size=4, mode="linear")
    dense = bias_lin(seq_len=12, num_all=12, device=torch.device("cpu"), dtype=torch.float32, is_dense_with_hub=True)
    step1 = bias_lin(
        seq_len=1, num_all=12, device=torch.device("cpu"), dtype=torch.float32, is_dense_with_hub=False, has_past_kv=True
    )
    assert (dense[:, :, -1:, :] - step1).abs().max().item() < 1e-6


def test_sinusoidal_3d_embedding_dims_and_dtype() -> None:
    for dim in (6, 8, 12, 32, 64, 256):
        emb = Sinusoidal3DEmbedding(dim=dim)
        z = torch.tensor([[0, 1, 5]])
        y = torch.tensor([[0, 2, 4]])
        x = torch.tensor([[0, 3, 7]])
        out = emb(z, y, x, dtype=torch.bfloat16)
        assert out.shape == (1, 3, dim)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()
    with pytest.raises(ValueError, match="even dim >= 6"):
        Sinusoidal3DEmbedding(dim=4)


@pytest.mark.parametrize("backend", ["ra", "sdpa"])
def test_latent_kv_attention_causality_and_kv_cache_parity(backend: str) -> None:
    torch.manual_seed(2)
    attn = LatentKVAttention(
        dim=32, num_heads=4, head_dim=8, kv_latent_dim=16, hub_size=4, bias_mode="linear", attention_backend=backend
    ).eval()

    x = torch.randn(2, 16, 32)  # 4 hub tokens + 12 text tokens
    with torch.no_grad():
        full_out, _ = attn(x, past_kv=None, is_dense_with_hub=True)
        # Future-token perturbation after index 10 must not affect tokens <= 10.
        x_pert = x.clone()
        x_pert[:, 11:, :] += 2.0
        pert_out, _ = attn(x_pert, past_kv=None, is_dense_with_hub=True)
        assert (full_out[:, :11, :] - pert_out[:, :11, :]).abs().max().item() < 1e-6

        # Incremental KV-cache decode matches full pass.
        prefill_out, kv = attn(x[:, :10, :], past_kv=None, is_dense_with_hub=True)
        outs = [prefill_out]
        for t in range(10, 16):
            step_out, kv = attn(x[:, t : t + 1, :], past_kv=kv, is_dense_with_hub=False)
            outs.append(step_out)
        cat_out = torch.cat(outs, dim=1)
        assert (full_out - cat_out).abs().max().item() < 1e-5


def test_slot_cross_attention_decorrelation_and_precision() -> None:
    torch.manual_seed(3)
    for backend in ("ra", "sdpa"):
        mod = SlotCrossAttention(dim=32, num_heads=4, num_terminals=4, slots_per_terminal=4, attention_backend=backend)
        term_base = torch.randn(2, 4, 32, requires_grad=True)
        ctx = torch.randn(2, 20, 32, requires_grad=True)
        out = mod(term_base, ctx)
        assert out.shape == (2, 16, 32) and torch.isfinite(out).all()
        out.pow(2).mean().backward()
        assert term_base.grad is not None and ctx.grad is not None

    # Verify orthogonally initialized slot queries remain decorrelated after FiLM conditioning and RaNorm.
    mod_sdpa = SlotCrossAttention(dim=32, num_heads=4, num_terminals=4, slots_per_terminal=4, attention_backend="sdpa")
    tb = torch.randn(2, 4, 32)
    with torch.no_grad():
        film = mod_sdpa.slot_film(tb).view(2, 4, 4, 2, 32)
        q_slots = mod_sdpa.q_norm(mod_sdpa.slot_queries * (1.0 + film[..., 0, :]) + film[..., 1, :])
        qn = F.normalize(q_slots, p=2, dim=-1)
        gram = torch.matmul(qn, qn.transpose(-1, -2))
        eye = torch.eye(4).view(1, 1, 4, 4)
        off_cos = (gram.abs() * (1.0 - eye)).sum().item() / (2 * 4 * 4 * 3)
        assert off_cos < 0.25

        # With larger-scale random context vectors (sharper attention weights), output slots remain decorrelated.
        out_sharp = mod_sdpa(tb, torch.randn(2, 20, 32) * 3.0)
        sn = F.normalize(out_sharp.view(2, 4, 4, 32), p=2, dim=-1)
        gram_out = torch.matmul(sn, sn.transpose(-1, -2))
        off_cos_out = (gram_out.abs() * (1.0 - eye)).sum().item() / (2 * 4 * 4 * 3)
        assert off_cos_out < 0.5

    mod_bf = SlotCrossAttention(dim=32, num_heads=4, num_terminals=2, slots_per_terminal=2).to(torch.bfloat16)
    out_bf = mod_bf(torch.randn(1, 2, 32, dtype=torch.bfloat16), torch.randn(1, 8, 32, dtype=torch.bfloat16))
    assert out_bf.dtype == torch.bfloat16 and torch.isfinite(out_bf).all()


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.bfloat16, 2e-3)])
def test_reversible_dkd_block_grad_parity(dtype: torch.dtype, tol: float) -> None:
    torch.manual_seed(4)
    layers = nn.ModuleList([
        ReversibleDKDBlock(
            dim=64, num_heads=4, head_dim=16, kv_latent_dim=32, hub_size=0, dt=0.5,
            num_layers=3, reversible=False, attention_backend="ra",
        ).to(dtype)
        for _ in range(3)
    ])

    q0 = torch.randn(2, 12, 32, dtype=dtype)
    p0 = torch.randn(2, 12, 32, dtype=dtype)

    # Standard autograd pass
    qs, ps = q0.clone().requires_grad_(True), p0.clone().requires_grad_(True)
    q, p = qs, ps
    for layer in layers:
        q, p, _ = layer(q, p)
    (q.float().pow(2).mean() + p.float().pow(2).mean()).backward()
    out_std = torch.cat((q, p), dim=-1).detach().float()
    gq_std, gp_std = qs.grad.detach().float(), ps.grad.detach().float()
    gp_params_std = [pr.grad.detach().float().clone() for pr in layers.parameters()]

    # Reversible autograd pass
    layers.zero_grad()
    for layer in layers:
        layer.reversible = True
    qr, pr_in = q0.clone().requires_grad_(True), p0.clone().requires_grad_(True)
    q, p = qr, pr_in
    for layer in layers:
        q, p, kv = layer(q, p)
        assert kv is None
    (q.float().pow(2).mean() + p.float().pow(2).mean()).backward()
    out_rev = torch.cat((q, p), dim=-1).detach().float()
    gq_rev, gp_rev = qr.grad.detach().float(), pr_in.grad.detach().float()
    gp_params_rev = [pr.grad.detach().float().clone() for pr in layers.parameters()]

    assert (out_std - out_rev).abs().max().item() < tol
    assert (gq_std - gq_rev).abs().max().item() < tol
    assert (gp_std - gp_rev).abs().max().item() < tol
    for a, b in zip(gp_params_std, gp_params_rev):
        assert (a - b).abs().max().item() < tol


def test_residual_rollout_orthogonal_hub_directive_gate_and_two_tier_memory() -> None:
    torch.manual_seed(5)
    # 1. ResidualRollout (ra_tanh default + BF16 finite check)
    roll = ResidualRollout(dim=32, num_steps=3, activation="ra_tanh").to(torch.bfloat16)
    xr = torch.randn(2, 6, 32, dtype=torch.bfloat16, requires_grad=True)
    yr = roll(xr)
    assert yr.dtype == torch.bfloat16 and torch.isfinite(yr).all()
    yr.float().pow(2).mean().backward()
    assert xr.grad is not None and torch.isfinite(xr.grad).all()

    # 2. OrthogonalHub (SO(H) orthogonality, BF16 support, eval cache invalidation)
    hub = OrthogonalHub(hub_size=8, dim=32, use_context_gate=True).to(torch.bfloat16)
    q_mat = hub.get_orthogonal_matrix().float()
    assert (q_mat.T @ q_mat - torch.eye(8)).abs().max().item() < 1e-2
    out_h = hub(2, context=torch.randn(2, 6, 32, dtype=torch.bfloat16))
    assert out_h.shape == (2, 8, 32) and out_h.dtype == torch.bfloat16 and torch.isfinite(out_h).all()
    hub.eval()
    q_e1 = hub.get_orthogonal_matrix().float().clone()
    with torch.no_grad():
        hub.weight_skew[0, 1] += 0.5
    q_e2 = hub.get_orthogonal_matrix().float()
    assert (q_e1 - q_e2).abs().max().item() > 0.1

    # 3. DirectiveGate preserves aligned vs orthogonal separation (not canceled by norm)
    dg = DirectiveGate(dim=32)
    dir_tok = torch.randn(2, 2, 32)
    d_vec = dg.directive_norm(dg.directive_proj(dir_tok.mean(dim=1, keepdim=True)))
    v_unit = F.normalize(d_vec, p=2, dim=-1)
    chunks = torch.randn(2, 8, 32)
    chunks = chunks - (chunks * v_unit).sum(dim=-1, keepdim=True) * v_unit
    chunks[:, 0:1, :] = v_unit * math.sqrt(32)
    dg_out = dg(chunks, dir_tok)
    rms_aligned = dg_out[:, 0, :].pow(2).mean().sqrt().item()
    rms_orth = dg_out[:, 1:, :].pow(2).mean().sqrt().item()
    assert rms_aligned > 1.3 * rms_orth

    # 4. TwoTierMemory propagates gradients to query in small and top-k multi-head cases.
    mem = TwoTierMemory(dim=32, max_l1=4, max_l2=8, num_heads=4, threshold=0.0, temperature=0.25)
    for n_chunks, expected_l2 in ((6, 6), (16, 8)):
        c = torch.randn(2, n_chunks, 32, requires_grad=True)
        q = torch.randn(2, 3, 32, requires_grad=True)
        l1, l2 = mem(c, q)
        assert l1.shape == (2, 4, 32) and l2.shape == (2, expected_l2, 32)
        (l1.pow(2).mean() + l2.pow(2).mean()).backward()
        assert q.grad is not None and q.grad.norm().item() > 0.0 and torch.isfinite(q.grad).all()
        assert c.grad is not None and torch.isfinite(c.grad).all()

    # Soft-only warmup mode keeps all chunks even when num_chunks > max_l2.
    l1_soft, l2_soft = mem(torch.randn(2, 16, 32), torch.randn(2, 3, 32), use_topk=False)
    assert l1_soft.shape == (2, 4, 32) and l2_soft.shape == (2, 16, 32)

    # Multi-head token-max scoring preserves minority query targets alongside repeated query topics.
    mem_noproj = TwoTierMemory(dim=32, max_l1=2, max_l2=4, num_heads=4, use_projections=False)
    basis = F.normalize(torch.eye(32), dim=-1)
    chunks_mh = basis[4:24].unsqueeze(0).clone()
    chunks_mh[0, 12] = basis[0]
    chunks_mh[0, 18] = basis[1]
    q_imbal = torch.stack([basis[0], basis[0], basis[0], basis[0], basis[1]], dim=0).unsqueeze(0)
    _, l2_mh = mem_noproj(chunks_mh, q_imbal)
    assert l2_mh.shape == (1, 4, 32) and torch.isfinite(l2_mh).all()
    # Both target basis vectors should align with the retrieved top-4 L2 set.
    sim_t0 = F.cosine_similarity(l2_mh[0], basis[0].unsqueeze(0), dim=-1).max().item()
    sim_t1 = F.cosine_similarity(l2_mh[0], basis[1].unsqueeze(0), dim=-1).max().item()
    assert sim_t0 > 0.8 and sim_t1 > 0.8


def test_reversible_latent_nn_modules_import_without_triton() -> None:
    script = (
        "import sys\n"
        "sys.modules['triton'] = None\n"
        "sys.modules['triton.language'] = None\n"
        f"sys.path.insert(0, {str(ROOT_PYTHON)!r})\n"
        "import torch\n"
        "import luminamoon.nn as lnn\n"
        "assert set(dir(lnn)) >= {'QRoPE', 'DistanceDecayBias', 'Sinusoidal3DEmbedding', "
        "'LatentKVAttention', 'SlotCrossAttention', 'ReversibleDKDBlock', 'ResidualRollout', "
        "'OrthogonalHub', 'DirectiveGate', 'TwoTierMemory'}\n"
        "r = lnn.QRoPE(head_dim=16, max_seq_len=32)\n"
        "y = r(torch.randn(1, 2, 8, 16))\n"
        "assert y.shape == (1, 2, 8, 16)\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"Subprocess failed:\n{proc.stdout}\n{proc.stderr}"


def test_mask_free_causal_path_attention_fn_and_custom_dkd_submodules() -> None:
    torch.manual_seed(6)
    calls: list[dict[str, object]] = []

    def _custom_attn(q, k, v, *, attn_mask, is_causal, scale):
        calls.append({"mask_is_none": attn_mask is None, "is_causal": is_causal, "scale": scale})
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale)

    # Mask-free causal path (attn_mask=None) when bias_mode="none" and hub_size=0 (prefill + 1-token decode).
    attn = LatentKVAttention(
        dim=32, num_heads=4, head_dim=8, kv_latent_dim=16,
        hub_size=0, bias_mode="none", attention_fn=_custom_attn,
    ).eval()
    x = torch.randn(2, 10, 32)
    with torch.no_grad():
        full_out, _ = attn(x, past_kv=None, is_dense_with_hub=True)
        assert calls[-1]["mask_is_none"] is True and calls[-1]["is_causal"] is True

        pre_out, kv = attn(x[:, :8, :], past_kv=None, is_dense_with_hub=True)
        step1_out, kv = attn(x[:, 8:9, :], past_kv=kv, is_dense_with_hub=False)
        assert calls[-1]["mask_is_none"] is True and calls[-1]["is_causal"] is False
        step2_out, _ = attn(x[:, 9:10, :], past_kv=kv, is_dense_with_hub=False)
        assert (full_out - torch.cat((pre_out, step1_out, step2_out), dim=1)).abs().max().item() < 1e-5

    # SlotCrossAttention also accepts custom attention_fn.
    slot_calls = len(calls)
    slot_attn = SlotCrossAttention(dim=32, num_heads=4, num_terminals=2, slots_per_terminal=2, attention_fn=_custom_attn)
    s_out = slot_attn(torch.randn(2, 2, 32), torch.randn(2, 8, 32))
    assert s_out.shape == (2, 4, 32) and len(calls) == slot_calls + 1

    # ReversibleDKDBlock wraps standard PyTorch LayerNorm + plain attention + MLP with reversible=True.
    class _PlainCausalAttn(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.qkv = nn.Linear(dim, 3 * dim, bias=False)
            self.out = nn.Linear(dim, dim, bias=False)

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            q, k, v = self.qkv(h).chunk(3, dim=-1)
            q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True).squeeze(1)
            return self.out(y)

    plain_mlp = nn.Sequential(nn.Linear(16, 32, bias=False), nn.SiLU(), nn.Linear(32, 16, bias=False))
    blk = ReversibleDKDBlock(
        dim=32, dt=0.5, reversible=False,
        kick_attn=_PlainCausalAttn(16), drift_ffn=plain_mlp, norm_cls=nn.LayerNorm,
    )
    q0, p0 = torch.randn(2, 8, 16), torch.randn(2, 8, 16)
    qs, ps = q0.clone().requires_grad_(True), p0.clone().requires_grad_(True)
    qo_s, po_s, _ = blk(qs, ps)
    (qo_s.pow(2).mean() + po_s.pow(2).mean()).backward()
    gq_s, gp_s = qs.grad.detach().clone(), ps.grad.detach().clone()

    blk.zero_grad()
    blk.reversible = True
    qr, pr = q0.clone().requires_grad_(True), p0.clone().requires_grad_(True)
    qo_r, po_r, _ = blk(qr, pr)
    (qo_r.pow(2).mean() + po_r.pow(2).mean()).backward()
    assert (qo_s - qo_r).abs().max().item() < 1e-5
    assert (gq_s - qr.grad).abs().max().item() < 1e-5
    assert (gp_s - pr.grad).abs().max().item() < 1e-5


def test_attention_backend_presets_inheritance_and_flex_parity() -> None:
    import types
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.manual_seed(7)
    # 1. Default attention_backend="auto" inherits outer PyTorch sdpa_kernel without overriding it.
    attn_auto = LatentKVAttention(dim=32, num_heads=4, head_dim=8, kv_latent_dim=16, hub_size=0, bias_mode="none").eval()
    assert attn_auto.attention_backend == "auto"
    x = torch.randn(2, 10, 32)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        y_math, _ = attn_auto(x)
    assert y_math.shape == (2, 10, 32) and torch.isfinite(y_math).all()

    # 2. use_attention_backend inherits into "auto" modules and does not override explicit backends.
    seen: list[str] = []

    def _tracked_attn(q, k, v, *, attn_mask, is_causal, scale):
        seen.append("custom_ctx")
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale)

    attn_explicit_sdpa = LatentKVAttention(
        dim=32, num_heads=4, head_dim=8, kv_latent_dim=16, hub_size=0, bias_mode="none", attention_backend="sdpa"
    ).eval()
    assert lnn.get_attention_backend() is None
    with lnn.use_attention_backend(_tracked_attn):
        assert lnn.get_attention_backend() is _tracked_attn
        with torch.no_grad():
            _ = attn_auto(x)
            _ = attn_explicit_sdpa(x)
        with lnn.use_attention_backend("auto"):
            assert lnn.get_attention_backend() is None
            with torch.no_grad():
                _ = attn_auto(x)
    assert lnn.get_attention_backend() is None
    assert seen == ["custom_ctx"]

    # 3. Verify "fa3" transposes BHSD <-> BSHD when calling a mocked flash_attn_interface.
    fa3_calls: list[dict[str, object]] = []

    def _fake_fa3(q_bshd, k_bshd, v_bshd, *, softmax_scale, causal):
        fa3_calls.append({"q_shape": tuple(q_bshd.shape), "causal": causal, "scale": softmax_scale})
        q_bhsd = q_bshd.transpose(1, 2)
        k_bhsd = k_bshd.transpose(1, 2)
        v_bhsd = v_bshd.transpose(1, 2)
        out = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd, is_causal=causal, scale=softmax_scale)
        return out.transpose(1, 2)

    fake_mod = types.SimpleNamespace(flash_attn_func=_fake_fa3)
    prev_mod = sys.modules.get("flash_attn_interface")
    sys.modules["flash_attn_interface"] = fake_mod  # type: ignore[assignment]
    try:
        attn_fa3 = LatentKVAttention(
            dim=32, num_heads=4, head_dim=8, kv_latent_dim=16, hub_size=0, bias_mode="none", attention_backend="fa3"
        ).eval()
        attn_fa3.load_state_dict(attn_auto.state_dict())
        with torch.no_grad():
            y_fa3, _ = attn_fa3(x)
        assert fa3_calls and fa3_calls[0]["q_shape"] == (2, 10, 4, 8) and fa3_calls[0]["causal"] is True
        assert (y_fa3 - y_math).abs().max().item() < 1e-5
    finally:
        if prev_mod is None:
            sys.modules.pop("flash_attn_interface", None)
        else:
            sys.modules["flash_attn_interface"] = prev_mod

    # 4. Mask validation for flash/cudnn/fa3 raises ValueError when dense bias is enabled.
    for backend in ("flash", "cudnn", "fa3"):
        attn_biased = LatentKVAttention(
            dim=32, num_heads=4, head_dim=8, kv_latent_dim=16, hub_size=0, bias_mode="linear", attention_backend=backend
        ).eval()
        with pytest.raises(ValueError, match="requires attn_mask=None"):
            with torch.no_grad():
                attn_biased(x)

    # 5. "flex" on-the-fly score_mod matches materialized "sdpa" across bias modes, hubs, and KV-cache decode.
    if hasattr(torch.nn.attention, "flex_attention"):
        for bias_mode, hub_size in (("none", 0), ("linear", 4), ("rational", 4)):
            ref = LatentKVAttention(
                dim=32, num_heads=4, head_dim=8, kv_latent_dim=16,
                hub_size=hub_size, bias_mode=bias_mode, attention_backend="sdpa",
            ).eval()
            flx = LatentKVAttention(
                dim=32, num_heads=4, head_dim=8, kv_latent_dim=16,
                hub_size=hub_size, bias_mode=bias_mode, attention_backend="flex",
            ).eval()
            flx.load_state_dict(ref.state_dict())
            x_seq = torch.randn(2, 12, 32)
            with torch.no_grad():
                y_ref, _ = ref(x_seq, past_kv=None, is_dense_with_hub=True)
                y_flx, _ = flx(x_seq, past_kv=None, is_dense_with_hub=True)
                assert (y_ref - y_flx).abs().max().item() < 1e-5

                pre_r, kv_r = ref(x_seq[:, :10, :], past_kv=None, is_dense_with_hub=True)
                pre_f, kv_f = flx(x_seq[:, :10, :], past_kv=None, is_dense_with_hub=True)
                step_r, _ = ref(x_seq[:, 10:11, :], past_kv=kv_r, is_dense_with_hub=False)
                step_f, _ = flx(x_seq[:, 10:11, :], past_kv=kv_f, is_dense_with_hub=False)
                assert (pre_r - pre_f).abs().max().item() < 1e-5
                assert (step_r - step_f).abs().max().item() < 1e-5

        s_ref = SlotCrossAttention(dim=32, num_heads=4, num_terminals=2, slots_per_terminal=2, attention_backend="sdpa").eval()
        s_flx = SlotCrossAttention(dim=32, num_heads=4, num_terminals=2, slots_per_terminal=2, attention_backend="flex").eval()
        s_flx.load_state_dict(s_ref.state_dict())
        tb, ctx = torch.randn(2, 2, 32), torch.randn(2, 8, 32)
        with torch.no_grad():
            assert (s_ref(tb, ctx) - s_flx(tb, ctx)).abs().max().item() < 1e-5


def test_kv_cache_mode_and_cached_twotier_keys() -> None:
    torch.manual_seed(8)
    # 1. LatentKVAttention cache_mode="kv" matches full prefill and cache_mode="latent".
    for bias_mode, hub_size in (("none", 0), ("linear", 4)):
        attn_lat = LatentKVAttention(
            dim=32, num_heads=4, head_dim=8, kv_latent_dim=16,
            hub_size=hub_size, bias_mode=bias_mode, attention_backend="sdpa", cache_mode="latent",
        ).eval()
        attn_kv = LatentKVAttention(
            dim=32, num_heads=4, head_dim=8, kv_latent_dim=16,
            hub_size=hub_size, bias_mode=bias_mode, attention_backend="sdpa", cache_mode="kv",
        ).eval()
        attn_kv.load_state_dict(attn_lat.state_dict())
        x = torch.randn(2, 14, 32)
        with torch.no_grad():
            full_lat, _ = attn_lat(x, past_kv=None, is_dense_with_hub=True)
            pre_kv, kv = attn_kv(x[:, :10, :], past_kv=None, is_dense_with_hub=True)
            assert isinstance(kv, tuple) and len(kv) == 2 and kv[0].shape == (2, 4, 10, 8)
            outs = [pre_kv]
            for t in range(10, 14):
                st_out, kv = attn_kv(x[:, t : t + 1, :], past_kv=kv, is_dense_with_hub=False)
                outs.append(st_out)
            assert (full_lat - torch.cat(outs, dim=1)).abs().max().item() < 1e-5

    # 2. ReversibleDKDBlock forwards cache_mode="kv" to LatentKVAttention.
    blk_lat = ReversibleDKDBlock(dim=64, num_heads=4, head_dim=16, kv_latent_dim=32, hub_size=0, bias_mode="none", cache_mode="latent").eval()
    blk_kv = ReversibleDKDBlock(dim=64, num_heads=4, head_dim=16, kv_latent_dim=32, hub_size=0, bias_mode="none", cache_mode="kv").eval()
    blk_kv.load_state_dict(blk_lat.state_dict())
    q_in, p_in = torch.randn(2, 10, 32), torch.randn(2, 10, 32)
    with torch.no_grad():
        q_full, p_full, _ = blk_lat(q_in, p_in, past_kv=None, is_dense_with_hub=True)
        q_pre, p_pre, kv = blk_kv(q_in[:, :8], p_in[:, :8], past_kv=None, is_dense_with_hub=True)
        q_st, p_st, kv = blk_kv(q_in[:, 8:9], p_in[:, 8:9], past_kv=kv, is_dense_with_hub=False)
        q_st2, p_st2, _ = blk_kv(q_in[:, 9:10], p_in[:, 9:10], past_kv=kv, is_dense_with_hub=False)
        assert (q_full - torch.cat((q_pre, q_st, q_st2), dim=1)).abs().max().item() < 1e-5
        assert (p_full - torch.cat((p_pre, p_st, p_st2), dim=1)).abs().max().item() < 1e-5

    # 3. TwoTierMemory.encode_keys + cached_keys matches uncached forward and validates shape.
    mem = TwoTierMemory(dim=32, max_l1=4, max_l2=8, num_heads=4).eval()
    chunks = torch.randn(2, 20, 32)
    query = torch.randn(2, 3, 32)
    with torch.no_grad():
        l1_ref, l2_ref = mem(chunks, query)
        k_cached = mem.encode_keys(chunks)
        assert k_cached.shape == (2, 4, 20, 8)
        l1_c, l2_c = mem(chunks, query, cached_keys=k_cached)
        assert (l1_ref - l1_c).abs().max().item() < 1e-6
        assert (l2_ref - l2_c).abs().max().item() < 1e-6
    with pytest.raises(ValueError, match="cached_keys must have shape"):
        mem(chunks, query, cached_keys=torch.randn(2, 4, 19, 8))

    # 4. FP64 finite-difference gradcheck for QRoPE and TwoTierMemory.
    qr64 = QRoPE(head_dim=8, max_seq_len=16).to(torch.float64)
    xq64 = torch.randn(1, 2, 4, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: qr64(t, offset=3), (xq64,), eps=1e-6, atol=1e-4, rtol=1e-3)

    mem64 = TwoTierMemory(dim=12, max_l1=2, max_l2=6, num_heads=3, use_topk=False).to(torch.float64)
    c64 = torch.randn(1, 5, 12, dtype=torch.float64, requires_grad=True)
    q64 = torch.randn(1, 2, 12, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda ch, qu: sum(t.sum() for t in mem64(ch, qu)), (c64, q64), eps=1e-6, atol=1e-4, rtol=1e-3
    )

    # 5. ReversibleDKDBlock on-demand grad_map backward matches non-reversible with partially frozen params.
    torch.manual_seed(77)
    blk_r = ReversibleDKDBlock(dim=32, num_heads=4, hub_size=2, reversible=True, bias_mode="linear")
    blk_s = ReversibleDKDBlock(dim=32, num_heads=4, hub_size=2, reversible=False, bias_mode="linear")
    blk_s.load_state_dict(blk_r.state_dict())
    blk_r.norm_q.weight.requires_grad_(False)
    blk_s.norm_q.weight.requires_grad_(False)
    qr = torch.randn(2, 10, 16, requires_grad=True)
    pr = torch.randn(2, 10, 16, requires_grad=True)
    qs = qr.detach().clone().requires_grad_(True)
    ps = pr.detach().clone().requires_grad_(True)
    oq_r, op_r, _ = blk_r(qr, pr)
    (oq_r.square().mean() + op_r.square().mean()).backward()
    oq_s, op_s, _ = blk_s(qs, ps)
    (oq_s.square().mean() + op_s.square().mean()).backward()
    assert blk_r.norm_q.weight.grad is None and blk_s.norm_q.weight.grad is None
    assert (qr.grad - qs.grad).abs().max().item() < 1e-4
    assert (pr.grad - ps.grad).abs().max().item() < 1e-4
    for (nr, pr_p), (_, ps_p) in zip(blk_r.named_parameters(), blk_s.named_parameters()):
        if pr_p.requires_grad:
            assert (pr_p.grad - ps_p.grad).abs().max().item() < 1e-4, f"grad mismatch on {nr}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton fused bias parity")
def test_triton_ra_fused_bias_cuda_parity() -> None:
    from luminamoon.nn._attention_ops import ra_attention_fused_bias
    from luminamoon.nn.attention import ra_attention

    device = torch.device("cuda")
    for bias_mode in ("linear", "rational"):
        for hub_size, is_dense, has_past, n_q, n_k in (
            (0, False, False, 128, 128),
            (16, True, False, 144, 144),
            (16, False, True, 64, 192),
        ):
            torch.manual_seed(123)
            q = torch.randn(2, 8, n_q, 64, device=device, dtype=torch.float32, requires_grad=True)
            k = torch.randn(2, 8, n_k, 64, device=device, dtype=torch.float32, requires_grad=True)
            v = torch.randn(2, 8, n_k, 64, device=device, dtype=torch.float32, requires_grad=True)
            db = DistanceDecayBias(num_heads=8, hub_size=hub_size, mode=bias_mode).to(device)
            lat = LatentKVAttention(dim=512, num_heads=8, hub_size=hub_size, bias_mode=bias_mode).to(device)
            scale = 1.0 / (64 ** 0.5)

            causal = lat._build_causal_mask(n_q, n_k, device, q.dtype, is_dense, has_past)
            decay = db(n_q, n_k, device, q.dtype, is_dense_with_hub=is_dense, has_past_kv=has_past)
            ref_out = ra_attention(q, k, v, attn_mask=(causal + decay), is_causal=False, scale=scale)
            grad_out = torch.randn_like(ref_out)
            dq_ref, dk_ref, dv_ref = torch.autograd.grad(ref_out, (q, k, v), grad_out)

            fused_out = ra_attention_fused_bias(
                q,
                k,
                v,
                distance_bias=db,
                hub_size=hub_size,
                q_offset=n_k - n_q,
                check_hub_q=(hub_size > 0 and is_dense and not has_past),
                is_causal=True,
                scale=scale,
            )
            dq_f, dk_f, dv_f = torch.autograd.grad(fused_out, (q, k, v), grad_out)
            # Triton tl.dot uses TF32 on Ampere+ GPUs when running float32 inputs.
            assert (fused_out - ref_out).abs().max().item() < 1e-2
            assert (dq_f - dq_ref).abs().max().item() < 3e-2
            assert (dk_f - dk_ref).abs().max().item() < 3e-2
            assert (dv_f - dv_ref).abs().max().item() < 3e-2


def test_selective_reversible_dkd_stack_and_compile_state_dict() -> None:
    from luminamoon.nn import RaMoE, ReversibleDKDStack, resolve_selective_reversible

    assert [resolve_selective_reversible(i, 8, selective_type="every", every=3) for i in range(8)] == [
        True, False, False, True, False, False, True, False
    ]
    assert [resolve_selective_reversible(i, 6, selective_type="even") for i in range(6)] == [
        True, False, True, False, True, False
    ]
    assert [resolve_selective_reversible(i, 6, selective_type="odd") for i in range(6)] == [
        False, True, False, True, False, True
    ]
    assert [resolve_selective_reversible(i, 6, selective_type="first_half") for i in range(6)] == [
        True, True, True, False, False, False
    ]
    assert [resolve_selective_reversible(i, 6, selective_type="second_half") for i in range(6)] == [
        False, False, False, True, True, True
    ]
    assert [resolve_selective_reversible(i, 5, selective_type="indices", selective_indices=[1, 4]) for i in range(5)] == [
        False, True, False, False, True
    ]
    assert resolve_selective_reversible(0, 4, selective_type="all") is True
    assert resolve_selective_reversible(0, 4, reversible=False, selective_type="none") is False

    with pytest.raises(ValueError, match="every must be > 0"):
        resolve_selective_reversible(0, 4, selective_type="every", every=0)
    with pytest.raises(ValueError, match="layer_idx must be"):
        resolve_selective_reversible(None, 4, selective_type="every", every=2)
    with pytest.raises(ValueError, match="selective_type must be one of"):
        resolve_selective_reversible(0, 4, selective_type="invalid")  # type: ignore[arg-type]

    torch.manual_seed(99)
    stack_sel = ReversibleDKDStack(
        num_layers=6,
        dim=32,
        num_heads=4,
        hub_size=2,
        selective_type="every",
        every=2,
        bias_mode="linear",
        drift_ffn_fn=lambda i: (
            RaMoE(16, 24, 16, num_experts=2, top_k=1, return_aux_loss=False) if (i + 1) % 3 == 0 else None
        ),
    )
    assert stack_sel.reversible_layer_indices == (0, 2, 4)

    stack_dense = ReversibleDKDStack(
        num_layers=6,
        dim=32,
        num_heads=4,
        hub_size=2,
        reversible=False,
        bias_mode="linear",
        drift_ffn_fn=lambda i: (
            RaMoE(16, 24, 16, num_experts=2, top_k=1, return_aux_loss=False) if (i + 1) % 3 == 0 else None
        ),
    )
    stack_dense.load_state_dict(stack_sel.state_dict())

    q_a = torch.randn(2, 10, 16, requires_grad=True)
    p_a = torch.randn(2, 10, 16, requires_grad=True)
    q_b = q_a.detach().clone().requires_grad_(True)
    p_b = p_a.detach().clone().requires_grad_(True)

    oq_a, op_a, _ = stack_sel(q_a, p_a)
    (oq_a.square().mean() + op_a.square().mean()).backward()
    oq_b, op_b, _ = stack_dense(q_b, p_b)
    (oq_b.square().mean() + op_b.square().mean()).backward()

    assert (oq_a - oq_b).abs().max().item() < 1e-5
    assert (op_a - op_b).abs().max().item() < 1e-5
    assert (q_a.grad - q_b.grad).abs().max().item() < 1e-4
    assert (p_a.grad - p_b.grad).abs().max().item() < 1e-4

    # In-place compilation preserves state_dict keys and numerical outputs.
    stack_compiled = ReversibleDKDStack(
        num_layers=4,
        dim=32,
        num_heads=4,
        hub_size=2,
        selective_type="every",
        every=2,
        compile_blocks=True,
        compile_kwargs={"backend": "eager"},
        bias_mode="linear",
    )
    stack_ref = ReversibleDKDStack(
        num_layers=4,
        dim=32,
        num_heads=4,
        hub_size=2,
        selective_type="every",
        every=2,
        compile_blocks=False,
        bias_mode="linear",
    )
    assert list(stack_compiled.state_dict().keys()) == list(stack_ref.state_dict().keys())
    stack_compiled.load_state_dict(stack_ref.state_dict())
    q_c = torch.randn(2, 8, 16, requires_grad=True)
    p_c = torch.randn(2, 8, 16, requires_grad=True)
    q_r = q_c.detach().clone().requires_grad_(True)
    p_r = p_c.detach().clone().requires_grad_(True)
    oqc, opc, _ = stack_compiled(q_c, p_c)
    (oqc.square().mean() + opc.square().mean()).backward()
    oqr, opr, _ = stack_ref(q_r, p_r)
    (oqr.square().mean() + opr.square().mean()).backward()
    assert (oqc - oqr).abs().max().item() < 1e-5
    assert (q_c.grad - q_r.grad).abs().max().item() < 1e-4


def test_compiled_ra_attention_bias_custom_op_parity() -> None:
    from luminamoon.nn._custom_ops import _HAS_CUSTOM_OP, compiled_ra_attention_bias
    from luminamoon.nn.attention import ra_attention

    if not _HAS_CUSTOM_OP:
        pytest.skip("torch.library.custom_op unavailable")

    torch.manual_seed(55)
    q = torch.randn(2, 4, 16, 16, requires_grad=True)
    k = torch.randn(2, 4, 16, 16, requires_grad=True)
    v = torch.randn(2, 4, 16, 16, requires_grad=True)
    db = DistanceDecayBias(num_heads=4, hub_size=2, mode="linear")
    lat = LatentKVAttention(dim=64, num_heads=4, hub_size=2, bias_mode="linear")
    scale = 0.25

    causal = lat._build_causal_mask(16, 16, q.device, q.dtype, True, False)
    decay = db(16, 16, q.device, q.dtype, is_dense_with_hub=True, has_past_kv=False)
    ref = ra_attention(q, k, v, attn_mask=(causal + decay), is_causal=False, scale=scale)
    g = torch.randn_like(ref)
    dq_r, dk_r, dv_r = torch.autograd.grad(ref, (q, k, v), g)

    slopes = db.slopes.reshape(-1)
    hpen = db.hub_penalty.reshape(-1)

    def _fn(qq, kk, vv):
        return compiled_ra_attention_bias(
            qq,
            kk,
            vv,
            bias_slopes=slopes,
            hub_penalty=hpen,
            bias_mode="linear",
            hub_size=2,
            q_offset=0,
            check_hub_q=True,
            is_causal=True,
            sm_scale=scale,
            ra_scale=0.0625,
        )

    compiled_fn = torch.compile(_fn, backend="eager", fullgraph=True)
    out = compiled_fn(q, k, v)
    dq_c, dk_c, dv_c = torch.autograd.grad(out, (q, k, v), g)
    assert (out - ref).abs().max().item() < 1e-5
    assert (dq_c - dq_r).abs().max().item() < 1e-4
    assert (dk_c - dk_r).abs().max().item() < 1e-4
    assert (dv_c - dv_r).abs().max().item() < 1e-4


def test_reversible_autocast_rng_kv_and_hub_cache_edge_cases() -> None:
    from luminamoon.nn import RaAttention, ReversibleDKDStack
    from luminamoon.nn._attention_dispatch import dispatch_attention

    # 1. ReversibleDKDBlock under CPU bfloat16 autocast matches non-reversible without dtype crash.
    torch.manual_seed(11)
    lin = torch.nn.Linear(16, 16)
    blk_r = ReversibleDKDBlock(dim=32, num_heads=4, hub_size=2, reversible=True, bias_mode="linear")
    blk_s = ReversibleDKDBlock(dim=32, num_heads=4, hub_size=2, reversible=False, bias_mode="linear")
    blk_s.load_state_dict(blk_r.state_dict())
    x = torch.randn(2, 8, 16)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        q1 = lin(x).detach().requires_grad_(True)
        p1 = lin(x).detach().requires_grad_(True)
        oq1, op1, _ = blk_r(q1, p1)
        loss1 = oq1.float().square().mean() + op1.float().square().mean()
    loss1.backward()

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        q2 = q1.detach().clone().requires_grad_(True)
        p2 = p1.detach().clone().requires_grad_(True)
        oq2, op2, _ = blk_s(q2, p2)
        loss2 = oq2.float().square().mean() + op2.float().square().mean()
    loss2.backward()
    assert q1.grad is not None and q2.grad is not None
    assert (q1.grad.float() - q2.grad.float()).abs().max().item() < 1e-3
    assert (p1.grad.float() - p2.grad.float()).abs().max().item() < 1e-3

    # 2. Stochastic kick_attn (dropout) stays consistent across reversible forward/backward recompute.
    torch.manual_seed(21)
    blk_dr = ReversibleDKDBlock(
        dim=32, num_heads=4, reversible=True, kick_attn=RaAttention(dim=16, num_heads=4, attn_drop=0.3)
    )
    blk_ds = ReversibleDKDBlock(
        dim=32, num_heads=4, reversible=False, kick_attn=RaAttention(dim=16, num_heads=4, attn_drop=0.3)
    )
    blk_ds.load_state_dict(blk_dr.state_dict())
    qa = torch.randn(2, 6, 16, requires_grad=True)
    pa = torch.randn(2, 6, 16, requires_grad=True)
    qb = qa.detach().clone().requires_grad_(True)
    pb = pa.detach().clone().requires_grad_(True)
    torch.manual_seed(777)
    oqa, opa, _ = blk_dr(qa, pa)
    (oqa.square().mean() + opa.square().mean()).backward()
    torch.manual_seed(777)
    oqb, opb, _ = blk_ds(qb, pb)
    (oqb.square().mean() + opb.square().mean()).backward()
    assert (qa.grad - qb.grad).abs().max().item() < 1e-4
    assert (pa.grad - pb.grad).abs().max().item() < 1e-4

    # 3. ReversibleDKDStack(..., return_past_kvs=True) populates KV caches during training.
    stk = ReversibleDKDStack(num_layers=2, dim=32, num_heads=4, selective_type="all")
    _, _, kvs = stk(torch.randn(1, 6, 16), torch.randn(1, 6, 16), return_past_kvs=True)
    assert kvs is not None and len(kvs) == 2 and all(kv is not None for kv in kvs)

    # 4. Constant custom drift_ffn (allow_unused None grad) does not crash reversible backward.
    class _ConstFFN(torch.nn.Module):
        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(z)

    blk_c = ReversibleDKDBlock(dim=32, num_heads=4, reversible=True, drift_ffn=_ConstFFN())
    qc = torch.randn(1, 4, 16, requires_grad=True)
    pc = torch.randn(1, 4, 16, requires_grad=True)
    oqc, opc, _ = blk_c(qc, pc)
    (oqc.sum() + opc.sum()).backward()
    assert qc.grad is not None and pc.grad is not None

    # 5. OrthogonalHub invalidates cache on assign=True and preserves autograd in eval mode.
    hub = OrthogonalHub(hub_size=8, dim=16).eval()
    with torch.no_grad():
        q_before = hub.get_orthogonal_matrix().clone()
    new_sd = {
        k: (v + torch.triu(torch.ones_like(v), diagonal=1) * 0.5) if k == "weight_skew" else v.clone()
        for k, v in hub.state_dict().items()
    }
    hub.load_state_dict(new_sd, assign=True)
    with torch.no_grad():
        q_after = hub.get_orthogonal_matrix()
    assert (q_after - q_before).abs().max().item() > 1e-2

    hub.zero_grad(set_to_none=True)
    _ = hub(2)
    out2 = hub(2)
    out2.sum().backward()
    assert hub.weight_skew.grad is not None

    # 6. QRoPE._build_cache does not double _cached_len when req <= _cached_len, and flex rejects attn_mask.
    qr = QRoPE(head_dim=8, max_seq_len=32)
    qr._build_cache(16, device=torch.device("cpu"))
    assert qr._cached_len == 32

    with pytest.raises(ValueError, match="does not accept an explicit attn_mask"):
        dispatch_attention(
            "flex", None, torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8),
            attn_mask=torch.zeros(1, 1, 4, 4), is_causal=False, scale=0.25,
        )


def test_whole_stack_compile_and_cudagraph_boundary_guard() -> None:
    from luminamoon.nn import RaMoE, ReversibleDKDStack
    from luminamoon.nn._reversible_ops import _is_cudagraph_compile_mode

    assert _is_cudagraph_compile_mode({"mode": "reduce-overhead"}) is True
    assert _is_cudagraph_compile_mode({"mode": "max-autotune"}) is True
    assert _is_cudagraph_compile_mode({"options": {"triton.cudagraphs": True}}) is True
    assert _is_cudagraph_compile_mode({"mode": "default"}) is False
    assert _is_cudagraph_compile_mode(None) is False

    torch.manual_seed(101)
    ref = ReversibleDKDStack(
        num_layers=4,
        dim=32,
        num_heads=4,
        hub_size=2,
        selective_type="none",
        reversible=False,
        compile_blocks=False,
        drift_ffn_fn=lambda i: RaMoE(16, 24, 16, num_experts=2, top_k=1, return_aux_loss=False) if i % 2 == 1 else None,
    )
    stk_whole = ReversibleDKDStack(
        num_layers=4,
        dim=32,
        num_heads=4,
        hub_size=2,
        selective_type="none",
        reversible=False,
        compile_blocks=True,
        compile_kwargs={"backend": "eager"},
        drift_ffn_fn=lambda i: RaMoE(16, 24, 16, num_experts=2, top_k=1, return_aux_loss=False) if i % 2 == 1 else None,
    )
    stk_whole.load_state_dict(ref.state_dict(), strict=True)
    assert stk_whole.whole_stack_compiled is True
    assert list(stk_whole.state_dict().keys()) == list(ref.state_dict().keys())

    # Multi-chunk forward (2 calls before 1 backward) must match the eager reference.
    q1, p1 = torch.randn(2, 8, 16, requires_grad=True), torch.randn(2, 8, 16, requires_grad=True)
    q2, p2 = torch.randn(2, 8, 16, requires_grad=True), torch.randn(2, 8, 16, requires_grad=True)
    oq1, op1, _ = stk_whole(q1, p1)
    oq2, op2, _ = stk_whole(q2, p2)
    (oq1.square().mean() + op1.square().mean() + oq2.square().mean() + op2.square().mean()).backward()

    q1r, p1r = q1.detach().clone().requires_grad_(True), p1.detach().clone().requires_grad_(True)
    q2r, p2r = q2.detach().clone().requires_grad_(True), p2.detach().clone().requires_grad_(True)
    rq1, rp1, _ = ref(q1r, p1r)
    rq2, rp2, _ = ref(q2r, p2r)
    (rq1.square().mean() + rp1.square().mean() + rq2.square().mean() + rp2.square().mean()).backward()
    assert (oq1 - rq1).abs().max().item() < 1e-5
    assert (oq2 - rq2).abs().max().item() < 1e-5
    assert (q1.grad - q1r.grad).abs().max().item() < 1e-4

    # Selective reversible stack keeps whole_stack_compiled=False and respects CUDA Graph guard flag.
    stk_sel = ReversibleDKDStack(
        num_layers=4,
        dim=32,
        num_heads=4,
        selective_type="every",
        every=2,
        compile_blocks=True,
        compile_kwargs={"backend": "eager", "options": {"triton.cudagraphs": True}},
    )
    assert stk_sel.whole_stack_compiled is False
    assert stk_sel.layers[0]._use_cudagraph_guard is True
    assert stk_sel.layers[1]._use_cudagraph_guard is True
    qs, ps = torch.randn(1, 6, 16, requires_grad=True), torch.randn(1, 6, 16, requires_grad=True)
    oqs, ops, _ = stk_sel(qs, ps)
    (oqs.square().mean() + ops.square().mean()).backward()
    assert qs.grad is not None and ps.grad is not None
