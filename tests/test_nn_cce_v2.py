from __future__ import annotations

import math

import pytest
import torch

from luminamoon.nn import LinearCrossEntropy, linear_cross_entropy
from luminamoon.nn.kernels import can_use_triton_cce, triton_cce_bwd, triton_cce_fwd
from luminamoon.nn.kernels._base import HAS_TRITON


def _reference_cce_split_algebra(
    x: torch.Tensor,
    w: torch.Tensor,
    t: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    grad_loss: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Exact CPU FP64 mirror of CCE's algebraic target + label-smoothing + softmax decomposition."""
    M, _ = x.shape
    V = w.shape[0]
    valid = (t != ignore_index) & (t >= 0) & (t < V)
    safe_t = torch.where(valid, t, 0).to(torch.long)

    z = x @ w.t()
    if bias is not None:
        z = z + bias
    if logit_scale != 1.0:
        z = z * logit_scale

    lse = torch.where(valid, torch.logsumexp(z, dim=-1), 0.0)
    z_tgt = (x * w[safe_t]).sum(dim=-1)
    if bias is not None:
        z_tgt = z_tgt + bias[safe_t]
    z_tgt = torch.where(valid, z_tgt * logit_scale, 0.0)

    if label_smoothing > 0.0:
        z_mean = x @ w.mean(dim=0)
        if bias is not None:
            z_mean = z_mean + bias.mean()
        z_mean = z_mean * logit_scale
        losses = torch.where(valid, lse - (1.0 - label_smoothing) * z_tgt - label_smoothing * z_mean, 0.0)
    else:
        losses = torch.where(valid, lse - z_tgt, 0.0)

    g_scaled = torch.where(valid, grad_loss * logit_scale, 0.0)
    c_target = -(1.0 - label_smoothing) * g_scaled

    dx = w[safe_t] * c_target.unsqueeze(-1)
    dw = torch.zeros_like(w)
    dw.index_add_(0, safe_t, x * c_target.unsqueeze(-1))
    db = torch.zeros_like(bias) if bias is not None else None
    if db is not None:
        db.index_add_(0, safe_t, c_target)

    if label_smoothing > 0.0:
        u_smooth = -(label_smoothing / float(V)) * g_scaled
        dx = dx + u_smooth.unsqueeze(-1) * w.sum(dim=0, keepdim=True)
        dw = dw + (u_smooth.unsqueeze(-1) * x).sum(dim=0, keepdim=True)
        if db is not None:
            db = db + u_smooth.sum()

    z_shift = torch.where(valid.unsqueeze(-1), z - lse.unsqueeze(-1), float("-inf"))
    p_scaled = torch.exp(z_shift) * g_scaled.unsqueeze(-1)
    dx = dx + p_scaled @ w
    dw = dw + p_scaled.t() @ x
    if db is not None:
        db = db + p_scaled.sum(dim=0)
    return losses, dx, dw, db


class TestCutCrossEntropyV2:
    @pytest.mark.parametrize("has_bias", [False, True])
    @pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
    @pytest.mark.parametrize("logit_scale", [1.0, 1.75])
    def test_algebraic_decomposition_exact_fp64_parity(
        self, has_bias: bool, label_smoothing: float, logit_scale: float
    ) -> None:
        torch.manual_seed(42)
        M, D, V = 19, 13, 37
        x = torch.randn(M, D, dtype=torch.float64, requires_grad=True)
        w = torch.randn(V, D, dtype=torch.float64, requires_grad=True)
        b = torch.randn(V, dtype=torch.float64, requires_grad=True) if has_bias else None
        t = torch.randint(0, V, (M,), dtype=torch.long)
        t[0] = -100
        t[5] = -3
        t[11] = V + 7

        # 1. Exact FP64 parity against PyTorch autograd cross_entropy.
        logits_ref = torch.nn.functional.linear(x, w, b) * logit_scale
        valid_t = (t != -100) & (t >= 0) & (t < V)
        safe_t_ref = torch.where(valid_t, t, -100)
        loss_f64 = torch.nn.functional.cross_entropy(
            logits_ref, safe_t_ref, ignore_index=-100, reduction="none", label_smoothing=label_smoothing,
        )
        g_loss = torch.randn(M, dtype=torch.float64)
        loss_f64.backward(g_loss)

        losses_cce, dx_cce, dw_cce, db_cce = _reference_cce_split_algebra(
            x.detach(), w.detach(), t, b.detach() if b is not None else None,
            ignore_index=-100, label_smoothing=label_smoothing, logit_scale=logit_scale,
            grad_loss=g_loss,
        )
        assert torch.allclose(losses_cce, loss_f64.detach(), atol=1e-11, rtol=1e-11)
        assert torch.allclose(dx_cce, x.grad, atol=1e-11, rtol=1e-11)
        assert torch.allclose(dw_cce, w.grad, atol=1e-11, rtol=1e-11)
        if has_bias:
            assert db_cce is not None and b is not None and b.grad is not None
            assert torch.allclose(db_cce, b.grad, atol=1e-11, rtol=1e-11)

        # 2. Parity against chunked linear_cross_entropy (internal FP32 path on CPU).
        loss_chunk = linear_cross_entropy(
            x.detach(), w.detach(), t, bias=b.detach() if b is not None else None,
            ignore_index=-100, reduction="none", label_smoothing=label_smoothing,
            logit_scale=logit_scale, chunk_size=7,
        )
        assert torch.allclose(losses_cce, loss_chunk, atol=1e-5, rtol=1e-5)

    def test_all_ignored_targets_and_filter_threshold_safety(self) -> None:
        torch.manual_seed(7)
        M, D, V = 8, 16, 50304
        # Step-0 uniform distribution: p = 1/V must stay well above filter_eps = 2^-10 / V.
        log_filter_eps = math.log(1.0 / (1024.0 * float(V)))
        assert math.log(1.0 / float(V)) > log_filter_eps + 6.0

        # Invalid rows with extreme logits must produce exact zeros, never NaN.
        x = torch.full((M, D), 100.0, dtype=torch.float64)
        w = torch.full((32, D), 100.0, dtype=torch.float64)
        t = torch.full((M,), -100, dtype=torch.long)
        g = torch.ones(M, dtype=torch.float64)
        losses, dx, dw, _ = _reference_cce_split_algebra(x, w, t, None, grad_loss=g)
        assert torch.isfinite(losses).all() and torch.all(losses == 0.0)
        assert torch.isfinite(dx).all() and torch.all(dx == 0.0)
        assert torch.isfinite(dw).all() and torch.all(dw == 0.0)

    def test_can_use_triton_cce_guards_and_meta_dispatch(self) -> None:
        x_cpu = torch.randn(4, 64, dtype=torch.bfloat16)
        w_cpu = torch.randn(16384, 64, dtype=torch.bfloat16)
        assert not can_use_triton_cce(x_cpu, w_cpu, None)
        assert not can_use_triton_cce(x_cpu, w_cpu, 1024)

        m = LinearCrossEntropy(64, 128, bias=True, device="meta", dtype=torch.bfloat16)
        out = m(
            torch.empty(2, 8, 64, device="meta", dtype=torch.bfloat16),
            torch.empty(2, 8, device="meta", dtype=torch.long),
        )
        assert out.device.type == "meta" and out.shape == ()

    @pytest.mark.skipif(not HAS_TRITON, reason="Triton compiler not available")
    def test_cce_triton_kernels_ptx_sm89_compile(self) -> None:
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource

        import luminamoon.nn.kernels.loss_cce_bwd as mb
        import luminamoon.nn.kernels.loss_cce_fwd as mf

        # Accumulating backward kernels must reset output buffers during autotune benchmarking.
        assert list(mb._cce_bwd_joint_kernel.reset_to_zero) == ["dx_ptr", "dw_ptr", "db_ptr"]
        assert list(mb._cce_bwd_dx_kernel.reset_to_zero) == ["dx_ptr"]
        assert list(mb._cce_bwd_dw_kernel.reset_to_zero) == ["dw_ptr", "db_ptr"]

        target = GPUTarget("cuda", 89, 32)
        for dtype in ("*bf16", "*fp16"):
            src_t = ASTSource(
                fn=mf._cce_target_logit_kernel,
                signature={
                    "x_ptr": dtype, "w_ptr": dtype, "bias_ptr": "*fp32", "target_ptr": "*i64", "out_ptr": "*fp32",
                    "stride_xm": "i32", "stride_xd": "i32", "stride_wv": "i32", "stride_wd": "i32",
                    "M": "i32", "V": "i32", "D": "i32", "ignore_index": "i32", "logit_scale": "fp32",
                },
                constexprs={"HAS_BIAS": True, "HAS_LOGIT_SCALE": True, "BLOCK_D": 512},
            )
            assert "ptx" in triton.compile(src_t, target=target).asm

        for even_d in (True, False):
            src_f = ASTSource(
                fn=mf._cce_lse_fwd_kernel.fn,
                signature={
                    "x_ptr": "*bf16", "w_ptr": "*bf16", "bias_ptr": "*fp32", "lse_ptr": "*fp32",
                    "stride_xm": "i32", "stride_xd": "i32", "stride_wv": "i32", "stride_wd": "i32",
                    "stride_lse_s": "i32", "M": "i32", "V": "i32", "D": "i32",
                    "v_chunk_size": "i32", "logit_scale": "fp32",
                },
                constexprs={
                    "HAS_BIAS": True, "HAS_LOGIT_SCALE": True, "EVEN_D": even_d,
                    "BLOCK_M": 64, "BLOCK_V": 128, "BLOCK_D": 64,
                },
            )
            assert "ptx" in triton.compile(src_f, target=target).asm

        src_j = ASTSource(
            fn=mb._cce_bwd_joint_kernel.fn,
            signature={
                "x_ptr": "*bf16", "w_ptr": "*bf16", "bias_ptr": "*fp32", "lse_ptr": "*fp32", "g_scaled_ptr": "*fp32",
                "dx_ptr": "*fp32", "dw_ptr": "*fp32", "db_ptr": "*fp32",
                "stride_xm": "i32", "stride_xd": "i32", "stride_wv": "i32", "stride_wd": "i32",
                "stride_dxm": "i32", "stride_dxd": "i32", "stride_dwv": "i32", "stride_dwd": "i32",
                "M": "i32", "V": "i32", "D": "i32", "logit_scale": "fp32", "log_filter_eps": "fp32",
            },
            constexprs={
                "HAS_BIAS": True, "HAS_LOGIT_SCALE": True, "EVEN_D": True, "USE_FILTER": True,
                "NEED_DX": True, "NEED_DW": True, "NEED_DB": True,
                "BLOCK_M": 64, "BLOCK_V": 128, "BLOCK_D": 64, "GROUP_SIZE_M": 8,
            },
        )
        assert "ptx" in triton.compile(src_j, target=target).asm

        src_dx = ASTSource(
            fn=mb._cce_bwd_dx_kernel.fn,
            signature={
                "x_ptr": "*bf16", "w_ptr": "*bf16", "bias_ptr": "*fp32", "lse_ptr": "*fp32", "g_scaled_ptr": "*fp32",
                "dx_ptr": "*fp32",
                "stride_xm": "i32", "stride_xd": "i32", "stride_wv": "i32", "stride_wd": "i32",
                "stride_dxm": "i32", "stride_dxd": "i32", "M": "i32", "V": "i32", "D": "i32", "logit_scale": "fp32",
            },
            constexprs={
                "HAS_BIAS": True, "HAS_LOGIT_SCALE": True, "EVEN_D": False,
                "BLOCK_M": 64, "BLOCK_V": 128, "BLOCK_D": 64,
            },
        )
        assert "ptx" in triton.compile(src_dx, target=target).asm

        src_dw = ASTSource(
            fn=mb._cce_bwd_dw_kernel.fn,
            signature={
                "x_ptr": "*bf16", "w_ptr": "*bf16", "bias_ptr": "*fp32", "target_ptr": "*i64",
                "lse_ptr": "*fp32", "g_scaled_ptr": "*fp32", "c_target_ptr": "*fp32",
                "dw_ptr": "*fp32", "db_ptr": "*fp32",
                "stride_xm": "i32", "stride_xd": "i32", "stride_wv": "i32", "stride_wd": "i32",
                "stride_dwv": "i32", "stride_dwd": "i32", "M": "i32", "V": "i32", "D": "i32", "logit_scale": "fp32",
            },
            constexprs={
                "HAS_BIAS": True, "HAS_LOGIT_SCALE": True, "EVEN_D": False, "NEED_DW": True, "NEED_DB": True,
                "BLOCK_M": 64, "BLOCK_V": 128, "BLOCK_D": 64,
            },
        )
        assert "ptx" in triton.compile(src_dw, target=target).asm

    @pytest.mark.skipif(not (HAS_TRITON and torch.cuda.is_available()), reason="CUDA + Triton required")
    @pytest.mark.parametrize("deterministic", [False, True])
    @pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
    def test_cuda_triton_cce_live_parity(self, deterministic: bool, label_smoothing: float) -> None:
        torch.manual_seed(123)
        M, D, V = 137, 96, 523
        x = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(V, D, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(V, device="cuda", dtype=torch.bfloat16)
        t = torch.randint(0, V, (M,), device="cuda", dtype=torch.long)
        t[0] = -100
        t[17] = V + 2

        x_ref = x.detach().float().requires_grad_(True)
        w_ref = w.detach().float().requires_grad_(True)
        b_ref = b.detach().float().requires_grad_(True)
        loss_ref = linear_cross_entropy(
            x_ref, w_ref, t, bias=b_ref, ignore_index=-100, reduction="none",
            label_smoothing=label_smoothing, logit_scale=1.25, chunk_size=64,
        )
        g_loss = torch.full((M,), 1.0 / (M - 2), device="cuda", dtype=torch.float32)
        loss_ref.backward(g_loss)

        losses_cce, lse_cce = triton_cce_fwd(
            x, w, t, bias=b, ignore_index=-100, label_smoothing=label_smoothing, logit_scale=1.25,
        )
        dx_cce, dw_cce, db_cce = triton_cce_bwd(
            x, w, t, lse_cce, g_loss, bias=b, ignore_index=-100,
            label_smoothing=label_smoothing, logit_scale=1.25,
            need_dx=True, need_dw=True, need_db=True, deterministic=deterministic,
        )
        assert torch.allclose(losses_cce, loss_ref.detach(), atol=5e-2, rtol=5e-2)
        assert dx_cce is not None and torch.allclose(dx_cce.float(), x_ref.grad, atol=5e-2, rtol=5e-2)
        assert dw_cce is not None and torch.allclose(dw_cce.float(), w_ref.grad, atol=5e-2, rtol=5e-2)
        assert db_cce is not None and torch.allclose(db_cce.float(), b_ref.grad, atol=5e-2, rtol=5e-2)
