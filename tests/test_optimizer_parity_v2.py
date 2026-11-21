from __future__ import annotations

from pathlib import Path
import sys

ROOT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(ROOT_PYTHON) not in sys.path:
    sys.path.insert(0, str(ROOT_PYTHON))

import numpy as np
import pytest
import torch

import luminamoon.optimizer.luminav as lum_mod
from luminamoon import LuminaV
from luminamoon.optimizer.torch_ops import _prng_torch, _prng_weyl_torch, _sr_quantize, _tanh_pade_torch
from luminamoon.optimizer.validation import _prng_numpy, _prng_weyl_numpy, _quantize_numpy


def test_prng_logical_shift_parity_and_full_32bit_range():
    idx_np = np.arange(2048, dtype=np.int32)
    idx_th = torch.arange(2048, dtype=torch.int32)

    out_np = _prng_numpy(1337, idx_np)
    out_th = _prng_torch(1337, idx_th).numpy()
    np.testing.assert_array_equal(out_np, out_th)

    neg_count = int(np.sum(out_np < 0))
    assert 850 < neg_count < 1200, f"Expected ~50% negative i32 values (bit 31=1), got {neg_count}/2048"


def test_stochastic_rounding_preserves_nan_across_torch_and_numpy():
    for dt in (torch.bfloat16, torch.float16):
        target = torch.zeros(4, dtype=dt)
        vals = torch.tensor([float("nan"), 1.0, -2.5, float("nan")], dtype=torch.float32)
        _sr_quantize(target, vals, use_sr=True, seed=1337)
        assert torch.isnan(target[0]) and torch.isnan(target[3])
        assert torch.isfinite(target[1]) and torch.isfinite(target[2])

    np_fp16 = np.zeros(4, dtype=np.float16)
    np_vals = np.array([np.nan, 1.0, -2.5, np.nan], dtype=np.float32)
    _quantize_numpy(np_fp16, np_vals, use_sr=True, seed=1337)
    assert np.isnan(np_fp16[0]) and np.isnan(np_fp16[3])
    assert np.isfinite(np_fp16[1]) and np.isfinite(np_fp16[2])


def test_tanh_pade_matches_std_tanh():
    xs = torch.linspace(-10.0, 10.0, 4001, dtype=torch.float32)
    err = (_tanh_pade_torch(xs) - torch.tanh(xs)).abs().max().item()
    assert err < 3.0e-7


def test_extreme_gradient_zeroing_parity():
    g_raw = torch.tensor([2.0e38, -2.0e38, float("inf"), float("nan"), 0.5, -0.25], dtype=torch.float32)

    p_rust = torch.nn.Parameter(torch.ones(6, dtype=torch.float32))
    p_eager = torch.nn.Parameter(torch.ones(6, dtype=torch.float32))
    p_np = np.ones(6, dtype=np.float32)

    p_rust.grad = g_raw.clone()
    p_eager.grad = g_raw.clone()

    opt_rust = LuminaV([p_rust], lr=1e-2, buffer=2, execution="simd", stochastic_rounding=False)
    opt_eager = LuminaV([p_eager], lr=1e-2, buffer=2, execution="eager", stochastic_rounding=False)
    opt_rust.step()
    opt_eager.step()

    orig_rs = lum_mod._luminamoon_rs
    try:
        lum_mod._luminamoon_rs = None
        opt_np = LuminaV([p_np], lr=1e-2, buffer=2, stochastic_rounding=False)
        opt_np.step(grads=[g_raw.numpy().copy()])
    finally:
        lum_mod._luminamoon_rs = orig_rs

    torch.testing.assert_close(p_rust, p_eager, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(p_rust.detach().numpy(), p_np, atol=1e-6, rtol=1e-6)
    # Elements with |g| >= 1e38 must receive zero gradient (only weight decay applies)
    assert torch.allclose(opt_eager.state[p_eager]["exp_avg"][:4], torch.zeros(4))


@pytest.mark.parametrize("buf", [2, 1, "t1", "t2"])
@pytest.mark.parametrize("bound_type", ["coordinate", "radial"])
def test_multistep_parity_rust_vs_eager_vs_numpy(buf, bound_type):
    torch.manual_seed(2026)
    init = torch.randn(64, dtype=torch.bfloat16)
    grads = [torch.randn(64, dtype=torch.bfloat16) * 0.1 for _ in range(4)]

    p_rust = torch.nn.Parameter(init.clone())
    p_eager = torch.nn.Parameter(init.clone())
    opt_rust = LuminaV(
        [p_rust], lr=1e-2, buffer=buf, bound_type=bound_type, execution="simd", stochastic_rounding=True
    )
    opt_eager = LuminaV(
        [p_eager], lr=1e-2, buffer=buf, bound_type=bound_type, execution="eager", stochastic_rounding=True
    )

    for g in grads:
        p_rust.grad = g.clone()
        p_eager.grad = g.clone()
        opt_rust.step()
        opt_eager.step()

    torch.testing.assert_close(p_rust.float(), p_eager.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(opt_rust.state[p_rust]["scratch"], opt_eager.state[p_eager]["scratch"], atol=1e-4, rtol=1e-4)


def test_numpy_load_state_dict_resets_and_pins_scratch():
    p1 = np.ones(16, dtype=np.float32)
    opt1 = LuminaV([p1], lr=1e-2, buffer=2, cautious=True, bound_type="radial")
    opt1.step(grads=[np.ones(16, dtype=np.float32)])
    sd = opt1.state_dict()
    sd["state"][0]["scratch"] = np.array([9.0, 8.0, 7.0, 1.5, 0.75, 6.0], dtype=np.float32)
    sd["param_groups"][0]["cautious"] = False
    sd["param_groups"][0]["bound_type"] = "coordinate"

    p2 = np.ones(16, dtype=np.float32)
    opt2 = LuminaV([p2], lr=1e-2, buffer=2, cautious=False, bound_type="coordinate")
    opt2.load_state_dict(sd)
    sc = opt2.state[p2]["scratch"]
    assert sc[0] == 0.0 and sc[1] == 0.0 and sc[2] == 0.0 and sc[5] == 0.0
    assert sc[3] == 1.0 and sc[4] == 1.0


def test_triton_respects_deterministic_algorithms_mode():
    p = torch.nn.Parameter(torch.randn(16))
    p.grad = torch.randn(16)
    prev = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        opt = LuminaV([p], lr=1e-2, execution="triton")
        opt.step()
        assert opt.state[p]["step"] == 1
    finally:
        torch.use_deterministic_algorithms(prev)


def test_weyl_kronecker_3d_bit_exact_parity_torch_vs_numpy():
    idx_np = np.arange(1024, dtype=np.int32)
    idx_th = torch.arange(1024, dtype=torch.int32)
    for mult in (0x9E3779B9, 0x6A09E667, 0xBB67AE85):
        step_off = (17 * mult) & 0xFFFFFFFF
        w_np = _prng_weyl_numpy(1337, idx_np, step_off).view(np.int32)
        w_th = _prng_weyl_torch(1337, idx_th, step_off).numpy()
        np.testing.assert_array_equal(w_np, w_th)

    # Bit-exact FP16 SR parity with non-zero Weyl step offset across Torch and NumPy
    vals_np = np.linspace(-2.5, 2.5, 512, dtype=np.float32)
    out_np = np.zeros(512, dtype=np.float16)
    out_th = torch.zeros(512, dtype=torch.float16)
    step_off = (23 * 0x9E3779B9) & 0xFFFFFFFF
    _quantize_numpy(out_np, vals_np, use_sr=True, seed=1337, step_offset=step_off)
    _sr_quantize(out_th, torch.from_numpy(vals_np), use_sr=True, seed=1337, step_offset=step_off)
    np.testing.assert_array_equal(out_np.view(np.uint16), out_th.numpy().view(np.uint16))


@pytest.mark.parametrize("buf", ["t1", "t2"])
@pytest.mark.parametrize("bound_type", ["coordinate", "radial"])
def test_cubecl_triangle_family_parity_vs_eager(buf, bound_type):
    if not LuminaV.detect_hardware().get("cubecl_available", False):
        pytest.skip("CubeCL GPU backend unavailable")
    torch.manual_seed(2026)
    init = torch.randn(2048, dtype=torch.float32)
    grads = [torch.randn(2048, dtype=torch.float32) * 0.1 for _ in range(4)]

    p_gpu = torch.nn.Parameter(init.clone())
    p_eager = torch.nn.Parameter(init.clone())
    opt_gpu = LuminaV(
        [p_gpu], lr=1e-2, buffer=buf, bound_type=bound_type, execution="cubecl", stochastic_rounding=False
    )
    opt_eager = LuminaV(
        [p_eager], lr=1e-2, buffer=buf, bound_type=bound_type, execution="eager", stochastic_rounding=False
    )

    for g in grads:
        p_gpu.grad = g.clone()
        p_eager.grad = g.clone()
        opt_gpu.step()
        opt_eager.step()

    torch.testing.assert_close(p_gpu, p_eager, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(opt_gpu.state[p_gpu]["scratch"], opt_eager.state[p_eager]["scratch"], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("buf", ["t1", "t2", 1, 2])
@pytest.mark.parametrize("bound_type", ["coordinate", "radial"])
def test_fuse_tensors_zero_distortion_and_zero_grad_reclaim(buf, bound_type):
    torch.manual_seed(99)
    shapes = [(16,), (64, 32), (128, 64), (8,)]
    init_tensors = [torch.randn(s, dtype=torch.bfloat16) for s in shapes]
    step_grads = [
        [torch.randn(s, dtype=torch.bfloat16) * 0.2 for s in shapes[:3]] + [None]
        for _ in range(4)
    ]

    p_fused = [torch.nn.Parameter(t.clone()) for t in init_tensors]
    p_plain = [torch.nn.Parameter(t.clone()) for t in init_tensors]
    opt_fused = LuminaV(
        p_fused, lr=5e-3, buffer=buf, bound_type=bound_type, cautious=True,
        stochastic_rounding=True, fuse_tensors=True, execution="eager", seed=1337,
    )
    opt_plain = LuminaV(
        p_plain, lr=5e-3, buffer=buf, bound_type=bound_type, cautious=True,
        stochastic_rounding=True, fuse_tensors=False, execution="eager", seed=1337,
    )

    # Confirm flat 1D backing for params/states and zero permanent flat_g allocation.
    assert len(opt_fused._fused_buckets) == 1
    assert "flat_g" not in opt_fused._fused_buckets[0]
    assert opt_fused._fused_buckets[0]["flat_scratches"].shape == (4, 6)

    for grads_step in step_grads:
        for pf, pp, g in zip(p_fused, p_plain, grads_step):
            pf.grad = None if g is None else g.clone()
            pp.grad = None if g is None else g.clone()
        opt_fused.step()
        opt_plain.step()

    for pf, pp in zip(p_fused, p_plain):
        assert torch.equal(pf.detach(), pp.detach())
    # Parameter with grad=None must remain untouched with step=0.
    assert torch.equal(p_fused[-1].detach(), init_tensors[-1])
    assert opt_fused.state[p_fused[-1]]["step"] == 0

    opt_fused.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in p_fused)


def test_multi_tensor_triton_universal_ptx_compile():
    from luminamoon.optimizer.backends.triton import (
        HAS_TRITON,
        _lumina_multi_tensor_step_kernel,
        _lumina_post_step_ema_multi_kernel,
    )

    if not HAS_TRITON:
        pytest.skip("Triton compiler not available")

    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    sig = {
        "p_ptrs_arr": "*i64", "mp_ptrs_arr": "*i64", "g_ptrs_arr": "*i64",
        "m_ptrs_arr": "*i64", "v_ptrs_arr": "*i64", "sc_ptrs_arr": "*i64",
        "numels_arr": "*i64", "block_offs_arr": "*i64", "seeds_arr": "*i32",
        "num_tensors": "i32", "total_blocks": "i64",
        "beta1": "fp32", "beta2": "fp32", "c1": "fp32", "c2": "fp32",
        "lr": "fp32", "weight_decay": "fp32", "eps": "fp32", "lam": "fp32",
        "bound_ratio": "fp32", "step_i32": "i32",
    }
    # Universal compile matrix across Turing (sm_75), Ampere (sm_80), Ada (sm_89), and Hopper (sm_90).
    for sm, dt_mode in ((75, 2), (80, 1), (89, 1), (90, 1)):
        target = GPUTarget("cuda", sm, 32)
        for mode in (0, 1, 2, 3):
            cexpr = {
                "MODE": mode, "bound_mode": 1 if mode in (0, 2) else 2,
                "dtype_mode": dt_mode, "state_dtype_mode": dt_mode,
                "use_sr": True, "cautious": True, "has_master": False,
                "NEED_STATS": True, "BLOCK_SIZE": 2048,
            }
            cc = triton.compile(
                ASTSource(fn=_lumina_multi_tensor_step_kernel, signature=sig, constexprs=cexpr),
                target=target,
            )
            assert "ptx" in cc.asm and len(cc.asm["ptx"]) > 1000

    sig_ema = {
        "sc_ptrs_arr": "*i64", "numels_arr": "*i64", "lr": "fp32",
        "clamp_min": "fp32", "ema_decay": "fp32", "bound_ratio": "fp32",
    }
    for sm in (75, 80, 89, 90):
        cc_ema = triton.compile(
            ASTSource(
                fn=_lumina_post_step_ema_multi_kernel,
                signature=sig_ema,
                constexprs={"bound_mode": 1, "cautious": True, "is_triangle": True},
            ),
            target=GPUTarget("cuda", sm, 32),
        )
        assert "ptx" in cc_ema.asm and len(cc_ema.asm["ptx"]) > 200


@pytest.mark.parametrize("buf", ["t1", "t2", 1, 2])
@pytest.mark.parametrize("bound_type", ["coordinate", "radial"])
def test_cuda_multi_tensor_and_fuse_tensors_live_parity(buf, bound_type):
    from luminamoon.optimizer.backends.triton import HAS_TRITON

    if not (HAS_TRITON and torch.cuda.is_available()):
        pytest.skip("CUDA + Triton required")

    torch.manual_seed(77)
    shapes = [(64,), (256, 128), (512, 256), (32,)]
    dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    init_tensors = [torch.randn(s, device="cuda", dtype=dt) for s in shapes]
    step_grads = [
        [torch.randn(s, device="cuda", dtype=dt) * 0.1 for s in shapes[:3]] + [None]
        for _ in range(4)
    ]

    p_fused = [torch.nn.Parameter(t.clone()) for t in init_tensors]
    p_plain = [torch.nn.Parameter(t.clone()) for t in init_tensors]
    p_eager = [torch.nn.Parameter(t.clone()) for t in init_tensors]

    opt_fused = LuminaV(
        p_fused, lr=3e-3, buffer=buf, bound_type=bound_type, cautious=True,
        stochastic_rounding=True, fuse_tensors=True, execution="triton", seed=1337,
    )
    opt_plain = LuminaV(
        p_plain, lr=3e-3, buffer=buf, bound_type=bound_type, cautious=True,
        stochastic_rounding=True, fuse_tensors=False, execution="triton", seed=1337,
    )
    opt_eager = LuminaV(
        p_eager, lr=3e-3, buffer=buf, bound_type=bound_type, cautious=True,
        stochastic_rounding=True, fuse_tensors=False, execution="eager", seed=1337,
    )

    for grads_step in step_grads:
        for pf, pp, pe, g in zip(p_fused, p_plain, p_eager, grads_step):
            pf.grad = None if g is None else g.clone()
            pp.grad = None if g is None else g.clone()
            pe.grad = None if g is None else g.clone()
        opt_fused.step()
        opt_plain.step()
        opt_eager.step()

    for pf, pp, pe in zip(p_fused, p_plain, p_eager):
        assert torch.equal(pf.detach(), pp.detach())
        torch.testing.assert_close(pp.float(), pe.float(), atol=5e-2, rtol=5e-2)
    assert torch.equal(p_fused[-1].detach(), init_tensors[-1])
