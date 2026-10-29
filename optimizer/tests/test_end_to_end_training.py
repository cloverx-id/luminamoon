from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import torch
import torch.nn as nn
from luminamoon import LuminaV


class SimpleMLP(nn.Module):
    def __init__(self, in_features: int = 64, hidden: int = 128, out_features: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_training_test(buffer_mode: int = 2, execution: str = "auto") -> None:
    mode_name = "Dual-Buffer CLM" if buffer_mode == 2 else "Single-Buffer TD-CLM"
    print(f"[Test] {mode_name} (buffer={buffer_mode}, execution={execution})")

    hw = LuminaV.detect_hardware()
    print(
        f"[Hardware] simd={hw.get('simd_backend')} vulkan={hw.get('vulkan_available')} device={hw.get('gpu_device_name')} wave32={hw.get('wave32_supported')}"
    )

    torch.manual_seed(42)

    model = SimpleMLP()
    criterion = nn.MSELoss()
    optimizer = LuminaV(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        buffer=buffer_mode,
        execution=execution,
        bound=True,
        cautious=True,
    )

    x = torch.randn(32, 64)
    target = torch.randn(32, 10)

    initial_weights = [p.clone().detach() for p in model.parameters()]
    losses = []

    for step in range(1, 11):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        print(f"Step {step:02d} | Loss: {loss_val:.6f}")

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: initial={losses[0]:.6f}, final={losses[-1]:.6f}"
    )

    for i, (init_w, cur_w) in enumerate(zip(initial_weights, model.parameters())):
        weight_diff = (cur_w - init_w).abs().max().item()
        assert weight_diff > 0.0, f"Parameter {i} was not updated (diff = 0.0)"

    pct_drop = ((losses[0] - losses[-1]) / losses[0]) * 100
    print(
        f"[Result] PASS (Loss: {losses[0]:.6f} to {losses[-1]:.6f}, Reduction: {pct_drop:.2f}%)"
    )


def run_error_validation_tests() -> None:
    p = [nn.Parameter(torch.randn(8, 8))]

    try:
        LuminaV(p, buffer=3)
        raise AssertionError("Expected ValueError for buffer=3")
    except ValueError as e:
        expected = "Invalid buffer count: 3. LuminaV only supports buffer=1 (Single-Buffer TD-CLM) or buffer=2 (Dual-Buffer CLM)."
        assert str(e) == expected, f"Unexpected buffer error: {e}"

    try:
        LuminaV(p, bound_type="radal")
        raise AssertionError("Expected ValueError for bound_type='radal'")
    except ValueError as e:
        expected = "Invalid bound_type: 'radal'. Expected 'coordinate' or 'radial'. Did you mean 'radial'?"
        assert str(e) == expected, f"Unexpected bound_type error: {e}"

    try:
        LuminaV(p, execution="vulkn")
        raise AssertionError("Expected ValueError for execution='vulkn'")
    except ValueError as e:
        expected = "Invalid execution mode: 'vulkn'. Expected 'auto', 'vulkan', 'simd', or 'scalar'. Did you mean 'vulkan'?"
        assert str(e) == expected, f"Unexpected execution error: {e}"

    try:
        LuminaV(p, master_weights="invalid_mw")
        raise AssertionError("Expected ValueError for master_weights='invalid_mw'")
    except ValueError as e:
        assert "Expected False, 'none', 'semi', or 'full'" in str(e), (
            f"Unexpected master_weights error: {e}"
        )

    try:
        LuminaV(p, lr=-0.01)
        raise AssertionError("Expected ValueError for lr=-0.01")
    except ValueError as e:
        assert "-0.01" in str(e) and "[0.0, inf)" in str(e), f"Unexpected lr error: {e}"

    try:
        LuminaV(p, betas=(1.2, 0.999))
        raise AssertionError("Expected ValueError for beta1=1.2")
    except ValueError as e:
        assert "1.2" in str(e) and "[0.0, 1.0)" in str(e), f"Unexpected beta1 error: {e}"

    try:
        LuminaV(p, eps=0.0)
        raise AssertionError("Expected ValueError for eps=0.0")
    except ValueError as e:
        assert "0.0" in str(e) and "(0.0, inf)" in str(e), f"Unexpected eps error: {e}"

    non_contig_p = nn.Parameter(torch.randn(8, 8).t())
    non_contig_p.grad = torch.randn(8, 8)
    opt_nc = LuminaV([non_contig_p])
    try:
        opt_nc.step()
        raise AssertionError("Expected ValueError for non-contiguous tensor")
    except ValueError as e:
        assert ".contiguous()" in str(e), f"Unexpected contiguity error: {e}"

    int_p = nn.Parameter(torch.zeros(8, 8, dtype=torch.float64), requires_grad=False)
    int_p.grad = torch.zeros(8, 8, dtype=torch.float64)
    opt_dt = LuminaV([int_p])
    try:
        opt_dt.step()
        raise AssertionError("Expected ValueError for float64 tensor")
    except ValueError as e:
        assert (
            "torch.float32" in str(e)
            and "torch.bfloat16" in str(e)
            and "torch.float16" in str(e)
        ), f"Unexpected dtype error: {e}"

    print("[Result] Error validation and typo suggestion tests PASS")


def run_mixed_precision_and_checkpoint_tests() -> None:
    torch.manual_seed(123)
    p_fp32 = nn.Parameter(torch.randn(16, 16, dtype=torch.float32))
    p_bf16 = nn.Parameter(torch.randn(16, 16, dtype=torch.bfloat16))
    p_fp16 = nn.Parameter(torch.randn(16, 16, dtype=torch.float16))
    p_empty = nn.Parameter(torch.empty(0, dtype=torch.float32))

    p_fp32.grad = torch.randn_like(p_fp32)
    p_bf16.grad = torch.randn_like(p_bf16)
    p_fp16.grad = torch.randn_like(p_fp16)
    p_empty.grad = torch.empty_like(p_empty)

    opt = LuminaV(
        [p_fp32, p_bf16, p_fp16, p_empty],
        lr=1e-2,
        master_weights=True,
        stochastic_rounding=False,
        execution="simd",
    )
    opt.step()

    assert "master_param" not in opt.state[p_fp32]
    assert "master_param" in opt.state[p_bf16]
    assert "master_param" in opt.state[p_fp16]

    saved = opt.state_dict()
    opt2 = LuminaV(
        [p_fp32, p_bf16, p_fp16, p_empty],
        lr=1e-2,
        master_weights="full",
        stochastic_rounding=True,
        execution="simd",
    )
    opt2.load_state_dict(saved)
    opt2.step()
    assert opt2.state[p_fp32]["step"] == 2
    assert opt2.state[p_bf16]["step"] == 2
    assert opt2.state[p_fp16]["step"] == 2
    print("[Result] Mixed-precision master_weights, FP16 RNE/SR, and state_dict tests PASS")


def run_pure_fp16_native_bf16_state_tests() -> None:
    import numpy as np
    from luminamoon.optimizer.torch_ops import _decode_state_f32

    hw = LuminaV.detect_hardware()
    exec_modes = ["auto", "simd", "scalar", "single"]
    if hw.get("vulkan_available", False):
        exec_modes.insert(1, "vulkan")

    torch.manual_seed(77)
    for exec_mode in exec_modes:
        for buf_mode in (1, 2):
            p = nn.Parameter(torch.full((256,), 0.5, dtype=torch.float16))
            p.grad = torch.full((256,), 1e-4, dtype=torch.float16)
            opt = LuminaV(
                [p],
                lr=1e-3,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01,
                buffer=buf_mode,
                master_weights=False,
                stochastic_rounding=False,
                execution=exec_mode,
            )
            for _ in range(5):
                opt.step()

            st = opt.state[p]
            exp_avg = st["exp_avg"]
            assert exp_avg.dtype == torch.float16, f"Expected float16 container, got {exp_avg.dtype}"
            decoded_m = _decode_state_f32(exp_avg)
            assert torch.all(decoded_m > 0.0), (
                f"Expected positive bitwise-BF16 exp_avg for exec={exec_mode}, buffer={buf_mode}"
            )
            if buf_mode == 2:
                exp_avg_sq = st["exp_avg_sq"]
                assert exp_avg_sq.dtype == torch.float16
                decoded_v = _decode_state_f32(exp_avg_sq)
                assert torch.all(decoded_v > 0.0), (
                    f"FP16 state underflow in exp_avg_sq for exec={exec_mode}: min={decoded_v.min().item()}"
                )
            assert torch.isfinite(p).all(), f"Non-finite parameter for exec={exec_mode}, buffer={buf_mode}"

    np_p = np.full((128,), 0.5, dtype=np.float16)
    np_g = np.full((128,), 1e-4, dtype=np.float16)
    opt_np = LuminaV(
        [np_p],
        lr=1e-3,
        buffer=2,
        master_weights=False,
        stochastic_rounding=False,
    )
    for _ in range(5):
        opt_np.step(grads=[np_g])
    np_st = opt_np.state[np_p]
    assert np_st["exp_avg_sq"].dtype == np.float16
    assert np.all(np_st["exp_avg_sq"].view(np.uint16) != 0), (
        "NumPy FP16 state underflowed to zero instead of preserving bitwise-BF16"
    )

    p_ckpt1 = nn.Parameter(torch.full((64,), 0.5, dtype=torch.float16))
    p_ckpt1.grad = torch.full((64,), 1e-4, dtype=torch.float16)
    opt_ckpt1 = LuminaV([p_ckpt1], lr=1e-3, buffer=2, master_weights=False, stochastic_rounding=False)
    for _ in range(3):
        opt_ckpt1.step()
    m_before = _decode_state_f32(opt_ckpt1.state[p_ckpt1]["exp_avg"]).clone()
    v_before = _decode_state_f32(opt_ckpt1.state[p_ckpt1]["exp_avg_sq"]).clone()
    sd = opt_ckpt1.state_dict()

    p_ckpt2 = nn.Parameter(p_ckpt1.detach().clone())
    p_ckpt2.grad = torch.full((64,), 1e-4, dtype=torch.float16)
    opt_ckpt2 = LuminaV([p_ckpt2], lr=1e-3, buffer=2, master_weights="full", stochastic_rounding=False)
    sd["param_groups"][0]["master_weights"] = "full"
    opt_ckpt2.load_state_dict(sd)
    assert opt_ckpt2.state[p_ckpt2]["exp_avg"].dtype == torch.float32
    assert torch.allclose(opt_ckpt2.state[p_ckpt2]["exp_avg"], m_before)
    assert torch.allclose(opt_ckpt2.state[p_ckpt2]["exp_avg_sq"], v_before)

    sd_full = opt_ckpt2.state_dict()
    sd_full["param_groups"][0]["master_weights"] = False
    p_ckpt3 = nn.Parameter(p_ckpt2.detach().clone())
    opt_ckpt3 = LuminaV([p_ckpt3], lr=1e-3, buffer=2, master_weights=False, stochastic_rounding=False)
    opt_ckpt3.load_state_dict(sd_full)
    assert opt_ckpt3.state[p_ckpt3]["exp_avg"].dtype == torch.float16
    assert torch.allclose(_decode_state_f32(opt_ckpt3.state[p_ckpt3]["exp_avg"]), m_before)
    assert torch.allclose(_decode_state_f32(opt_ckpt3.state[p_ckpt3]["exp_avg_sq"]), v_before)
    print("[Result] Pure FP16 Native Bitwise-BF16 state promotion tests PASS")


def run_contiguous_fusion_and_prewarm_tests() -> None:
    from luminamoon.optimizer.constants import TEMP_STATE_DICT_KEYS

    assert LuminaV.ALGORITHM_SPEC == "1.5.5"

    # 1. Default flags verification: fuse_tensors=False, prewarm_jit=True, bucket_cap_mb=500.0
    p_def = nn.Parameter(torch.randn(32))
    opt_def = LuminaV([p_def])
    assert opt_def.fuse_tensors is False
    assert opt_def.bucket_cap_mb == 500.0
    assert opt_def.prewarm_jit is True
    assert opt_def.param_groups[0]["fuse_tensors"] is False
    assert opt_def.param_groups[0]["bucket_cap_mb"] == 500.0
    assert opt_def.param_groups[0]["prewarm_jit"] is True
    assert len(opt_def._fused_buckets) == 0

    # Validation of invalid bucket_cap_mb
    try:
        LuminaV([p_def], bucket_cap_mb=0.0)
        raise AssertionError("Expected ValueError for bucket_cap_mb <= 0")
    except ValueError:
        pass
    try:
        LuminaV([p_def], bucket_cap_mb=-10.0)
        raise AssertionError("Expected ValueError for bucket_cap_mb < 0")
    except ValueError:
        pass

    # 1b. Warning when bucket_cap_mb is specified but fuse_tensors is False
    with warnings.catch_warnings(record=True) as recorded_w:
        warnings.simplefilter("always")
        LuminaV([p_def], bucket_cap_mb=250.0, fuse_tensors=False)
        assert any("Bounded bucket chunking only applies when" in str(w.message) for w in recorded_w), "Expected warning for bucket_cap_mb without fuse_tensors"

    with warnings.catch_warnings(record=True) as recorded_w2:
        warnings.simplefilter("always")
        LuminaV([p_def], fuse_tensors=False)
        assert not any("Bounded bucket chunking only applies when" in str(w.message) for w in recorded_w2), "Unexpected warning for default bucket_cap_mb"

    # 2. Validation of invalid boolean flags
    try:
        LuminaV([p_def], fuse_tensors="invalid")  # type: ignore[arg-type]
        raise AssertionError("Expected ValueError for non-boolean fuse_tensors")
    except ValueError:
        pass
    try:
        LuminaV([p_def], prewarm_jit="invalid")  # type: ignore[arg-type]
        raise AssertionError("Expected ValueError for non-boolean prewarm_jit")
    except ValueError:
        pass

    # 3. Contiguous bucket zero-copy shared storage & element-wise parity (buffer=1 & buffer=2)
    for buf_mode in (1, 2):
        for dt in (torch.float32, torch.float16, torch.bfloat16):
            torch.manual_seed(100 + buf_mode)
            shapes = [(16, 8), (16,), (32, 16), (32,), (8, 32)]
            params_unfused = [nn.Parameter(torch.randn(s, dtype=dt)) for s in shapes]
            params_fused = [nn.Parameter(p.detach().clone()) for p in params_unfused]

            opt_unfused = LuminaV(
                params_unfused,
                lr=2e-3,
                buffer=buf_mode,
                cautious=False,
                bound=True,
                bound_type="coordinate",
                stochastic_rounding=False,
                fuse_tensors=False,
                prewarm_jit=True,
            )
            opt_fused = LuminaV(
                params_fused,
                lr=2e-3,
                buffer=buf_mode,
                cautious=False,
                bound=True,
                bound_type="coordinate",
                stochastic_rounding=False,
                fuse_tensors=True,
                prewarm_jit=True,
            )

            assert len(opt_fused._fused_buckets) == 1
            bucket = opt_fused._fused_buckets[0]
            flat_p = bucket["flat_p"]
            flat_g = bucket["flat_g"]
            assert flat_p.numel() == sum(p.numel() for p in params_fused)

            # Verify zero_grad preserves p.grad view pointers into flat_g
            opt_fused.zero_grad(set_to_none=True)
            for p, _, g_slice, _, g_ptr, _, _ in bucket["slices"]:
                assert p.grad is g_slice and p.grad.data_ptr() == g_ptr

            for step_idx in range(4):
                opt_unfused.zero_grad(set_to_none=True)
                opt_fused.zero_grad(set_to_none=True)
                for pu, pf in zip(params_unfused, params_fused):
                    g = torch.randn_like(pu) * 0.05
                    pu.grad = g.clone()
                    assert pf.grad is not None
                    pf.grad.copy_(g)
                opt_unfused.step()
                opt_fused.step()

            for pu, pf in zip(params_unfused, params_fused):
                assert torch.allclose(pu, pf, atol=1e-5, rtol=1e-4), (
                    f"Mismatch between fused and unfused for buffer={buf_mode}, dtype={dt}"
                )

            # Verify LSO-Eval Stage II state_dict cleanliness & load_state_dict round-trip
            sd_fused = opt_fused.state_dict()
            for p_state in sd_fused["state"].values():
                for k in p_state:
                    assert k not in TEMP_STATE_DICT_KEYS, f"Leaked temporary key {k} in state_dict()"
                assert p_state["exp_avg"].shape in shapes

            # Mutate state and reload via load_state_dict
            params_reloaded = [nn.Parameter(p.detach().clone()) for p in params_fused]
            opt_reloaded = LuminaV(
                params_reloaded,
                lr=2e-3,
                buffer=buf_mode,
                cautious=False,
                bound=True,
                bound_type="coordinate",
                stochastic_rounding=False,
                fuse_tensors=True,
                prewarm_jit=True,
            )
            opt_reloaded.load_state_dict(sd_fused)
            rel_bucket = opt_reloaded._fused_buckets[0]
            for p, p_slice, _, p_ptr, _, st, _ in rel_bucket["slices"]:
                assert p.data_ptr() == p_ptr and p.data_ptr() == p_slice.data_ptr()
                assert opt_reloaded.state[p] is st

    # 4. Multi-chunk bounded bucket partitioning (500 MB logic with small cap test)
    shapes_multi = [(32, 16), (16,), (64, 32), (32,), (128, 64)]
    p_unfused_multi = [nn.Parameter(torch.randn(s)) for s in shapes_multi]
    p_chunked_multi = [nn.Parameter(p.detach().clone()) for p in p_unfused_multi]
    # Set cap to ~4 KB so each medium layer gets split into separate chunks
    opt_unfused_m = LuminaV(p_unfused_multi, lr=1e-3, buffer=1, fuse_tensors=False)
    opt_chunked_m = LuminaV(p_chunked_multi, lr=1e-3, buffer=1, fuse_tensors=True, bucket_cap_mb=0.004)
    assert len(opt_chunked_m._fused_buckets) >= 2, f"Expected >= 2 buckets, got {len(opt_chunked_m._fused_buckets)}"
    for _ in range(5):
        opt_unfused_m.zero_grad(set_to_none=True)
        opt_chunked_m.zero_grad(set_to_none=True)
        for pu, pc in zip(p_unfused_multi, p_chunked_multi):
            g = torch.randn_like(pu) * 0.01
            pu.grad = g.clone()
            pc.grad.copy_(g)
        opt_unfused_m.step()
        opt_chunked_m.step()
    for pu, pc in zip(p_unfused_multi, p_chunked_multi):
        assert torch.allclose(pu, pc, atol=1e-5, rtol=1e-4), "Mismatch in multi-chunk bounded execution"

    # State dict round-trip across multi-chunk buckets
    sd_chunked = opt_chunked_m.state_dict()
    p_reload_m = [nn.Parameter(p.detach().clone()) for p in p_chunked_multi]
    opt_reload_m = LuminaV(p_reload_m, lr=1e-3, buffer=1, fuse_tensors=True, bucket_cap_mb=0.004)
    opt_reload_m.load_state_dict(sd_chunked)
    assert len(opt_reload_m._fused_buckets) == len(opt_chunked_m._fused_buckets)

    print("[Result] Bounded bucket chunking (500 MB & sequential migration) tests PASS")
    print("[Result] Contiguous buffer fusion (fuse_tensors=True) & JIT pre-warmup tests PASS")


if __name__ == "__main__":
    run_training_test(buffer_mode=2, execution="auto")
    run_training_test(buffer_mode=1, execution="auto")
    run_error_validation_tests()
    run_mixed_precision_and_checkpoint_tests()
    run_pure_fp16_native_bf16_state_tests()
    run_contiguous_fusion_and_prewarm_tests()
    print("[Summary] All end-to-end training and validation tests completed.")
