from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

ROOT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(ROOT_PYTHON) not in sys.path:
    if any((ROOT_PYTHON / "luminamoon" / "optimizer").glob("_luminamoon_optimizer_rs*")):
        sys.path.insert(0, str(ROOT_PYTHON))
    else:
        sys.path.append(str(ROOT_PYTHON))

import numpy as np
import pytest
import torch

import luminamoon.optimizer.backends.gpu as gpu_mod
import luminamoon.optimizer.backends.probe as probe_mod
import luminamoon.optimizer.luminav as lum_mod
from luminamoon import LuminaV
from luminamoon.optimizer.backends import (
    backend_unavailable_message,
    reset_autotune_cache,
    reset_probe_caches,
    triton_install_spec,
)


def test_triton_install_spec_matches_os_and_torch_minor() -> None:
    pkg_214, note_214 = triton_install_spec(platform="win32", torch_version="2.14.1+cu124")
    assert pkg_214 == "triton-windows<3.9"
    assert "triton-windows" in note_214 and "MSVC" in note_214

    pkg_215, _ = triton_install_spec(platform="win32", torch_version="2.15.0+cu128")
    assert pkg_215 == "triton-windows<3.10"

    pkg_win_def, _ = triton_install_spec(platform="win32", torch_version="")
    assert pkg_win_def == "triton-windows>=3.0.0"

    pkg_linux, note_linux = triton_install_spec(platform="linux", torch_version="2.14.1")
    assert pkg_linux == "triton>=3.0.0"
    assert "luminamoon[gpu]" in note_linux

    msg_triton = backend_unavailable_message("triton", reason="missing", platform="win32", torch_version="2.14.1")
    assert "triton-windows<3.9" in msg_triton and "luminamoon setup triton" in msg_triton

    msg_cubecl = backend_unavailable_message("cubecl", reason="not compiled")
    assert "maturin develop --release --features cubecl" in msg_cubecl and "execution='auto'" in msg_cubecl


def test_auto_dispatch_real_cpu_and_fallback_telemetry() -> None:
    reset_probe_caches()
    has_rs = lum_mod._luminamoon_rs is not None

    # 1. Small and large CPU PyTorch tensors with execution="auto" run tail-step benchmark first
    reset_autotune_cache()
    for n in (32, 20_000):
        p = torch.nn.Parameter(torch.ones(n, dtype=torch.float32))
        p.grad = torch.full((n,), 0.1, dtype=torch.float32)
        opt = LuminaV([p], lr=1e-2, buffer="t1", execution="auto")
        opt.step()
        assert opt.param_groups[0]["execution"] == "auto"
        assert torch.isfinite(p).all() and float(p.detach().mean().item()) < 1.0
        if has_rs:
            assert opt.benchmark_results is not None
            assert opt.resolved_execution in opt.benchmark_results["timings_ms"]
            assert opt.last_backend in ("simd", "triton_aot", "cubecl", "scalar", "eager")
        else:
            assert opt.last_backend == "eager"

    # 2. Fused CPU bucket also sets last_backend cleanly under execution="auto"
    p_f = torch.nn.Parameter(torch.ones(32, dtype=torch.float32))
    p_f.grad = torch.full((32,), 0.1, dtype=torch.float32)
    opt_f = LuminaV([p_f], lr=1e-2, execution="auto", fuse_tensors=True)
    opt_f.step()
    assert opt_f.last_backend in ("simd", "triton_aot", "cubecl", "scalar", "eager")

    # 3. Pure-Python fallback (native Rust extension disabled)
    orig_rs = lum_mod._luminamoon_rs
    try:
        lum_mod._luminamoon_rs = None
        p_th = torch.nn.Parameter(torch.ones(16, dtype=torch.float32))
        p_th.grad = torch.full((16,), 0.1, dtype=torch.float32)
        opt_th = LuminaV([p_th], lr=1e-2, execution="auto")
        opt_th.step()
        assert opt_th.last_backend == "eager"

        p_np = np.ones(16, dtype=np.float32)
        opt_np = LuminaV([p_np], lr=1e-2, execution="auto")
        opt_np.step(grads=[np.full(16, 0.1, dtype=np.float32)])
        assert opt_np.last_backend == "numpy"
    finally:
        lum_mod._luminamoon_rs = orig_rs


def test_auto_dispatch_gpu_probe_success_failure_and_midstep_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_probe_caches()
    p = torch.nn.Parameter(torch.ones(32, dtype=torch.float32))
    g = torch.full((32,), 0.1, dtype=torch.float32)
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)
    sc = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    opt = LuminaV([p], lr=1e-2, execution="auto")
    grp = opt.param_groups[0]

    # Simulate an accelerator device by monkeypatching device checks in gpu_mod.step_gpu_group
    class _FakeDev:
        type = "cuda"
        index = 0

    class _DevProxy(torch.Tensor):
        @property
        def device(self) -> Any:  # type: ignore[override]
            return _FakeDev()

    # Helper: run step_gpu_group with mocked accelerator check via a thin wrapper tensor
    def _make_fake_cuda_param() -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base = torch.ones(32, dtype=torch.float32)
        proxy = base.as_subclass(_DevProxy)
        return (
            proxy,
            torch.full((32,), 0.1, dtype=torch.float32),
            torch.zeros(32, dtype=torch.float32),
            torch.zeros(32, dtype=torch.float32),
            torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float32),
        )

    # Case 1: auto + Triton probe OK -> dispatches to Triton
    called = {"triton": 0}

    def _fake_triton_ok(*args: Any, **kwargs: Any) -> None:
        called["triton"] += 1

    monkeypatch.setattr(gpu_mod, "HAS_TRITON", True)
    monkeypatch.setattr(gpu_mod, "probe_triton_runtime", lambda dev=None, force=False: (True, "ok"))
    monkeypatch.setattr(gpu_mod, "dispatch_triton_group", _fake_triton_ok)
    p1, g1, m1, v1, sc1 = _make_fake_cuda_param()
    gpu_mod.step_gpu_group(opt, grp, [p1], [g1], [m1], [v1], [None], [sc1], [1], orig_indices=[0])
    assert called["triton"] == 1
    assert opt.last_backend == "triton"

    # Case 2: auto + Triton probe FAILS (e.g. broken MSVC on Windows) -> falls back to eager cleanly
    monkeypatch.setattr(gpu_mod, "probe_triton_runtime", lambda dev=None, force=False: (False, "cl.exe not found"))
    p2, g2, m2, v2, sc2 = _make_fake_cuda_param()
    eager_called = {"count": 0}
    orig_eager = gpu_mod.step_eager_group

    def _spy_eager(*args: Any, **kwargs: Any) -> None:
        eager_called["count"] += 1
        # Unwrap proxy to real CPU tensor so eager math executes cleanly
        args_l = list(args)
        args_l[2] = [t.as_subclass(torch.Tensor) for t in args_l[2]]
        orig_eager(*args_l, **kwargs)

    monkeypatch.setattr(gpu_mod, "step_eager_group", _spy_eager)
    gpu_mod.step_gpu_group(opt, grp, [p2], [g2], [m2], [v2], [None], [sc2], [1], orig_indices=[0])
    assert eager_called["count"] == 1
    assert opt.last_backend == "eager"
    assert float(p2.as_subclass(torch.Tensor).mean().item()) < 1.0

    # Case 3: auto + Triton probe OK BUT dispatch raises mid-step -> resets scratch & recovers to eager
    monkeypatch.setattr(gpu_mod, "probe_triton_runtime", lambda dev=None, force=False: (True, "ok"))
    monkeypatch.setattr(
        gpu_mod,
        "dispatch_triton_group",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("PTX launch failed")),
    )
    p3, g3, m3, v3, sc3 = _make_fake_cuda_param()
    sc3[0] = 99.0
    gpu_mod.step_gpu_group(opt, grp, [p3], [g3], [m3], [v3], [None], [sc3], [1], orig_indices=[0])
    assert eager_called["count"] == 2
    assert opt.last_backend == "eager"
    assert torch.isfinite(p3.as_subclass(torch.Tensor)).all()

    # Case 4: auto + deterministic_algorithms(True) -> routes to eager without calling Triton
    prev_det = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        monkeypatch.setattr(
            gpu_mod,
            "dispatch_triton_group",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Triton must not run in deterministic mode")),
        )
        p4, g4, m4, v4, sc4 = _make_fake_cuda_param()
        gpu_mod.step_gpu_group(opt, grp, [p4], [g4], [m4], [v4], [None], [sc4], [1], orig_indices=[0])
        assert opt.last_backend == "eager"
    finally:
        torch.use_deterministic_algorithms(prev_det)


def test_explicit_triton_and_cubecl_fail_fast_with_actionable_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_probe_caches()
    hw = LuminaV.detect_hardware()

    # 1. Explicit execution="cubecl" fails loudly when CubeCL is unavailable
    if not hw.get("cubecl_available", False):
        p = torch.nn.Parameter(torch.ones(16, dtype=torch.float32))
        p.grad = torch.full((16,), 0.1, dtype=torch.float32)
        opt_c = LuminaV([p], lr=1e-2, execution="cubecl")
        with pytest.raises(RuntimeError, match="maturin develop --release --features cubecl"):
            opt_c.step()

    # 2. Explicit execution="triton" on CPU without Triton AOT fails loudly with OS-aware install hint
    if not hw.get("triton_aot_available", False):
        p_t = torch.nn.Parameter(torch.ones(16, dtype=torch.float32))
        p_t.grad = torch.full((16,), 0.1, dtype=torch.float32)
        opt_t = LuminaV([p_t], lr=1e-2, execution="triton")
        expected_pkg = "triton-windows" if sys.platform.startswith("win") else "triton"
        with pytest.raises(RuntimeError, match=expected_pkg):
            opt_t.step()

    # 3. Explicit execution="triton" in step_gpu_group when HAS_TRITON=False fails loudly
    monkeypatch.setattr(gpu_mod, "HAS_TRITON", False)
    p_g = torch.nn.Parameter(torch.ones(16, dtype=torch.float32))
    opt_g = LuminaV([p_g], lr=1e-2, execution="triton")
    with pytest.raises(RuntimeError, match="luminamoon setup triton"):
        gpu_mod.step_gpu_group(
            opt_g, opt_g.param_groups[0], [p_g], [torch.ones(16)], [torch.zeros(16)],
            [torch.zeros(16)], [None], [torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])], [1],
        )

    # 4. Explicit execution="cubecl" routed to step_gpu_group fails loudly
    opt_cg = LuminaV([p_g], lr=1e-2, execution="cubecl")
    with pytest.raises(RuntimeError, match="execution='cubecl'"):
        gpu_mod.step_gpu_group(
            opt_cg, opt_cg.param_groups[0], [p_g], [torch.ones(16)], [torch.zeros(16)],
            [torch.zeros(16)], [None], [torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])], [1],
        )


def test_autotune_full_model_reset_budgeted_fallback_and_luminav_integration() -> None:
    from luminamoon import AutoTune as RootAutoTune
    from luminamoon.optimizer import AutoTune

    assert RootAutoTune is AutoTune
    reset_autotune_cache()
    torch.manual_seed(7)

    # Multi-layer PyTorch model (6 params -> eligible for fuse_tensors=False vs True comparison)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
    )
    orig_weights = [p.detach().clone() for p in model.parameters()]
    assert all(p.grad is None for p in model.parameters())

    # 1. Full-model AutoTune (6 steps, timing only last 2 steady-state steps)
    tune = AutoTune(
        model, steps=6, measure_last=2, strategy="full", buffer="t1", lr=1e-2, use_cache=False, verbose=False
    )
    assert tune.strategy == "full"
    assert tune.steps == 6 and tune.measure_last == 2
    assert tune.sampled_params == len(orig_weights)
    assert tune.execution in ("simd", "scalar", "eager", "triton", "cubecl")
    assert any("+fuse" in k for k in tune.timings_ms), f"Expected fused candidates in {list(tune.timings_ms)}"
    for cand, tail in tune.per_step_ms.items():
        assert len(tail) == 2 and all(t >= 0.0 for t in tail), f"Bad tail timings for {cand}: {tail}"
    assert "LuminaV AutoTune Summary" in tune.summary()
    assert "[fastest]" in tune.summary()

    # CRITICAL: All real model weights must be restored BIT-EXACTLY and grads back to None!
    for p, w_orig in zip(model.parameters(), orig_weights):
        assert torch.equal(p.detach(), w_orig), "AutoTune full strategy must restore weights bit-exactly"
        assert p.grad is None, "AutoTune full strategy must restore original None grads"

    # 2. Plug directly into LuminaV(model.parameters(), tuning=tune) -> overrides defaults while keeping execution='auto' portable
    opt = LuminaV(model.parameters(), tuning=tune)
    assert opt.tuning is tune
    assert opt.benchmark_results == tune.results
    assert opt.param_groups[0]["execution"] == "auto"
    assert opt.resolved_execution == tune.execution
    assert opt.param_groups[0]["buffer"] == "t1"
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-2)
    assert opt.param_groups[0]["fuse_tensors"] is tune.fuse_tensors
    assert opt.param_groups[0]["prewarm_jit"] is False

    # 3. Explicit user arguments take precedence over tuning overrides
    other_exec = "scalar" if tune.execution != "scalar" else "eager"
    opt_override = LuminaV(model.parameters(), lr=5e-4, execution=other_exec, tuning=tune)  # type: ignore[arg-type]
    assert opt_override.param_groups[0]["lr"] == pytest.approx(5e-4)
    assert opt_override.param_groups[0]["execution"] == other_exec

    # 4. Run a real training step using tuned optimizer
    for p in model.parameters():
        p.grad = torch.full_like(p, 0.05)
    opt.step()
    assert opt.last_backend is not None

    # 5. Budgeted strategy respects max_memory_mb and leaves real model untouched
    w_big = torch.nn.Parameter(torch.randn(256, 256))
    w_big_orig = w_big.detach().clone()
    tune_b = AutoTune(
        [w_big], steps=5, measure_last=1, strategy="budgeted", max_memory_mb=0.05, use_cache=False, verbose=False
    )
    assert tune_b.strategy == "budgeted"
    assert tune_b.measure_last == 1
    assert tune_b.sampled_bytes <= int(0.05 * 1024 * 1024) + 64
    assert torch.equal(w_big.detach(), w_big_orig)

    # 6. Validation errors
    with pytest.raises(ValueError, match="steps"):
        AutoTune([w_big], steps=2, measure_last=1, verbose=False)
    with pytest.raises(ValueError, match="measure_last"):
        AutoTune([w_big], steps=5, measure_last=5, verbose=False)
    with pytest.raises(TypeError, match="tuning"):
        LuminaV([w_big], tuning=123)

    # 7. Standalone NumPy AutoTune restores original array bit-exactly
    arr = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    arr_orig = arr.copy()
    tune_np = AutoTune([arr], steps=5, measure_last=2, strategy="full", buffer=1, use_cache=False, verbose=False)
    assert tune_np.strategy == "full"
    np.testing.assert_array_equal(arr, arr_orig)
    opt_np = LuminaV([arr], tuning=tune_np)
    opt_np.step(grads=[np.full_like(arr, 0.05)])
    assert opt_np.last_backend is not None


def test_cli_doctor_and_opt_in_setup_guards(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import subprocess
    from luminamoon import _cli_backends as cb
    from luminamoon.cli import main as cli_main

    # 1. luminamoon doctor & --benchmark
    rc = cli_main(["doctor", "--benchmark", "--steps", "5", "--measure-last", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "LUMINAMOON BACKEND DOCTOR" in out
    assert "Steady-State Backend Benchmark (5 steps, timing last 1):" in out
    assert "[fastest]" in out

    # 2. setup dry-run resolves triton-windows on Windows and triton on Linux without running pip
    def _forbid_run(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("subprocess.run must not execute during dry-run or declined setup")

    monkeypatch.setattr(subprocess, "run", _forbid_run)

    rc_win214 = cb.run_setup("triton", dry_run=True, platform_override="win32", torch_version_override="2.14.1+cu124")
    assert rc_win214 == 0
    assert "triton-windows<3.9" in capsys.readouterr().out

    rc_win215 = cb.run_setup("triton", dry_run=True, platform_override="win32", torch_version_override="2.15.0+cu128")
    assert rc_win215 == 0
    assert "triton-windows<3.10" in capsys.readouterr().out

    rc_lin = cb.run_setup("triton", dry_run=True, platform_override="linux", torch_version_override="2.14.1")
    assert rc_lin == 0
    assert "triton>=3.0.0" in capsys.readouterr().out

    rc_ccl = cli_main(["setup", "cubecl", "--dry-run"])
    assert rc_ccl == 0
    assert "maturin develop --release --features cubecl" in capsys.readouterr().out

    # 3. Declined prompt does not invoke pip
    rc_no = cb.run_setup("triton", yes=False, input_fn=lambda _: "n")
    assert rc_no == 0
    assert "cancelled" in capsys.readouterr().out.lower()

    # 4. Distributed multi-rank guard blocks setup inside torchrun
    monkeypatch.setenv("WORLD_SIZE", "4")
    rc_dist = cb.run_setup("triton", yes=True)
    assert rc_dist == 1
    assert "multi-rank" in capsys.readouterr().err.lower()


def test_new_backend_modules_respect_line_limits() -> None:
    for rel, lim in (
        ("luminamoon/optimizer/luminav.py", 450),
        ("luminamoon/optimizer/autotune.py", 450),
        ("luminamoon/optimizer/backends/gpu.py", 450),
        ("luminamoon/optimizer/backends/probe.py", 450),
        ("luminamoon/optimizer/backends/autotune.py", 450),
        ("luminamoon/_cli_backends.py", 450),
        ("luminamoon/cli.py", 450),
    ):
        n = len((ROOT_PYTHON / rel).read_text(encoding="utf-8").splitlines())
        assert n <= lim, f"{rel} has {n} lines (limit {lim})"
