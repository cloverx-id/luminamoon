from __future__ import annotations

from pathlib import Path
import sys

ROOT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(ROOT_PYTHON) not in sys.path:
    sys.path.insert(0, str(ROOT_PYTHON))

import numpy as np
import pytest
import torch

from luminamoon import AutoTune, LuminaV
from luminamoon.optimizer import MomentumType
from luminamoon.optimizer.backends.numpy_backend import _apply_rpm_numpy
from luminamoon.optimizer.torch_ops import _apply_rpm_torch


def test_z3_formal_rpm_invariants():
    z3 = pytest.importorskip("z3")

    m0, m1, g0, g1, beta1, eps = z3.Reals("m0 m1 g0 g1 beta1 eps")
    pre = z3.And(beta1 >= 0, beta1 < 1, eps > 0)

    c = m0 * g1 - m1 * g0
    d = m0 * g0 + m1 * g1
    abs_d = z3.If(d >= 0, d, -d)
    abs_c = z3.If(c >= 0, c, -c)
    denom = 2 * abs_d + abs_c + eps
    s = (1 - beta1) * c / denom
    s2 = s * s
    c_rot = (1 - s2) / (1 + s2)
    s_rot = (2 * s) / (1 + s2)
    m0_rot = m0 * c_rot - m1 * s_rot
    m1_rot = m0 * s_rot + m1 * c_rot

    def _prove(prop, label: str) -> None:
        solver = z3.Solver()
        solver.add(pre)
        solver.add(z3.Not(prop))
        assert solver.check() == z3.unsat, f"Z3 counterexample for {label}: {solver.model()}"

    # 1. Bounded Cayley parameter: |s| < 1 - beta1 <= 1.
    abs_s = z3.If(s >= 0, s, -s)
    _prove(abs_s < (1 - beta1), "|s| < 1 - beta1")

    # 2. Exact orthogonality: c_rot^2 + s_rot^2 == 1.
    _prove(c_rot * c_rot + s_rot * s_rot == 1, "unit-circle orthogonality")

    # 3. Zero-gain norm preservation: ||m_rot||^2 == ||m||^2.
    _prove(m0_rot * m0_rot + m1_rot * m1_rot == m0 * m0 + m1 * m1, "norm preservation")

    # 4. Exact identity when m and g are collinear (c == 0).
    _prove(z3.Implies(c == 0, z3.And(m0_rot == m0, m1_rot == m1)), "collinear identity")

    # 5. Unconditional strict alignment improvement whenever c != 0:
    #    (a) Exact algebraic identity: dot_after - dot_before == 2*s*(c - s*d) / (1 + s^2)
    #    (b) Strict positivity: c != 0 => s*(c - s*d) > 0 for all real d.
    dot_before = m0 * g0 + m1 * g1
    dot_after = m0_rot * g0 + m1_rot * g1
    _prove(
        (dot_after - dot_before) * (1 + s2) == 2 * s * (c - s * d),
        "alignment delta identity",
    )
    c_sym, d_sym = z3.Reals("c_sym d_sym")
    abs_d_s = z3.If(d_sym >= 0, d_sym, -d_sym)
    abs_c_s = z3.If(c_sym >= 0, c_sym, -c_sym)
    s_sym = (1 - beta1) * c_sym / (2 * abs_d_s + abs_c_s + eps)
    solver_align = z3.Solver()
    solver_align.add(pre, c_sym != 0, z3.Not(s_sym * (c_sym - s_sym * d_sym) > 0))
    assert solver_align.check() == z3.unsat, f"Z3 counterexample for strict alignment: {solver_align.model()}"


def test_rpm_config_validation_and_odd_tail_identity():
    p = torch.nn.Parameter(torch.randn(5))
    opt = LuminaV([p], momentum_type="RPM")
    assert opt.param_groups[0]["momentum_type"] == "rpm"
    assert MomentumType is not None

    with pytest.raises(ValueError, match="momentum_type"):
        LuminaV([p], momentum_type="invalid_mode")  # type: ignore[arg-type]

    # Trailing odd coordinate must remain exact identity in both Torch and NumPy helpers.
    m = torch.tensor([1.0, 2.0, -0.5, 0.75, 3.25], dtype=torch.float32)
    g = torch.tensor([0.2, -0.4, 0.8, 0.1, -2.0], dtype=torch.float32)
    r_th = _apply_rpm_torch(m, g, 0.9, 1e-8)
    r_np = _apply_rpm_numpy(m.numpy(), g.numpy(), 0.9, 1e-8)
    assert float(r_th[-1].item()) == pytest.approx(3.25, abs=0.0)
    assert float(r_np[-1]) == pytest.approx(3.25, abs=0.0)
    np.testing.assert_allclose(r_th.numpy(), r_np, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("buf", ["t1", "t2", 1, 2])
@pytest.mark.parametrize("bound_type", ["coordinate", "radial"])
@pytest.mark.parametrize("numel", [1, 3, 17, 64, 129])
def test_rpm_cpu_cross_backend_parity(buf, bound_type, numel):
    torch.manual_seed(2026 + numel)
    init = torch.randn(numel, dtype=torch.float32) * 0.5
    grads = [torch.randn(numel, dtype=torch.float32) * 0.15 for _ in range(4)]

    common = dict(
        lr=3e-3,
        betas=(0.9, 0.999),
        weight_decay=0.02,
        cautious_weight_decay=True,
        variance_type="rtv",
        momentum_type="rpm",
        buffer=buf,
        bound_type=bound_type,
        cautious=True,
        stochastic_rounding=False,
        seed=1337,
    )

    p_eager = torch.nn.Parameter(init.clone())
    p_simd = torch.nn.Parameter(init.clone())
    p_scalar = torch.nn.Parameter(init.clone())
    p_np = init.numpy().copy()

    opt_eager = LuminaV([p_eager], execution="eager", **common)
    opt_simd = LuminaV([p_simd], execution="simd", **common)
    opt_scalar = LuminaV([p_scalar], execution="scalar", **common)
    opt_np = LuminaV([p_np], execution="eager", **common)

    for g in grads:
        p_eager.grad = g.clone()
        p_simd.grad = g.clone()
        p_scalar.grad = g.clone()
        opt_eager.step()
        opt_simd.step()
        opt_scalar.step()
        opt_np.step(grads=[g.numpy().copy()])

    torch.testing.assert_close(p_simd.detach(), p_eager.detach(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(p_scalar.detach(), p_eager.detach(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(p_eager.detach().numpy(), p_np, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("buf", ["t1", "t2", 1, 2])
def test_rpm_state_dict_fuse_tensors_and_autotune(buf):
    torch.manual_seed(77)
    shapes = [(15,), (33,), (64, 16)]
    init = [torch.randn(s, dtype=torch.bfloat16) for s in shapes]
    grads = [[torch.randn(s, dtype=torch.bfloat16) * 0.1 for s in shapes] for _ in range(3)]

    p_plain = [torch.nn.Parameter(t.clone()) for t in init]
    p_fused = [torch.nn.Parameter(t.clone()) for t in init]

    opt_plain = LuminaV(
        p_plain, lr=2e-3, buffer=buf, momentum_type="rpm",
        fuse_tensors=False, execution="eager", stochastic_rounding=True, seed=1337,
    )
    opt_fused = LuminaV(
        p_fused, lr=2e-3, buffer=buf, momentum_type="rpm",
        fuse_tensors=True, execution="eager", stochastic_rounding=True, seed=1337,
    )

    for g_step in grads:
        for pp, pf, g in zip(p_plain, p_fused, g_step):
            pp.grad = g.clone()
            pf.grad = g.clone()
        opt_plain.step()
        opt_fused.step()

    for pp, pf in zip(p_plain, p_fused):
        assert torch.equal(pp.detach(), pf.detach())

    sd = opt_plain.state_dict()
    assert sd["param_groups"][0]["momentum_type"] == "rpm"
    p_reload = [torch.nn.Parameter(t.clone()) for t in init]
    opt_reload = LuminaV(p_reload, lr=2e-3, buffer=buf, momentum_type="standard", execution="eager")
    opt_reload.load_state_dict(sd)
    assert opt_reload.param_groups[0]["momentum_type"] == "rpm"

    # AutoTune preserves explicit momentum_type="rpm" override and resolves auto execution safely.
    model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Linear(16, 4))
    tune = AutoTune(model)
    opt_tuned = LuminaV(model.parameters(), momentum_type="rpm", tuning=tune, execution="eager")
    assert opt_tuned.param_groups[0]["momentum_type"] == "rpm"

    # execution="auto" with momentum_type="rpm" must never select incompatible precompiled Triton AOT or poison cache.
    opt_auto = LuminaV(model.parameters(), momentum_type="rpm", tuning=tune, execution="auto")
    for p_m in model.parameters():
        p_m.grad = torch.randn_like(p_m) * 0.1
    opt_auto.step()
    assert opt_auto.last_backend is not None

    # Dynamic change of momentum_type must invalidate cached auto execution resolution.
    opt_auto.param_groups[0]["momentum_type"] = "standard"
    for p_m in model.parameters():
        p_m.grad = torch.randn_like(p_m) * 0.1
    opt_auto.step()
    assert len(opt_auto._resolved_auto_exec) >= 2


@pytest.mark.parametrize("buf", ["t1", "t2", 1, 2])
@pytest.mark.parametrize("bound_type", ["coordinate", "radial"])
def test_rpm_cuda_triton_single_and_multi_tensor_parity(buf, bound_type):
    from luminamoon.optimizer.backends.triton import HAS_TRITON

    if not (HAS_TRITON and torch.cuda.is_available()):
        pytest.skip("CUDA + Triton required")

    torch.manual_seed(2026)
    shapes = [(17,), (63,), (256, 33)]
    init = [torch.randn(s, device="cuda", dtype=torch.float32) * 0.5 for s in shapes]
    grads = [[torch.randn(s, device="cuda", dtype=torch.float32) * 0.1 for s in shapes] for _ in range(4)]

    p_eager = [torch.nn.Parameter(t.clone()) for t in init]
    p_triton = [torch.nn.Parameter(t.clone()) for t in init]
    p_fused = [torch.nn.Parameter(t.clone()) for t in init]

    common = dict(
        lr=3e-3,
        betas=(0.9, 0.999),
        weight_decay=0.02,
        cautious_weight_decay=True,
        variance_type="rtv",
        momentum_type="rpm",
        buffer=buf,
        bound_type=bound_type,
        cautious=True,
        stochastic_rounding=False,
        seed=1337,
    )
    opt_eager = LuminaV(p_eager, execution="eager", fuse_tensors=False, **common)
    opt_triton = LuminaV(p_triton, execution="triton", fuse_tensors=False, **common)
    opt_fused = LuminaV(p_fused, execution="triton", fuse_tensors=True, **common)

    for g_step in grads:
        for pe, pt, pf, g in zip(p_eager, p_triton, p_fused, g_step):
            pe.grad = g.clone()
            pt.grad = g.clone()
            pf.grad = g.clone()
        opt_eager.step()
        opt_triton.step()
        opt_fused.step()

    for pe, pt, pf in zip(p_eager, p_triton, p_fused):
        torch.testing.assert_close(pt.detach(), pe.detach(), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(pf.detach(), pe.detach(), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("buf", ["t1", "t2", 1, 2])
def test_rpm_cubecl_parity_vs_eager(buf):
    if not LuminaV.detect_hardware().get("cubecl_available", False):
        pytest.skip("CubeCL GPU backend unavailable")

    torch.manual_seed(2026)
    init = torch.randn(1025, dtype=torch.float32) * 0.5
    grads = [torch.randn(1025, dtype=torch.float32) * 0.1 for _ in range(4)]

    p_cube = torch.nn.Parameter(init.clone())
    p_eager = torch.nn.Parameter(init.clone())
    common = dict(
        lr=3e-3, buffer=buf, momentum_type="rpm", bound_type="coordinate",
        cautious=True, stochastic_rounding=False, seed=1337,
    )
    opt_cube = LuminaV([p_cube], execution="cubecl", **common)
    opt_eager = LuminaV([p_eager], execution="eager", **common)

    for g in grads:
        p_cube.grad = g.clone()
        p_eager.grad = g.clone()
        opt_cube.step()
        opt_eager.step()

    torch.testing.assert_close(p_cube.detach(), p_eager.detach(), atol=1e-5, rtol=1e-5)
