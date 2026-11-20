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


def test_canonical_v2_enum_normalization():
    p = torch.nn.Parameter(torch.randn(16))

    for buf_in, buf_out in ((2, 2), (1, 1), ("t1", "t1"), ("T1", "t1"), ("t2", "t2"), ("T2", "t2")):
        opt = LuminaV([p], buffer=buf_in)  # type: ignore[arg-type]
        assert opt.param_groups[0]["buffer"] == buf_out

    for bound_flag in (True, False):
        for bt in ("coordinate", "radial"):
            opt = LuminaV([p], bound=bound_flag, bound_type=bt)
            assert opt.param_groups[0]["bound"] is bound_flag
            assert opt.param_groups[0]["bound_type"] == bt

    for mw in ("none", "semi", "full"):
        opt = LuminaV([p], master_weights=mw)
        assert opt.param_groups[0]["master_weights"] == mw

    for ex in ("auto", "triton", "vulkan", "simd", "eager", "scalar"):
        opt = LuminaV([p], execution=ex)
        assert opt.param_groups[0]["execution"] == ex


def test_rejects_v1_legacy_aliases_and_non_string_modes():
    p = torch.nn.Parameter(torch.randn(8))

    for bad_bound in (1, 0, "true"):
        with pytest.raises(ValueError):
            LuminaV([p], bound=bad_bound)  # type: ignore[arg-type]

    for bad_buf in ("dual", "single", "clm", "td-clm", "1", "2", True, 3):
        with pytest.raises(ValueError):
            LuminaV([p], buffer=bad_buf)  # type: ignore[arg-type]

    for bad_bt in (0, 1, 2, True, "1", "2", "none"):
        with pytest.raises(ValueError):
            LuminaV([p], bound_type=bad_bt)  # type: ignore[arg-type]

    for bad_mw in (False, True, 0, 1, 2, "false", "true", "fp32"):
        with pytest.raises(ValueError):
            LuminaV([p], master_weights=bad_mw)  # type: ignore[arg-type]

    for bad_ex in (0, 1, 2, "foreach", "single", "scalar_core"):
        with pytest.raises(ValueError):
            LuminaV([p], execution=bad_ex)  # type: ignore[arg-type]


def test_bound_false_and_eager_execution_step():
    torch.manual_seed(0)
    for buf in (2, 1, "t1", "t2"):
        p = torch.nn.Parameter(torch.randn(32))
        p.grad = torch.randn(32)
        p_init = p.detach().clone()
        opt = LuminaV([p], lr=1e-2, buffer=buf, bound=False, execution="eager")
        opt.step()
        opt.step()
        assert opt.state[p]["step"] == 2
        assert torch.isfinite(p).all()
        assert not torch.allclose(p, p_init)


def test_dynamic_fallback_without_native_extension():
    orig_rs = lum_mod._luminamoon_rs
    try:
        lum_mod._luminamoon_rs = None
        for buf in (2, 1, "t1", "t2"):
            p_np = np.ones(16, dtype=np.float32)
            opt_np = LuminaV([p_np], lr=1e-2, buffer=buf, bound=True, bound_type="radial")
            opt_np.step(grads=[np.ones(16, dtype=np.float32)])
            opt_np.step(grads=[np.ones(16, dtype=np.float32)])
            assert opt_np.state[p_np]["step"] == 2
            assert np.isfinite(p_np).all()

            p_th = torch.nn.Parameter(torch.ones(16, dtype=torch.float32))
            p_th.grad = torch.ones(16, dtype=torch.float32)
            opt_th = LuminaV([p_th], lr=1e-2, buffer=buf, bound=True, bound_type="radial")
            opt_th.step()
            opt_th.step()
            assert opt_th.state[p_th]["step"] == 2
            assert torch.isfinite(p_th).all()
    finally:
        lum_mod._luminamoon_rs = orig_rs


def test_optimizer_python_files_respect_line_budget():
    opt_root = ROOT_PYTHON / "luminamoon" / "optimizer"
    for py_file in sorted(opt_root.rglob("*.py")):
        lines = len(py_file.read_text(encoding="utf-8").splitlines())
        limit = 600 if "triton" in py_file.parts else 450
        assert lines <= limit, f"{py_file.relative_to(ROOT_PYTHON)} has {lines} lines (limit {limit})"


def test_optimizer_rust_files_respect_line_budget():
    crates_root = ROOT_PYTHON.parent / "optimizer" / "crates"
    kernel_dirs = {"avx2", "avx512", "neon", "reference_cpu"}
    for rs_file in sorted(crates_root.rglob("*.rs")):
        lines = len(rs_file.read_text(encoding="utf-8").splitlines())
        parts = set(rs_file.parts)
        if "tests" in parts:
            limit = 650
        elif (parts & kernel_dirs) or rs_file.name == "ra.rs":
            limit = 600
        else:
            limit = 450
        assert lines <= limit, f"{rs_file.relative_to(crates_root)} has {lines} lines (limit {limit})"


def test_empty_first_param_group_detects_torch_and_numpy_dict_grads():
    p_th = torch.nn.Parameter(torch.ones(8, dtype=torch.float32))
    p_th.grad = torch.ones(8, dtype=torch.float32)
    opt_th = LuminaV([{"params": []}, {"params": [p_th]}], lr=1e-2)
    assert opt_th._is_torch_mode is True
    opt_th.step()
    assert opt_th.state[p_th]["step"] == 1

    p_np = np.ones(8, dtype=np.float32)
    g_np = np.ones(8, dtype=np.float32)
    opt_np = LuminaV([p_np], lr=1e-2)
    opt_np.step(grads={id(p_np): g_np})
    assert opt_np.state[p_np]["step"] == 1

