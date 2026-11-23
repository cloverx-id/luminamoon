from __future__ import annotations

from pathlib import Path
import sys

ROOT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(ROOT_PYTHON) not in sys.path:
    if any((ROOT_PYTHON / "luminamoon" / "optimizer").glob("_luminamoon_optimizer_rs*")):
        sys.path.insert(0, str(ROOT_PYTHON))
    else:
        sys.path.append(str(ROOT_PYTHON))

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

    for ex in ("auto", "triton", "cubecl", "simd", "eager", "scalar"):
        opt = LuminaV([p], execution=ex)
        assert opt.param_groups[0]["execution"] == ex


def test_cubecl_execution_code_and_detect_hardware_fallback():
    from luminamoon.optimizer.config import build_rust_config

    p = torch.nn.Parameter(torch.randn(16))
    opt = LuminaV([p], execution="cubecl")
    cfg = build_rust_config(opt.param_groups[0])
    assert cfg["execution"] == 6

    orig_rs = lum_mod._luminamoon_rs
    try:
        lum_mod._luminamoon_rs = None
        hw = LuminaV.detect_hardware()
        assert hw["cubecl_available"] is False
        assert hw["cubecl_backend"] is None
    finally:
        lum_mod._luminamoon_rs = orig_rs


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

    for bad_ex in (0, 1, 2, 3, "foreach", "single", "vulkan", "scalar_core"):
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


def test_triton_optimizer_kernels_step1_specialization_and_ptx_compile():
    from luminamoon.optimizer.backends.triton import (
        HAS_TRITON,
        _lumina_dual_fused_single_pass_kernel,
        _lumina_dual_pass2_kernel,
        _lumina_dual_single_pass_kernel,
        _lumina_t1_single_pass_kernel,
        _lumina_t2_fused_single_pass_kernel,
        _lumina_t2_pass2_kernel,
        _lumina_t2_single_pass_kernel,
        _lumina_tdclm_pass2_kernel,
        _lumina_tdclm_single_pass_kernel,
    )

    if not HAS_TRITON:
        pytest.skip("Triton is not installed")

    weyl_kernels = [
        _lumina_tdclm_single_pass_kernel,
        _lumina_tdclm_pass2_kernel,
        _lumina_dual_fused_single_pass_kernel,
        _lumina_dual_pass2_kernel,
        _lumina_dual_single_pass_kernel,
        _lumina_t1_single_pass_kernel,
        _lumina_t2_fused_single_pass_kernel,
        _lumina_t2_single_pass_kernel,
        _lumina_t2_pass2_kernel,
    ]
    for k in weyl_kernels:
        assert getattr(k, "do_not_specialize", None) == ["seed", "step_i32"]

    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, compile as triton_compile

    target = GPUTarget("cuda", 80, 32)
    for fn, has_v, is_tri_flag in [
        (_lumina_tdclm_single_pass_kernel, False, False),
        (_lumina_t1_single_pass_kernel, False, False),
        (_lumina_dual_fused_single_pass_kernel, True, False),
        (_lumina_t2_fused_single_pass_kernel, True, False),
    ]:
        sig = {
            "p_ptr": "*bf16",
            "master_p_ptr": "*bf16",
            "grad_ptr": "*bf16",
            "exp_avg_ptr": "*bf16",
            "scratch_ptr": "*fp32",
            "n_elements": "i32",
            "beta1": "fp32",
            "lr": "fp32",
            "weight_decay": "fp32",
            "eps": "fp32",
            "bound_ratio": "fp32",
            "bound_floor": "fp32",
        }
        if has_v:
            sig["exp_avg_sq_ptr"] = "*bf16"
            sig["beta2"] = "fp32"
            sig["c1"] = "fp32"
            sig["c2"] = "fp32"
            sig["variance_temper"] = "fp32"
            sig["inv_sqrt_bc2_prev"] = "fp32"
        else:
            sig["lam"] = "fp32"
        cexpr = {
            "seed": 1,
            "step_i32": 1,
            "bound_mode": 1,
            "dtype_mode": 1,
            "state_dtype_mode": 1,
            "use_sr": True,
            "cautious": True,
            "has_master": False,
            "cautious_wd": True,
            "BLOCK_SIZE": 1024,
        }
        if has_v:
            cexpr["use_rtv"] = True
        if is_tri_flag:
            cexpr["is_triangle"] = True
        cc = triton_compile(ASTSource(fn=fn, signature=sig, constexprs=cexpr), target=target)
        assert "weyl" not in cc.asm["ptx"] or len(cc.asm["ptx"]) > 1000


def test_optimizer_seed_bounds_multigroup_grads_and_dynamic_unfreeze() -> None:
    import inspect
    torch = pytest.importorskip("torch")
    from luminamoon.optimizer import LuminaV
    from luminamoon.optimizer.backends.triton.multi_tensor import launch_multi_tensor_step

    # [L20] Seed upper bound matches u32 [0, 4294967295]
    p_seed = torch.nn.Parameter(torch.randn(8))
    with pytest.raises(ValueError, match="seed"):
        LuminaV([p_seed], seed=(1 << 32))
    opt_max = LuminaV([p_seed], seed=0xFFFFFFFF)
    assert opt_max.param_groups[0]["seed"] == 0xFFFFFFFF

    # [H7] Multi-group explicit grads=list maps globally across groups (fused & unfused)
    for fuse in (False, True):
        p0 = torch.nn.Parameter(torch.ones(16))
        p1 = torch.nn.Parameter(torch.ones(16))
        opt = LuminaV(
            [{"params": [p0], "lr": 1e-2}, {"params": [p1], "lr": 1e-2}],
            fuse_tensors=fuse,
        )
        g0 = torch.full_like(p0, 0.5)
        g1 = torch.full_like(p1, -0.5)
        opt.step(grads=[g0, g1])
        assert float(p0.detach().mean()) < 1.0
        assert float(p1.detach().mean()) > 1.0

    # [H6] Dynamic unfreezing post-__init__ with fuse_tensors=True preserves state and steps unfrozen param
    p_a = torch.nn.Parameter(torch.ones(16))
    p_b = torch.nn.Parameter(torch.ones(16), requires_grad=False)
    opt_u = LuminaV([p_a, p_b], lr=1e-2, fuse_tensors=True)
    p_a.grad = torch.full_like(p_a, 0.5)
    opt_u.step()
    assert opt_u.state[p_a]["step"] == 1
    assert torch.allclose(p_b.detach(), torch.ones(16))

    p_b.requires_grad_(True)
    p_a.grad = torch.full_like(p_a, 0.5)
    p_b.grad = torch.full_like(p_b, 0.5)
    opt_u.step()
    assert opt_u.state[p_a]["step"] == 2
    assert opt_u.state[p_b]["step"] == 1
    assert float(p_b.detach().mean()) < 1.0

    # [C1] & [H8] Verify launch_multi_tensor_step exposes on_step_committed and dispatcher guards mixed dtype
    sig = inspect.signature(launch_multi_tensor_step)
    assert "on_step_committed" in sig.parameters
    import luminamoon.optimizer.backends.triton.dispatcher as disp_mod

    disp_src = inspect.getsource(disp_mod._step_triton_single_device)
    assert "g_i.dtype == p.dtype" in disp_src
    assert "if step_committed:" in disp_src


def test_removed_v1_api_migration_messages() -> None:
    import luminamoon as root_pkg
    import luminamoon.optimizer as opt_pkg
    import luminamoon.optimizer.version as ver_mod

    p = torch.nn.Parameter(torch.randn(8))

    with pytest.raises(TypeError, match="MATH_VERSION"):
        LuminaV([p], version="1.5.5")
    with pytest.raises(TypeError, match="MATH_VERSION"):
        LuminaV([p], spec="1.6.0")
    with pytest.raises(TypeError, match="centered_defect"):
        LuminaV([p], centered_defect=True)
    with pytest.raises(TypeError, match="lam"):
        LuminaV([p], alpha_ss=0.5)
    with pytest.raises(TypeError, match="seed"):
        LuminaV([p], deterministic=True)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        LuminaV([p], unknown_kwarg=123)

    with pytest.raises(ValueError, match="MATH_VERSION"):
        LuminaV([{"params": [p], "version": "1.5.5"}])
    with pytest.raises(ValueError, match="MATH_VERSION"):
        LuminaV([{"params": [p], "spec": "1.6.0"}])

    opt = LuminaV([p])
    for attr, expected in (
        ("ALGORITHM_MATH", "MATH_VERSION"),
        ("ALGORITHM_SPEC", "MATH_VERSION"),
        ("version", "MATH_VERSION"),
        ("spec", "MATH_VERSION"),
        ("math_version", "MATH_VERSION"),
        ("get_version", "get_math_version"),
        ("getVersion", "get_math_version"),
        ("get_spec", "get_math_version"),
        ("getSpec", "get_math_version"),
        ("getMathVersion", "get_math_version"),
        ("set_version", "Multi-version switching"),
        ("setVersion", "Multi-version switching"),
        ("set_spec", "Multi-version switching"),
        ("setSpec", "Multi-version switching"),
        ("list_versions", "Multi-version switching"),
        ("listVersions", "Multi-version switching"),
        ("get_capabilities", "detect_hardware"),
    ):
        with pytest.raises(AttributeError, match=expected):
            getattr(LuminaV, attr)
        with pytest.raises(AttributeError, match=expected):
            getattr(opt, attr)
        if attr != "version":
            with pytest.raises(AttributeError, match=expected):
                getattr(opt_pkg, attr)
    assert opt_pkg.version is ver_mod

    for submod, expected in (
        ("gpu_backend", "backends.gpu"),
        ("fsdp_sync", "backends.fsdp"),
        ("numpy_backend", "backends.numpy_backend"),
        ("triton_kernels", "backends.triton"),
    ):
        with pytest.raises(AttributeError, match=expected):
            getattr(opt_pkg, submod)

    for vattr, expected in (
        ("DEFAULT_VERSION", "MATH_VERSION"),
        ("DEFAULT_LUMINAV_VERSION", "MATH_VERSION"),
        ("ALGORITHM_SPEC_VERSION", "MATH_VERSION"),
        ("SUPPORTED_VERSIONS", "Multi-version switching"),
        ("VERSION_CAPABILITIES", "detect_hardware"),
        ("ModularVersionResolver", "Multi-version switching"),
        ("get_active_optimizer_class", "luminamoon.optimizer"),
        ("get_capabilities", "detect_hardware"),
        ("get_version", "get_math_version"),
        ("list_versions", "Multi-version switching"),
        ("set_version", "Multi-version switching"),
        ("validate_version", "Multi-version switching"),
    ):
        with pytest.raises(AttributeError, match=expected):
            getattr(ver_mod, vattr)

    for top_attr, expected in (
        ("list_versions", "get_math_version"),
        ("set_version", "MATH_VERSION"),
    ):
        assert hasattr(root_pkg, top_attr) is False
        with pytest.raises(AttributeError, match=expected):
            getattr(root_pkg, top_attr)

    for kwargs, expected in (
        ({"buffer": "dual"}, "buffer=2"),
        ({"buffer": "td-clm"}, "buffer=1"),
        ({"buffer": "td-t1"}, "buffer='t1'"),
        ({"buffer": "td-t2"}, "buffer='t2'"),
        ({"buffer": "1"}, "buffer=1"),
        ({"bound_type": "none"}, "bound=False"),
        ({"bound_type": 1}, "bound_type='radial'"),
        ({"bound_type": 2}, "bound_type='coordinate'"),
        ({"master_weights": False}, "master_weights='none'"),
        ({"master_weights": True}, "master_weights='full'"),
        ({"master_weights": "fp32"}, "master_weights='full'"),
        ({"master_weights": "hybrid"}, "master_weights='semi'"),
        ({"execution": "foreach"}, "execution='auto'"),
        ({"execution": "single"}, "execution='eager'"),
        ({"execution": "scalar_core"}, "execution='scalar'"),
        ({"execution": "vulkan"}, "execution='cubecl'"),
        ({"execution": 3}, "execution='cubecl'"),
        ({"bound": 1}, "bound=True"),
    ):
        with pytest.raises(ValueError, match=expected):
            LuminaV([p], **kwargs)  # type: ignore[arg-type]


def test_v2_new_knobs_defaults_and_validation() -> None:
    from luminamoon.optimizer import VarianceType

    assert VarianceType is not None
    p = torch.nn.Parameter(torch.randn(8))
    opt = LuminaV([p])
    grp = opt.param_groups[0]
    assert grp["variance_type"] == "standard"
    assert grp["variance_temper"] == pytest.approx(3.0)
    assert grp["cautious_weight_decay"] is False
    assert grp["bound_floor"] == pytest.approx(1e-2)

    opt_rtv = LuminaV([p], variance_type="RTV", variance_temper=4.0, cautious_weight_decay=True, bound_floor=0.1)
    grp2 = opt_rtv.param_groups[0]
    assert grp2["variance_type"] == "rtv"
    assert grp2["variance_temper"] == pytest.approx(4.0)
    assert grp2["cautious_weight_decay"] is True
    assert grp2["bound_floor"] == pytest.approx(0.1)

    with pytest.raises(ValueError, match="variance_type"):
        LuminaV([p], variance_type="bad_mode")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="variance_temper"):
        LuminaV([p], variance_temper=0.0)
    with pytest.raises(ValueError, match="cautious_weight_decay"):
        LuminaV([p], cautious_weight_decay=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bound_floor"):
        LuminaV([p], bound_floor=0.0)
    with pytest.raises(ValueError, match="bound_floor"):
        LuminaV([p], bound_floor=-1e-2)

    rs_ext = lum_mod._luminamoon_rs

    if rs_ext is not None:
        p_raw = torch.ones(16, dtype=torch.float32)
        g_raw = torch.full((16,), 0.1, dtype=torch.float32)
        m_raw = torch.zeros(16, dtype=torch.float32)
        v_raw = torch.zeros(16, dtype=torch.float32)
        sc_raw = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
        rs_ext.step_pyo3(
            [p_raw], [g_raw], [m_raw], [v_raw], [None], [sc_raw],
            [1], [0], {"lr": 1e-2, "buffer": 2, "ema_decay": "auto"},
        )
        assert float(p_raw.mean().item()) < 1.0


def test_luminamoon_cli_and_help_module(tmp_path) -> None:
    import luminamoon as lm
    from luminamoon.cli import format_help, main as cli_main

    full_text = lm.help(print_out=False)
    assert "====" not in full_text and "----" not in full_text
    assert "___" in full_text
    assert "cautious_weight_decay" in full_text
    assert "variance_type" in full_text
    assert "variance_temper" in full_text
    assert "bound_floor" in full_text
    assert "RaNorm" in full_text
    assert "FusedPipeline" in full_text

    out_file = tmp_path / "output.txt"
    rc = cli_main(["--help", f"--output={out_file}", "--quiet"])
    assert rc == 0
    assert out_file.exists()
    saved = out_file.read_text(encoding="utf-8")
    assert "LUMINAMOON" in saved and "bound_floor" in saved

    opt_file = tmp_path / "opt_help.txt"
    opt_text = LuminaV.help(output=opt_file, print_out=False)
    assert opt_file.exists() and "Master Weights Matrix" in opt_text
    assert '"vulkan"' not in opt_text
    assert "vulkan_available" not in LuminaV.detect_hardware()

    with pytest.raises(ValueError, match="Invalid help topic"):
        format_help("invalid_topic")



