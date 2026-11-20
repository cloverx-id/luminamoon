from __future__ import annotations

import inspect
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_PYTHON = ROOT_DIR / "python"
if str(ROOT_PYTHON) not in sys.path:
    sys.path.insert(0, str(ROOT_PYTHON))


def _run_py(script: str, *, pythonpath: Path = ROOT_PYTHON) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(pythonpath)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(ROOT_DIR),
        env=env,
    )


def test_root_import_is_lazy_in_fresh_subprocess() -> None:
    script_nn = (
        "import sys\n"
        "import luminamoon as lm\n"
        "assert isinstance(lm.__version__, str) and lm.get_version() == lm.__version__\n"
        "for mod in ('torch', 'luminamoon.optimizer', 'luminamoon.nn', 'luminamoon.data'):\n"
        "    assert mod not in sys.modules, f'Eagerly imported {mod}'\n"
        "assert 'LuminaV' in dir(lm) and 'nn' in dir(lm) and 'data' in dir(lm)\n"
        "_ = lm.nn\n"
        "assert 'luminamoon.nn' in sys.modules\n"
        "assert 'luminamoon.data' not in sys.modules\n"
        "print('LAZY_NN_OK')\n"
    )
    res_nn = _run_py(script_nn)
    assert res_nn.returncode == 0, f"Subprocess failed: {res_nn.stderr}"
    assert "LAZY_NN_OK" in res_nn.stdout

    script_data = (
        "import sys\n"
        "import luminamoon as lm\n"
        "_ = lm.data\n"
        "assert 'luminamoon.data' in sys.modules\n"
        "assert 'luminamoon.optimizer' not in sys.modules, 'lm.data imported luminamoon.optimizer'\n"
        "assert 'torch' not in sys.modules, 'lm.data imported torch'\n"
        "assert lm.data.clean_text('  hi  ') == 'hi'\n"
        "_ = lm.LuminaV\n"
        "assert 'luminamoon.optimizer' in sys.modules\n"
        "print('LAZY_DATA_OK')\n"
    )
    res_data = _run_py(script_data)
    assert res_data.returncode == 0, f"Subprocess failed: {res_data.stderr}"
    assert "LAZY_DATA_OK" in res_data.stdout


def test_pure_python_install_does_not_leak_foreign_native_binary() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        pure_root = Path(tmp_dir)
        dst_pkg = pure_root / "luminamoon"
        shutil.copytree(
            ROOT_PYTHON / "luminamoon",
            dst_pkg,
            ignore=shutil.ignore_patterns("*.pyd", "*.so", "*.dylib", "*.pdb", "__pycache__"),
        )
        script = (
            "import sys\n"
            "import luminamoon as lm\n"
            "from luminamoon.data import _ffi, fallbacks\n"
            "assert _ffi.StringTensor is fallbacks.StringTensor\n"
            "assert lm.data.data_cpu_isa() == 'SCALAR'\n"
            "assert 'torch' not in sys.modules\n"
            "print('PURE_PY_ISOLATED_OK')\n"
        )
        res = _run_py(script, pythonpath=pure_root)
        assert res.returncode == 0, f"Pure-Python isolation failed: {res.stderr}"
        assert "PURE_PY_ISOLATED_OK" in res.stdout


def test_root_lazy_exports_and_unknown_attr() -> None:
    import luminamoon as lm

    for name in lm.__all__:
        assert hasattr(lm, name), f"Missing root export: {name}"

    with pytest.raises(AttributeError):
        _ = getattr(lm, "nonexistent_v2_symbol")


def test_data_ffi_stubs_and_fallbacks_surface_parity() -> None:
    from luminamoon.data import _ffi, _stubs, fallbacks

    expected_symbols = set(_ffi.__all__)
    for sym in expected_symbols:
        assert hasattr(_ffi, sym), f"_ffi missing {sym}"
        assert hasattr(_stubs, sym), f"_stubs missing {sym}"
        assert hasattr(fallbacks, sym), f"fallbacks missing {sym}"

    class_methods = {
        "StreamCoordinate": ("to_bytes", "from_bytes"),
        "StreamDeduplicator": (
            "compute_hash",
            "is_seen",
            "try_add",
            "filter_batch",
            "commit",
            "rollback",
            "clear",
            "count",
            "save_wal",
            "load_wal",
            "export_hashes",
        ),
        "StateTracker": ("read", "update", "commit", "committed_batch_id"),
        "QualityFilter": ("is_high_quality", "metrics", "filter_flags", "filter_batch"),
        "MinHashDeduplicator": ("is_unique", "similarity", "count", "clear"),
        "StringTensor": (
            "from_list",
            "from_gpu_buffers",
            "to_gpu_buffers",
            "to_offsets",
            "to_payload",
            "get",
            "to_list",
            "lengths",
            "is_inlined_mask",
            "total_bytes",
            "filter_by_length",
            "filter_by_prefix",
            "hash_all",
        ),
    }

    for cls_name, methods in class_methods.items():
        ffi_cls = getattr(_ffi, cls_name)
        fb_cls = getattr(fallbacks, cls_name)
        stub_cls = getattr(_stubs, cls_name)
        for m in methods:
            assert hasattr(ffi_cls, m), f"_ffi.{cls_name} missing method {m}"
            assert hasattr(fb_cls, m), f"fallbacks.{cls_name} missing method {m}"
            assert hasattr(stub_cls, m), f"_stubs.{cls_name} missing method {m}"

    fn_names = (
        "dataloader_cpu_isa",
        "suggest_closest_matches",
        "serialize_table_to_bytes",
        "clean_single_record",
        "clean_batch_records",
        "postprocess_and_dedup_records",
        "fused_pipeline_process_batch",
    )
    for fn_name in fn_names:
        stub_params = list(inspect.signature(getattr(_stubs, fn_name)).parameters.values())
        fb_params = list(inspect.signature(getattr(fallbacks, fn_name)).parameters.values())
        assert [p.name for p in stub_params] == [p.name for p in fb_params], f"Param names mismatch for {fn_name}"
        assert [p.default for p in stub_params] == [p.default for p in fb_params], (
            f"Param defaults mismatch for {fn_name}"
        )


def test_root_python_file_line_limit() -> None:
    root_init = ROOT_PYTHON / "luminamoon" / "__init__.py"
    assert len(root_init.read_text(encoding="utf-8").splitlines()) <= 450
