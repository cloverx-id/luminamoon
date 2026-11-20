from __future__ import annotations

from pathlib import Path
import sys
import tempfile

ROOT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(ROOT_PYTHON) not in sys.path:
    sys.path.insert(0, str(ROOT_PYTHON))

import pytest
import torch

import luminamoon
from luminamoon import LuminaV
from luminamoon.optimizer import MATH_VERSION, get_math_version


def test_package_version_read_only():
    assert isinstance(luminamoon.__version__, str) and len(luminamoon.__version__) > 0
    assert luminamoon.get_version() == luminamoon.__version__
    assert not hasattr(luminamoon, "set_version")
    assert not hasattr(luminamoon, "list_versions")


def test_optimizer_math_version_read_only():
    assert MATH_VERSION == "1.6.0"
    assert get_math_version() == MATH_VERSION
    assert LuminaV.MATH_VERSION == MATH_VERSION
    assert LuminaV.get_math_version() == MATH_VERSION

    p = torch.nn.Parameter(torch.randn(8))
    opt = LuminaV([p])
    assert opt.get_math_version() == MATH_VERSION

    for removed in (
        "SPEC_VERSION",
        "ALGORITHM_SPEC",
        "ALGORITHM_MATH",
        "set_version",
        "setVersion",
        "get_version",
        "getVersion",
        "set_spec",
        "setSpec",
        "get_spec",
        "getSpec",
        "list_versions",
        "listVersions",
        "get_capabilities",
        "getCapabilities",
        "getMathVersion",
    ):
        assert not hasattr(LuminaV, removed), f"Legacy attribute {removed} must be removed in v2.0.0"

    with pytest.raises(TypeError):
        LuminaV([p], version="1.5.5")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        LuminaV([p], spec="1.5.5")  # type: ignore[call-arg]


def test_version_detect_ignores_unrelated_parent_pyproject():
    from luminamoon import _version

    with tempfile.TemporaryDirectory() as tmpdir:
        fake_root = Path(tmpdir)
        (fake_root / "pyproject.toml").write_text(
            '[project]\nname = "unrelated-user-app"\nversion = "9.9.9"\n',
            encoding="utf-8",
        )
        fake_pkg = fake_root / "python" / "luminamoon"
        fake_pkg.mkdir(parents=True)
        fake_file = fake_pkg / "_version.py"
        fake_file.write_text("", encoding="utf-8")

        orig_file = _version.__file__
        try:
            _version.__file__ = str(fake_file)
            detected = _version._detect_version()
            assert detected != "9.9.9"
        finally:
            _version.__file__ = orig_file
