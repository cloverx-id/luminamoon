from __future__ import annotations

from pathlib import Path
import sys

ROOT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(ROOT_PYTHON) not in sys.path:
    sys.path.insert(0, str(ROOT_PYTHON))

import pytest
import torch

from luminamoon import LuminaV
from luminamoon.optimizer.version import (
    SUPPORTED_VERSIONS,
    VERSION_CAPABILITIES,
    get_capabilities,
    validate_version,
)


def test_supported_versions_list():
    assert LuminaV.list_versions() == ["1.6.0", "1.5.5", "1.5.4", "1.5.3"]
    assert SUPPORTED_VERSIONS == ["1.6.0", "1.5.5", "1.5.4", "1.5.3"]


def test_version_capabilities_metadata():
    for v in ["1.6.0", "1.5.5", "1.5.4", "1.5.3"]:
        caps = LuminaV.get_capabilities(v)
        assert caps["version"] == v
        assert "allowed_buffers" in caps
        assert "supports_bucket_fusion" in caps
        assert "supports_triangle_family" in caps

    assert LuminaV.get_capabilities("1.6.0")["supports_triangle_family"] is True
    assert LuminaV.get_capabilities("1.5.5")["supports_triangle_family"] is False
    assert LuminaV.get_capabilities("1.5.4")["supports_bucket_fusion"] is False
    assert LuminaV.get_capabilities("1.5.3")["supports_bucket_fusion"] is False

    with pytest.raises(ValueError):
        LuminaV.get_capabilities("1.2.0")


def test_option_b_triangle_family_guard():
    p = torch.nn.Parameter(torch.randn(10))

    # 1.6.0 permits T1 and T2
    opt_t2 = LuminaV([p], buffer="T2", version="1.6.0")
    assert opt_t2.version == "1.6.0"
    opt_t1 = LuminaV([p], buffer="T1", version="1.6.0")
    assert opt_t1.version == "1.6.0"

    # Pre-1.6.0 strictly rejects T1 and T2
    for legacy_ver in ["1.5.5", "1.5.4", "1.5.3"]:
        with pytest.raises(ValueError, match="does not support Triangle Family"):
            LuminaV([p], buffer="T2", version=legacy_ver)
        with pytest.raises(ValueError, match="does not support Triangle Family"):
            LuminaV([p], buffer="T1", version=legacy_ver)


def test_option_b_bucket_fusion_guard():
    p = torch.nn.Parameter(torch.randn(10))

    # 1.6.0 and 1.5.5 permit fuse_tensors
    opt_155 = LuminaV([p], buffer=2, fuse_tensors=True, version="1.5.5")
    assert opt_155.version == "1.5.5"

    # 1.5.4 and 1.5.3 strictly reject fuse_tensors
    for legacy_ver in ["1.5.4", "1.5.3"]:
        with pytest.raises(ValueError, match="does not support contiguous bucket fusion"):
            LuminaV([p], buffer=2, fuse_tensors=True, version=legacy_ver)


def test_legacy_versions_allowed_configurations():
    p = torch.nn.Parameter(torch.randn(10))

    # Standard dual and td-clm without bucket fusion are valid in 1.5.4 and 1.5.3
    opt_154_dual = LuminaV([p], buffer=2, fuse_tensors=False, version="1.5.4")
    assert opt_154_dual.version == "1.5.4"
    opt_154_single = LuminaV([p], buffer=1, fuse_tensors=False, version="1.5.4")
    assert opt_154_single.version == "1.5.4"

    opt_153_single = LuminaV([p], buffer=1, fuse_tensors=False, version="1.5.3")
    assert opt_153_single.version == "1.5.3"


def test_backward_compatible_spec_aliases():
    p = torch.nn.Parameter(torch.randn(10))

    # Passing spec in constructor
    opt = LuminaV([p], spec="1.5.5")
    assert opt.version == "1.5.5"
    assert opt.spec == "1.5.5"
    assert opt.math_version == "1.5.5"
    assert opt.get_spec() == "1.5.5"
    assert opt.getSpec() == "1.5.5"
    assert opt.get_math_version() == "1.5.5"
    assert opt.getMathVersion() == "1.5.5"

    # Class-level properties and methods
    assert LuminaV.ALGORITHM_SPEC == "1.6.0"
    assert LuminaV.ALGORITHM_MATH == "1.6.0"
    assert LuminaV.get_spec() == "1.6.0"
    assert LuminaV.getSpec() == "1.6.0"
    assert LuminaV.getMathVersion() == "1.6.0"

    # setSpec instance and class
    opt.setSpec("1.5.4")
    assert opt.version == "1.5.4"
    assert opt.spec == "1.5.4"
