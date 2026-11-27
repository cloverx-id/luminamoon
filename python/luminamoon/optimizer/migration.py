from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

REMOVED_KWARGS: Dict[str, str] = {
    "version": (
        "Parameter 'version' was removed in LuminaV v2.0.0. LuminaV now always uses math v2.0.0; "
        "remove 'version' from LuminaV(...) and use LuminaV.MATH_VERSION or LuminaV.get_math_version()."
    ),
    "spec": (
        "Parameter 'spec' was removed in LuminaV v2.0.0. LuminaV now always uses math v2.0.0; "
        "remove 'spec' from LuminaV(...) and use LuminaV.MATH_VERSION or LuminaV.get_math_version()."
    ),
    "centered_defect": (
        "Parameter 'centered_defect' was removed in LuminaV v1.5.4. Gradient-momentum residuals are now "
        "computed locally per coordinate; remove 'centered_defect' from LuminaV(...)."
    ),
    "alpha_ss": (
        "Parameter 'alpha_ss' was removed in LuminaV v1.5.3. Single-buffer updates now use the "
        "Triangle Inequality Envelope; remove 'alpha_ss' and tune 'lam' (default 0.3) instead."
    ),
    "deterministic": (
        "Parameter 'deterministic' is not a LuminaV constructor argument. Set 'seed' (default 1337) "
        "for deterministic stochastic rounding, and enable torch.use_deterministic_algorithms(True) "
        "when deterministic GPU reductions are also required."
    ),
}

_VERSION_SWITCHER_MSG = (
    "Multi-version switching was removed in LuminaV v2.0.0. LuminaV now uses a single math "
    "specification (v2.0.0); use LuminaV.MATH_VERSION or LuminaV.get_math_version() instead."
)

REMOVED_ATTRIBUTES: Dict[str, str] = {
    "ALGORITHM_MATH": "'LuminaV.ALGORITHM_MATH' was removed in LuminaV v2.0.0. Use 'LuminaV.MATH_VERSION' instead.",
    "ALGORITHM_SPEC": "'LuminaV.ALGORITHM_SPEC' was removed in LuminaV v2.0.0. Use 'LuminaV.MATH_VERSION' instead.",
    "version": "'LuminaV.version' was removed in LuminaV v2.0.0. Use 'LuminaV.MATH_VERSION' or 'LuminaV.get_math_version()' instead.",
    "spec": "'LuminaV.spec' was removed in LuminaV v2.0.0. Use 'LuminaV.MATH_VERSION' or 'LuminaV.get_math_version()' instead.",
    "math_version": "'LuminaV.math_version' was removed in LuminaV v2.0.0. Use 'LuminaV.MATH_VERSION' or 'LuminaV.get_math_version()' instead.",
    "get_version": "'LuminaV.get_version()' was removed in LuminaV v2.0.0. Use 'LuminaV.get_math_version()' instead.",
    "getVersion": "'LuminaV.getVersion()' was removed in LuminaV v2.0.0. Use 'LuminaV.get_math_version()' instead.",
    "get_spec": "'LuminaV.get_spec()' was removed in LuminaV v2.0.0. Use 'LuminaV.get_math_version()' instead.",
    "getSpec": "'LuminaV.getSpec()' was removed in LuminaV v2.0.0. Use 'LuminaV.get_math_version()' instead.",
    "getMathVersion": "'LuminaV.getMathVersion()' was removed in LuminaV v2.0.0. Use 'LuminaV.get_math_version()' instead.",
    "set_version": f"'LuminaV.set_version()' was removed in LuminaV v2.0.0. {_VERSION_SWITCHER_MSG}",
    "setVersion": f"'LuminaV.setVersion()' was removed in LuminaV v2.0.0. {_VERSION_SWITCHER_MSG}",
    "set_spec": f"'LuminaV.set_spec()' was removed in LuminaV v2.0.0. {_VERSION_SWITCHER_MSG}",
    "setSpec": f"'LuminaV.setSpec()' was removed in LuminaV v2.0.0. {_VERSION_SWITCHER_MSG}",
    "list_versions": f"'LuminaV.list_versions()' was removed in LuminaV v2.0.0. {_VERSION_SWITCHER_MSG}",
    "listVersions": f"'LuminaV.listVersions()' was removed in LuminaV v2.0.0. {_VERSION_SWITCHER_MSG}",
    "get_capabilities": (
        "'LuminaV.get_capabilities()' was removed in LuminaV v2.0.0. All v2.0.0 optimizer features "
        "(buffer='t1'/'t2', fuse_tensors, etc.) are enabled by default; inspect backend availability "
        "with 'LuminaV.detect_hardware()' instead."
    ),
}

REMOVED_VERSION_EXPORTS: Dict[str, str] = {
    "ALGORITHM_MATH_VERSION": "'luminamoon.optimizer.version.ALGORITHM_MATH_VERSION' was removed in v2.0.0. Use 'MATH_VERSION' instead.",
    "ALGORITHM_SPEC_VERSION": "'luminamoon.optimizer.version.ALGORITHM_SPEC_VERSION' was removed in v2.0.0. Use 'MATH_VERSION' instead.",
    "DEFAULT_VERSION": "'luminamoon.optimizer.version.DEFAULT_VERSION' was removed in v2.0.0. Use 'MATH_VERSION' instead.",
    "DEFAULT_LUMINAV_VERSION": "'luminamoon.optimizer.version.DEFAULT_LUMINAV_VERSION' was removed in v2.0.0. Use 'MATH_VERSION' instead.",
    "REFERENCE_LUMINAV_VERSION": "'luminamoon.optimizer.version.REFERENCE_LUMINAV_VERSION' was removed in v2.0.0. Use 'MATH_VERSION' instead.",
    "SUPPORTED_VERSIONS": f"'luminamoon.optimizer.version.SUPPORTED_VERSIONS' was removed in v2.0.0. {_VERSION_SWITCHER_MSG}",
    "SUPPORTED_BUNDLED_VERSIONS": f"'luminamoon.optimizer.version.SUPPORTED_BUNDLED_VERSIONS' was removed in v2.0.0. {_VERSION_SWITCHER_MSG}",
    "VERSION_CAPABILITIES": "'luminamoon.optimizer.version.VERSION_CAPABILITIES' was removed in v2.0.0. Use 'LuminaV.detect_hardware()' for backend availability.",
    "VERSION_TAG_MAP": f"'luminamoon.optimizer.version.VERSION_TAG_MAP' was removed in v2.0.0. {_VERSION_SWITCHER_MSG}",
    "CACHE_ROOT": f"'luminamoon.optimizer.version.CACHE_ROOT' was removed in v2.0.0. {_VERSION_SWITCHER_MSG}",
    "GITHUB_REPO": f"'luminamoon.optimizer.version.GITHUB_REPO' was removed in v2.0.0. {_VERSION_SWITCHER_MSG}",
    "HybridVersionMethod": f"'luminamoon.optimizer.version.HybridVersionMethod' was removed in v2.0.0. {_VERSION_SWITCHER_MSG}",
    "ModularVersionResolver": f"'luminamoon.optimizer.version.ModularVersionResolver' was removed in v2.0.0. {_VERSION_SWITCHER_MSG}",
    "get_active_optimizer_class": "'luminamoon.optimizer.version.get_active_optimizer_class()' was removed in v2.0.0. Import 'LuminaV' directly from 'luminamoon.optimizer'.",
    "get_capabilities": "'luminamoon.optimizer.version.get_capabilities()' was removed in v2.0.0. Use 'LuminaV.detect_hardware()' for backend availability.",
    "get_version": "'luminamoon.optimizer.version.get_version()' was removed in v2.0.0. Use 'get_math_version()' (math spec) or 'get_package_version()' (package version).",
    "list_versions": f"'luminamoon.optimizer.version.list_versions()' was removed in v2.0.0. {_VERSION_SWITCHER_MSG}",
    "set_version": f"'luminamoon.optimizer.version.set_version()' was removed in v2.0.0. {_VERSION_SWITCHER_MSG}",
    "validate_version": f"'luminamoon.optimizer.version.validate_version()' was removed in v2.0.0. {_VERSION_SWITCHER_MSG}",
}

REMOVED_OPTIMIZER_SUBMODULES: Dict[str, str] = {
    "gpu_backend": "'luminamoon.optimizer.gpu_backend' was moved in v2.0.0. Use 'luminamoon.optimizer.backends.gpu' instead.",
    "fsdp_sync": "'luminamoon.optimizer.fsdp_sync' was moved in v2.0.0. Use 'luminamoon.optimizer.backends.fsdp' instead.",
    "numpy_backend": "'luminamoon.optimizer.numpy_backend' was moved in v2.0.0. Use 'luminamoon.optimizer.backends.numpy_backend' instead.",
    "triton_kernels": "'luminamoon.optimizer.triton_kernels' was moved in v2.0.0. Use 'luminamoon.optimizer.backends.triton' instead.",
}

REMOVED_TOP_LEVEL_EXPORTS: Dict[str, str] = {
    "list_versions": (
        "'luminamoon.list_versions' was removed in v2.0.0. Multi-version switching was removed; "
        "use 'luminamoon.get_version()' for the package version or 'LuminaV.get_math_version()' for optimizer math."
    ),
    "set_version": (
        "'luminamoon.set_version' was removed in v2.0.0. Multi-version switching was removed; "
        "LuminaV always uses math v2.0.0 ('LuminaV.MATH_VERSION')."
    ),
}

_LEGACY_BUFFER_STR_MAP: Dict[str, str] = {
    "1": "buffer=1 (int)",
    "tdclm": "buffer=1",
    "td-clm": "buffer=1",
    "single": "buffer=1",
    "2": "buffer=2 (int)",
    "dual": "buffer=2",
    "clm": "buffer=2",
    "td-t1": "buffer='t1'",
    "t1-clm": "buffer='t1'",
    "triangle1": "buffer='t1'",
    "td-clm-t1": "buffer='t1'",
    "tdclm-t1": "buffer='t1'",
    "td-t2": "buffer='t2'",
    "t2-clm": "buffer='t2'",
    "td-dual": "buffer='t2'",
    "triangle2": "buffer='t2'",
    "clm-t2": "buffer='t2'",
    "dual-t2": "buffer='t2'",
}

_LEGACY_CHOICE_STR_MAPS: Dict[str, Dict[str, str]] = {
    "bound_type": {
        "none": "bound=False",
        "0": "bound=False",
        "1": "bound_type='radial'",
        "2": "bound_type='coordinate'",
    },
    "master_weights": {
        "false": "master_weights='none'",
        "off": "master_weights='none'",
        "0": "master_weights='none'",
        "hybrid": "master_weights='semi'",
        "half": "master_weights='semi'",
        "1": "master_weights='semi'",
        "true": "master_weights='full'",
        "fp32": "master_weights='full'",
        "masterweight": "master_weights='full'",
        "2": "master_weights='full'",
    },
    "execution": {
        "foreach": "execution='auto'",
        "0": "execution='auto'",
        "1": "execution='triton'",
        "single": "execution='eager'",
        "2": "execution='eager'",
        "vulkan": "execution='cubecl'",
        "3": "execution='cubecl'",
        "4": "execution='simd'",
        "scalar_core": "execution='scalar'",
        "5": "execution='scalar'",
        "6": "execution='cubecl'",
    },
}

_LEGACY_CHOICE_INT_MAPS: Dict[str, Dict[int, str]] = {
    "bound_type": {0: "bound=False", 1: "bound_type='radial'", 2: "bound_type='coordinate'"},
    "master_weights": {0: "master_weights='none'", 1: "master_weights='semi'", 2: "master_weights='full'"},
    "execution": {
        0: "execution='auto'",
        1: "execution='triton'",
        2: "execution='eager'",
        3: "execution='cubecl'",
        4: "execution='simd'",
        5: "execution='scalar'",
        6: "execution='cubecl'",
    },
}


def reject_removed_kwargs(kwargs: Mapping[str, Any], *, as_value_error: bool = False) -> None:
    for key in kwargs:
        msg = REMOVED_KWARGS.get(key)
        if msg is not None:
            if as_value_error:
                raise ValueError(msg)
            raise TypeError(msg)
        if not as_value_error:
            raise TypeError(f"LuminaV.__init__() got an unexpected keyword argument '{key}'")


def removed_attr_message(name: str) -> Optional[str]:
    return REMOVED_ATTRIBUTES.get(name)


def removed_version_attr_message(name: str) -> Optional[str]:
    return REMOVED_VERSION_EXPORTS.get(name)


def removed_optimizer_attr_message(name: str) -> Optional[str]:
    if name == "version":
        return None
    msg = REMOVED_ATTRIBUTES.get(name)
    if msg is not None:
        return msg
    return REMOVED_OPTIMIZER_SUBMODULES.get(name)


def removed_top_level_attr_message(name: str) -> Optional[str]:
    return REMOVED_TOP_LEVEL_EXPORTS.get(name)


def legacy_buffer_message(value: Any) -> Optional[str]:
    if isinstance(value, str):
        repl = _LEGACY_BUFFER_STR_MAP.get(value.strip().lower())
        if repl is not None:
            return f"Legacy buffer={value!r} was removed in LuminaV v2.0.0. Use {repl} instead."
    return None


def legacy_bound_message(value: Any) -> Optional[str]:
    if isinstance(value, int) and value in (0, 1):
        return (
            f"Invalid bound: {value!r}. In LuminaV v2.0.0, 'bound' must be a boolean (True or False). "
            f"Use bound={bool(value)} instead."
        )
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in ("true", "false"):
            return (
                f"Invalid bound: {value!r}. In LuminaV v2.0.0, 'bound' must be a boolean (True or False). "
                f"Use bound={lower == 'true'} instead."
            )
    return None


def legacy_choice_message(name: str, value: Any) -> Optional[str]:
    if isinstance(value, bool):
        if name == "master_weights":
            repl = "master_weights='full'" if value else "master_weights='none'"
            return f"Legacy master_weights={value!r} was removed in LuminaV v2.0.0. Use {repl} instead."
        return None
    if isinstance(value, int):
        repl = _LEGACY_CHOICE_INT_MAPS.get(name, {}).get(value)
        if repl is not None:
            return f"Legacy {name}={value!r} was removed in LuminaV v2.0.0. Use {repl} instead."
        return None
    if isinstance(value, str):
        repl = _LEGACY_CHOICE_STR_MAPS.get(name, {}).get(value.strip().lower())
        if repl is not None:
            return f"Legacy {name}={value!r} was removed in LuminaV v2.0.0. Use {repl} instead."
    return None
