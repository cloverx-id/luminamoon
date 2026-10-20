from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._ffi import (
    _py_damerau_levenshtein,
    _py_find_closest_matches,
    suggest_closest_matches,
)
from .constants import UNSUPPORTED_HF_URI_SCHEMES, URI_SCHEME_TYPO_MAP


def _format_did_you_mean(query: str, candidates: Sequence[str]) -> str:
    try:
        matches = suggest_closest_matches(str(query), [str(c) for c in candidates], 3)
    except BaseException:
        matches = _py_find_closest_matches(str(query), [str(c) for c in candidates], 3)
    if not matches:
        return ""
    if len(matches) == 1:
        return f" Did you mean '{matches[0]}'?"
    quoted = ", ".join(f"'{m}'" for m in matches)
    return f" Did you mean one of [{quoted}]?"


def _validate_known_kwargs(
    func_name: str,
    kwargs: Mapping[str, Any],
    valid_keys: Sequence[str],
) -> None:
    if not kwargs:
        return
    valid_set = set(valid_keys)
    for key in kwargs:
        if key not in valid_set:
            hint = _format_did_you_mean(key, valid_keys)
            valid_preview = ", ".join(f"'{k}'" for k in sorted(valid_keys))
            raise TypeError(
                f"{func_name}() got an unexpected keyword argument '{key}'.{hint} "
                f"Valid keyword arguments: [{valid_preview}]"
            )


def _diagnose_uri_typo_py(uri: str) -> str | None:
    trimmed = uri.strip()
    lower = trimmed.lower()
    if lower.startswith("hf://dataset/"):
        rest = trimmed[len("hf://dataset/") :]
        return (
            f"Invalid HuggingFace URI '{trimmed}': Did you mean "
            f"'hf://datasets/{rest}'? (Note plural 'datasets')"
        )
    if lower.startswith(("hf:/datasets/", "hf:///datasets/")):
        rest = trimmed.split("datasets/", 1)[1]
        return f"Malformed HuggingFace URI '{trimmed}': Did you mean 'hf://datasets/{rest}'?"
    for bad_prefix in UNSUPPORTED_HF_URI_SCHEMES:
        if lower.startswith(bad_prefix):
            after = trimmed[len(bad_prefix) :]
            if after.lower().startswith("datasets/"):
                after = after[len("datasets/") :]
            return f"Unsupported URI scheme in '{trimmed}': Did you mean 'hf://datasets/{after}'?"
    for typo, fixed in URI_SCHEME_TYPO_MAP:
        if lower.startswith(typo) and not lower.startswith(fixed):
            rest = trimmed[len(typo) :].lstrip("/")
            return f"Malformed URI '{trimmed}': Did you mean '{fixed}{rest}'?"
    return None


__all__ = [
    "_diagnose_uri_typo_py",
    "_format_did_you_mean",
    "_py_damerau_levenshtein",
    "_py_find_closest_matches",
    "_validate_known_kwargs",
]
