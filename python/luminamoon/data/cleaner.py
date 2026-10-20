from __future__ import annotations

from collections.abc import Sequence
from typing import overload

from ._ffi import clean_batch_records, clean_single_record


class TextCleaner:
    """High-performance text cleaning and normalization backed by Rust SIMD and Rayon.

    Features:
    - AVX-512 / AVX2 / NEON fast carriage-return stripping (`\\r\\n` -> `\\n`).
    - Whitespace trimming.
    - JSON field extraction via dotted path without Python JSON overhead.
    - Byte length filtering (minimum and maximum bounds).
    - Optional UTF-8 safe boundary truncation.
    """

    def __init__(
        self,
        trim_whitespace: bool = True,
        normalize_newlines: bool = True,
        min_byte_len: int | None = None,
        max_byte_len: int | None = None,
        truncate_over_max: bool = False,
        json_extract_key: str | None = None,
    ) -> None:
        self.trim_whitespace = trim_whitespace
        self.normalize_newlines = normalize_newlines
        self.min_byte_len = 0 if min_byte_len is None else max(0, min_byte_len)
        self.max_byte_len = 0 if max_byte_len is None else max(0, max_byte_len)
        self.truncate_over_max = truncate_over_max
        self.json_extract_key = json_extract_key

    def clean_single(self, text: str) -> str | None:
        """Cleans a single string. Returns None if filtered by length or format."""
        return clean_single_record(
            text,
            trim_whitespace=self.trim_whitespace,
            normalize_newlines=self.normalize_newlines,
            min_byte_len=self.min_byte_len,
            max_byte_len=self.max_byte_len,
            truncate_over_max=self.truncate_over_max,
            json_extract_key=self.json_extract_key,
        )

    def clean_batch(self, texts: Sequence[str]) -> list[str]:
        """Cleans a sequence of strings in parallel using Rayon without Python GIL."""
        if not texts:
            return []
        return clean_batch_records(
            texts,
            trim_whitespace=self.trim_whitespace,
            normalize_newlines=self.normalize_newlines,
            min_byte_len=self.min_byte_len,
            max_byte_len=self.max_byte_len,
            truncate_over_max=self.truncate_over_max,
            json_extract_key=self.json_extract_key,
        )

    @overload
    def __call__(self, target: str) -> str | None: ...  # pyright: ignore[reportOverlappingOverload]

    @overload
    def __call__(self, target: Sequence[str]) -> list[str]: ...

    def __call__(self, target: str | Sequence[str]) -> str | None | list[str]:
        if isinstance(target, str):
            return self.clean_single(target)
        return self.clean_batch(target)


def clean_text(
    text: str,
    trim_whitespace: bool = True,
    normalize_newlines: bool = True,
    min_byte_len: int | None = None,
    max_byte_len: int | None = None,
    truncate_over_max: bool = False,
    json_extract_key: str | None = None,
) -> str | None:
    """Convenience function to clean a single text string."""
    return clean_single_record(
        text,
        trim_whitespace=trim_whitespace,
        normalize_newlines=normalize_newlines,
        min_byte_len=0 if min_byte_len is None else max(0, min_byte_len),
        max_byte_len=0 if max_byte_len is None else max(0, max_byte_len),
        truncate_over_max=truncate_over_max,
        json_extract_key=json_extract_key,
    )


def clean_batch(
    texts: Sequence[str],
    trim_whitespace: bool = True,
    normalize_newlines: bool = True,
    min_byte_len: int | None = None,
    max_byte_len: int | None = None,
    truncate_over_max: bool = False,
    json_extract_key: str | None = None,
) -> list[str]:
    """Convenience function to clean a batch of text strings in parallel."""
    return clean_batch_records(
        texts,
        trim_whitespace=trim_whitespace,
        normalize_newlines=normalize_newlines,
        min_byte_len=0 if min_byte_len is None else max(0, min_byte_len),
        max_byte_len=0 if max_byte_len is None else max(0, max_byte_len),
        truncate_over_max=truncate_over_max,
        json_extract_key=json_extract_key,
    )
