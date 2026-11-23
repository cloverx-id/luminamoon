from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ._ffi import suggest_closest_matches
from .cleaner import TextCleaner
from .dedup import StreamDeduplicator
from .masker import PIIMasker
from .minhash import MinHashDeduplicator
from .quality import QualityFilter
from .string_tensor import StringTensor

if TYPE_CHECKING:
    from .fused import FusedPipeline


def _extract_text(item: Any, text_key: str = "text") -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    if isinstance(item, Mapping):
        if text_key not in item:
            available = [str(k) for k in item.keys()]
            suggestions = suggest_closest_matches(text_key, available, 3)
            hint = (
                f" Did you mean '{suggestions[0]}'?"
                if len(suggestions) == 1
                else (f" Did you mean one of {suggestions}?" if suggestions else "")
            )
            raise KeyError(
                f"Target column '{text_key}' not found in dataset record.{hint} Available columns: {available}"
            )
        val = item.get(text_key)
        return str(val) if val is not None else None
    if isinstance(item, (list, tuple)) and len(item) > 0:
        return str(item[0])
    return str(item) if item is not None else None


def create_resumable_collate(
    cleaner: TextCleaner | None = None,
    quality_filter: QualityFilter | None = None,
    deduplicator: StreamDeduplicator | None = None,
    minhash: MinHashDeduplicator | None = None,
    masker: PIIMasker | None = None,
    fused_pipeline: FusedPipeline | None = None,
    tokenizer: Any | None = None,
    text_key: str = "text",
    max_length: int | None = None,
    padding: bool | str = True,
    truncation: bool = True,
    return_tensors: str = "pt",
    return_string_tensor: bool = False,
) -> Callable[[list[Any]], Any]:
    """Create a `collate_fn` compatible with PyTorch DataLoader."""
    if fused_pipeline is not None and (cleaner is not None or quality_filter is not None):
        raise ValueError(
            "Pass either `fused_pipeline` OR individual `cleaner`/`quality_filter`, not both."
        )

    def collate_fn(batch: list[Any]) -> Any:
        raw_texts: list[str] = []
        for item in batch:
            t = _extract_text(item, text_key=text_key)
            if t is not None:
                raw_texts.append(t)

        if not raw_texts:
            return {} if tokenizer else ([] if not return_string_tensor else StringTensor([]))

        if fused_pipeline is not None:
            st = fused_pipeline(raw_texts)
            texts = st.to_list()
        else:
            texts = cleaner.clean_batch(raw_texts) if cleaner else raw_texts
            if quality_filter and texts:
                texts = quality_filter.filter_batch(texts)
            if deduplicator and texts:
                texts = deduplicator.filter_batch(texts)

        if minhash and texts:
            texts = minhash.filter_batch(texts)
        if masker and texts:
            texts = masker.mask_batch(texts)

        if not texts:
            if tokenizer is not None:
                try:
                    return tokenizer([], padding=padding, truncation=truncation, return_tensors=return_tensors)
                except Exception:
                    return {}
            return [] if not return_string_tensor else StringTensor([])

        if tokenizer is not None:
            if (
                padding
                and getattr(tokenizer, "pad_token", None) is None
                and getattr(tokenizer, "eos_token", None) is not None
            ):
                tokenizer.pad_token = tokenizer.eos_token
            kwargs: dict[str, Any] = {
                "padding": padding,
                "truncation": truncation,
                "return_tensors": return_tensors,
            }
            if max_length is not None:
                kwargs["max_length"] = max_length
            return tokenizer(texts, **kwargs)

        if return_string_tensor:
            return StringTensor(texts)

        return texts

    return collate_fn


__all__ = ["create_resumable_collate"]
