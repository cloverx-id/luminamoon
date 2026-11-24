from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            'NumPy is required for sequence packing. Install it via `pip install "luminamoon[numpy]"` or `pip install numpy`.'
        ) from exc
    return np


def _try_import_torch() -> Any | None:
    try:
        import torch

        return torch
    except ImportError:
        return None


@dataclass
class TokenBatch:
    """Packed causal-LM training batch with strict document-boundary metadata.

    Attributes:
        input_ids: `(B, T)` int64 input token IDs.
        labels: `(B, T)` int64 next-token targets (`ignore_index` on pad and cross-doc boundaries).
        doc_ids: `(B, T)` int32 document IDs per row (`0` = padding, `1..K` = distinct documents).
        position_ids: `(B, T)` int64 per-document position indices resetting to `0` at each doc start.
        cu_seqlens: list of per-row cumulative document segment lengths (`[0, l1, l1+l2, ...]`).
        attention_mask: `(B, T)` bool mask (`True` for real tokens, `False` for padding).
        source_refs: list of `(source_name, source_row_idx)` records represented in this batch.
    """

    input_ids: Any
    labels: Any
    doc_ids: Any
    position_ids: Any
    cu_seqlens: list[list[int]]
    attention_mask: Any
    source_refs: list[tuple[str, int]] = field(default_factory=list)

    def as_xy(self) -> tuple[Any, Any]:
        """Return classic `(input_ids, labels)` pair."""
        return self.input_ids, self.labels

    def to(self, device: Any, *, non_blocking: bool = True) -> TokenBatch:
        """Move all PyTorch tensors in this batch to `device`."""
        if not hasattr(self.input_ids, "to"):
            return self
        return TokenBatch(
            input_ids=self.input_ids.to(device=device, non_blocking=non_blocking),
            labels=self.labels.to(device=device, non_blocking=non_blocking),
            doc_ids=self.doc_ids.to(device=device, non_blocking=non_blocking),
            position_ids=self.position_ids.to(device=device, non_blocking=non_blocking),
            cu_seqlens=self.cu_seqlens,
            attention_mask=self.attention_mask.to(device=device, non_blocking=non_blocking),
            source_refs=self.source_refs,
        )

    def causal_doc_mask(self, *, as_bool: bool = True, dtype: Any | None = None) -> Any:
        """Build `(B, 1, T, T)` block-diagonal causal mask isolating packed documents.

        Token `i` attends to token `j` iff `j <= i`, `doc_ids[i] == doc_ids[j]`, and `doc_ids[i] > 0`.
        Directly compatible with `luminamoon.nn.RaAttention(..., attn_mask=batch.causal_doc_mask())`.
        """
        torch_mod = _try_import_torch()
        if torch_mod is not None and isinstance(self.doc_ids, torch_mod.Tensor):
            d = self.doc_ids
            B, T = d.shape
            same_doc = (d.unsqueeze(2) == d.unsqueeze(1)) & (d.unsqueeze(2) > 0)
            causal = torch_mod.tril(torch_mod.ones((T, T), dtype=torch_mod.bool, device=d.device))
            mask = (same_doc & causal.unsqueeze(0)).unsqueeze(1)
            if as_bool:
                return mask
            out_dtype = dtype if dtype is not None else torch_mod.float32
            additive = torch_mod.zeros((B, 1, T, T), dtype=out_dtype, device=d.device)
            return additive.masked_fill(~mask, float("-inf"))

        np = _import_numpy()
        d_np = np.asarray(self.doc_ids)
        B, T = d_np.shape
        same_doc = (d_np[:, :, None] == d_np[:, None, :]) & (d_np[:, :, None] > 0)
        causal = np.tril(np.ones((T, T), dtype=bool))[None, :, :]
        mask_np = (same_doc & causal)[:, None, :, :]
        if as_bool:
            return mask_np
        out = np.zeros((B, 1, T, T), dtype=np.float32)
        out[~mask_np] = -np.inf
        return out


@dataclass
class _PackedSegment:
    tokens: list[int]
    source_name: str
    row_idx: int
    pos_offset: int = 0
    is_final: bool = True


@dataclass
class _PackedRow:
    segments: list[_PackedSegment] = field(default_factory=list)
    used_len: int = 0


def _split_long_segment(seg: _PackedSegment, max_len: int) -> list[_PackedSegment]:
    if len(seg.tokens) <= max_len:
        return [seg]
    chunks: list[_PackedSegment] = []
    start = 0
    total = len(seg.tokens)
    while start < total:
        end = min(start + max_len, total)
        chunks.append(
            _PackedSegment(
                tokens=seg.tokens[start:end],
                source_name=seg.source_name,
                row_idx=seg.row_idx,
                pos_offset=seg.pos_offset + start,
                is_final=seg.is_final and (end >= total),
            )
        )
        start = end
    return chunks


def _pack_best_fit_rows(segments: Sequence[_PackedSegment], seq_len: int) -> list[_PackedRow]:
    """First-Fit/Best-Fit Decreasing bin packing: zero intra-window document splits."""
    expanded: list[_PackedSegment] = []
    for seg in segments:
        if seg.tokens:
            expanded.extend(_split_long_segment(seg, seq_len))

    # Stable sort descending by length so long documents anchor bins cleanly.
    ordered = sorted(expanded, key=lambda s: len(s.tokens), reverse=True)
    rows: list[_PackedRow] = []
    rem_caps: list[int] = []

    for seg in ordered:
        seg_len = len(seg.tokens)
        best_idx = -1
        best_rem = seq_len + 1
        for idx, rem in enumerate(rem_caps):
            if seg_len <= rem < best_rem:
                best_rem = rem
                best_idx = idx
        if best_idx >= 0:
            rows[best_idx].segments.append(seg)
            rows[best_idx].used_len += seg_len
            rem_caps[best_idx] -= seg_len
        else:
            rows.append(_PackedRow(segments=[seg], used_len=seg_len))
            rem_caps.append(seq_len - seg_len)
    return rows


def _pack_concat_rows(segments: Sequence[_PackedSegment], seq_len: int) -> list[_PackedRow]:
    """Contiguous document-aware concatenation without padding waste."""
    rows: list[_PackedRow] = []
    cur = _PackedRow()
    for seg in segments:
        offset = 0
        toks = seg.tokens
        n = len(toks)
        while offset < n:
            rem = seq_len - cur.used_len
            if rem == 0:
                rows.append(cur)
                cur = _PackedRow()
                rem = seq_len
            take = min(rem, n - offset)
            cur.segments.append(
                _PackedSegment(
                    tokens=toks[offset : offset + take],
                    source_name=seg.source_name,
                    row_idx=seg.row_idx,
                    pos_offset=seg.pos_offset + offset,
                    is_final=seg.is_final and (offset + take >= n),
                )
            )
            cur.used_len += take
            offset += take
    if cur.used_len > 0:
        rows.append(cur)
    return rows


def _pack_pad_rows(segments: Sequence[_PackedSegment], seq_len: int) -> list[_PackedRow]:
    rows: list[_PackedRow] = []
    for seg in segments:
        if not seg.tokens:
            continue
        clipped = seg.tokens[:seq_len]
        rows.append(
            _PackedRow(
                segments=[
                    _PackedSegment(
                        tokens=clipped,
                        source_name=seg.source_name,
                        row_idx=seg.row_idx,
                        pos_offset=seg.pos_offset,
                    )
                ],
                used_len=len(clipped),
            )
        )
    return rows


def materialize_token_batch(
    rows: Sequence[_PackedRow],
    *,
    seq_len: int,
    pad_token_id: int = 0,
    ignore_index: int = -100,
    device: Any | None = None,
    non_blocking: bool = True,
) -> TokenBatch:
    """Materialize a sequence of `_PackedRow` into a `TokenBatch` with exact doc boundaries."""
    np = _import_numpy()
    B = len(rows)
    T = int(seq_len)

    input_ids = np.full((B, T), int(pad_token_id), dtype=np.int64)
    labels = np.full((B, T), int(ignore_index), dtype=np.int64)
    doc_ids = np.zeros((B, T), dtype=np.int32)
    position_ids = np.zeros((B, T), dtype=np.int64)
    attention_mask = np.zeros((B, T), dtype=bool)
    cu_seqlens: list[list[int]] = []
    source_refs: list[tuple[str, int]] = []

    for b, row in enumerate(rows):
        cursor = 0
        row_cu = [0]
        for doc_idx, seg in enumerate(row.segments, start=1):
            toks = seg.tokens
            seg_len = min(len(toks), T - cursor)
            if seg_len <= 0:
                break
            end = cursor + seg_len
            arr = np.asarray(toks[:seg_len], dtype=np.int64)
            input_ids[b, cursor:end] = arr
            # Causal shift strictly within the same document segment (no cross-document target).
            if seg_len > 1:
                labels[b, cursor : end - 1] = arr[1:]
            doc_ids[b, cursor:end] = doc_idx
            position_ids[b, cursor:end] = np.arange(
                seg.pos_offset, seg.pos_offset + seg_len, dtype=np.int64
            )
            attention_mask[b, cursor:end] = True
            if seg.is_final:
                source_refs.append((seg.source_name, seg.row_idx))
            cursor = end
            row_cu.append(cursor)
        cu_seqlens.append(row_cu)

    torch_mod = _try_import_torch()
    if torch_mod is not None:
        batch = TokenBatch(
            input_ids=torch_mod.from_numpy(input_ids),
            labels=torch_mod.from_numpy(labels),
            doc_ids=torch_mod.from_numpy(doc_ids),
            position_ids=torch_mod.from_numpy(position_ids),
            cu_seqlens=cu_seqlens,
            attention_mask=torch_mod.from_numpy(attention_mask),
            source_refs=source_refs,
        )
        return batch.to(device, non_blocking=non_blocking) if device is not None else batch

    return TokenBatch(
        input_ids=input_ids,
        labels=labels,
        doc_ids=doc_ids,
        position_ids=position_ids,
        cu_seqlens=cu_seqlens,
        attention_mask=attention_mask,
        source_refs=source_refs,
    )


class SequencePacker:
    """Streaming sequence packer supporting `'best_fit'`, `'concat'`, and `'pad'` modes."""

    VALID_MODES = ("best_fit", "concat", "pad")

    def __init__(
        self,
        *,
        batch_size: int = 4,
        seq_len: int = 2048,
        mode: str = "best_fit",
        pack_window: int = 64,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        drop_last: bool = False,
        device: Any | None = None,
        non_blocking: bool = True,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported packing mode '{mode}'. Expected one of {self.VALID_MODES}.")
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError(f"batch_size ({batch_size}) and seq_len ({seq_len}) must be > 0.")

        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.mode = mode
        self.pack_window = max(self.batch_size, int(pack_window))
        self.pad_token_id = int(pad_token_id)
        self.ignore_index = int(ignore_index)
        self.drop_last = bool(drop_last)
        self.device = device
        self.non_blocking = bool(non_blocking)

        self._window: list[_PackedSegment] = []
        self._ready_rows: list[_PackedRow] = []

    def add_document(self, tokens: Sequence[int], source_name: str = "default", row_idx: int = 0) -> None:
        tok_list = [int(t) for t in tokens]
        if not tok_list:
            return
        self._window.append(_PackedSegment(tokens=tok_list, source_name=source_name, row_idx=row_idx))
        if len(self._window) >= self.pack_window:
            self._pack_window()

    def _pack_window(self) -> None:
        if not self._window:
            return
        if self.mode == "best_fit":
            self._ready_rows.extend(_pack_best_fit_rows(self._window, self.seq_len))
        elif self.mode == "concat":
            self._ready_rows.extend(_pack_concat_rows(self._window, self.seq_len))
        else:
            self._ready_rows.extend(_pack_pad_rows(self._window, self.seq_len))
        self._window.clear()

    def pop_ready_batches(self) -> list[TokenBatch]:
        batches: list[TokenBatch] = []
        while len(self._ready_rows) >= self.batch_size:
            chunk = self._ready_rows[: self.batch_size]
            del self._ready_rows[: self.batch_size]
            batches.append(
                materialize_token_batch(
                    chunk,
                    seq_len=self.seq_len,
                    pad_token_id=self.pad_token_id,
                    ignore_index=self.ignore_index,
                    device=self.device,
                    non_blocking=self.non_blocking,
                )
            )
        return batches

    def flush(self) -> list[TokenBatch]:
        self._pack_window()
        batches = self.pop_ready_batches()
        if self._ready_rows and not self.drop_last:
            chunk = list(self._ready_rows)
            self._ready_rows.clear()
            batches.append(
                materialize_token_batch(
                    chunk,
                    seq_len=self.seq_len,
                    pad_token_id=self.pad_token_id,
                    ignore_index=self.ignore_index,
                    device=self.device,
                    non_blocking=self.non_blocking,
                )
            )
        else:
            self._ready_rows.clear()
        return batches


__all__ = ["SequencePacker", "TokenBatch"]
