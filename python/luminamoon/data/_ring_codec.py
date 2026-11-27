"""Tokenization and item-extraction helpers for TokenRingBuffer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .collate import _extract_text
from .string_tensor import StringTensor


def encode_text_to_tokens(tokenizer: Any, eos_token_id: int | None, text: str) -> list[int]:
    if not text:
        return []
    if tokenizer is not None:
        if hasattr(tokenizer, "encode"):
            try:
                encoded: Any = tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                encoded = tokenizer.encode(text)
        elif callable(tokenizer):
            try:
                res: Any = tokenizer(text, add_special_tokens=False)
            except TypeError:
                res = tokenizer(text)
            encoded = res["input_ids"] if isinstance(res, Mapping) and "input_ids" in res else res
        else:
            raise TypeError(f"Unsupported tokenizer object: {type(tokenizer)}")
        toks = [int(t) for t in encoded]
    else:
        toks = list(text.encode("utf-8", errors="replace"))
    if eos_token_id is not None and toks and toks[-1] != int(eos_token_id):
        toks.append(int(eos_token_id))
    return toks


def encode_text_batch(
    tokenizer: Any,
    eos_token_id: int | None,
    batch_tok_mode: str | None,
    texts: list[str],
) -> tuple[list[int], str | None]:
    non_empty = [t for t in texts if t]
    if not non_empty:
        return [], batch_tok_mode
    eos = int(eos_token_id) if eos_token_id is not None else None
    if tokenizer is None:
        out: list[int] = []
        for t in non_empty:
            toks = list(t.encode("utf-8", errors="replace"))
            if toks:
                out.extend(toks)
                if eos is not None and out[-1] != eos:
                    out.append(eos)
        return out, batch_tok_mode

    mode = batch_tok_mode
    if callable(tokenizer) and len(non_empty) > 1 and mode != "single":
        res: Any = None
        if mode in (None, "batch_no_special"):
            try:
                res = tokenizer(non_empty, add_special_tokens=False)
                mode = "batch_no_special"
            except TypeError:
                res = None
        if res is None and mode in (None, "batch_plain"):
            try:
                res = tokenizer(non_empty)
                mode = "batch_plain"
            except TypeError:
                mode = "single"
                res = None

        if res is not None:
            ids_batch = res["input_ids"] if isinstance(res, Mapping) and "input_ids" in res else res
            if (
                isinstance(ids_batch, (list, tuple))
                and len(ids_batch) == len(non_empty)
                and (not ids_batch or isinstance(ids_batch[0], (list, tuple)))
            ):
                out = []
                for row in ids_batch:
                    if row:
                        out.extend(int(t) for t in row)
                        if eos is not None and out[-1] != eos:
                            out.append(eos)
                return out, mode
            mode = "single"

    out = []
    for t in non_empty:
        out.extend(encode_text_to_tokens(tokenizer, eos_token_id, t))
    return out, mode


def try_extract_raw_text(item: Any, text_key: str) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping) and "input_ids" not in item:
        val = item.get(text_key)
        if isinstance(val, str):
            return val
    return None


def extract_tokens_from_item(
    item: Any,
    *,
    tokenizer: Any,
    text_key: str,
    eos_token_id: int | None,
    torch_mod: Any,
    np_mod: Any,
) -> list[int]:
    if isinstance(item, int) and not isinstance(item, bool):
        return [item]
    if isinstance(item, StringTensor):
        out: list[int] = []
        for text in item.to_list():
            out.extend(encode_text_to_tokens(tokenizer, eos_token_id, text))
        return out
    if torch_mod is not None and isinstance(item, torch_mod.Tensor):
        t_item: Any = item
        if t_item.ndim == 2:
            out = []
            for row in t_item.detach().cpu().tolist():
                if row:
                    out.extend(int(t) for t in row)
                    if eos_token_id is not None and out[-1] != int(eos_token_id):
                        out.append(int(eos_token_id))
            return out
        toks = [int(t) for t in t_item.detach().cpu().reshape(-1).tolist()]
        if eos_token_id is not None and toks and toks[-1] != int(eos_token_id):
            toks.append(int(eos_token_id))
        return toks
    if isinstance(item, np_mod.ndarray):
        arr: Any = item
        if arr.ndim == 2:
            out = []
            for row in arr.tolist():
                if row:
                    out.extend(int(t) for t in row)
                    if eos_token_id is not None and out[-1] != int(eos_token_id):
                        out.append(int(eos_token_id))
            return out
        toks = [int(t) for t in arr.reshape(-1).tolist()]
        if eos_token_id is not None and toks and toks[-1] != int(eos_token_id):
            toks.append(int(eos_token_id))
        return toks
    if isinstance(item, Mapping) and "input_ids" in item:
        raw_ids = item["input_ids"]
        if (torch_mod is not None and isinstance(raw_ids, torch_mod.Tensor)) or isinstance(raw_ids, np_mod.ndarray):
            return extract_tokens_from_item(
                raw_ids, tokenizer=tokenizer, text_key=text_key, eos_token_id=eos_token_id, torch_mod=torch_mod, np_mod=np_mod
            )
        return extract_tokens_from_item(
            list(raw_ids), tokenizer=tokenizer, text_key=text_key, eos_token_id=eos_token_id, torch_mod=torch_mod, np_mod=np_mod
        )
    if isinstance(item, (list, tuple)):
        if not item:
            return []
        first = item[0]
        if isinstance(first, int) and not isinstance(first, bool):
            toks = [int(t) for t in item]
            if eos_token_id is not None and toks and toks[-1] != int(eos_token_id):
                toks.append(int(eos_token_id))
            return toks
        out = []
        for elem in item:
            out.extend(
                extract_tokens_from_item(
                    elem, tokenizer=tokenizer, text_key=text_key, eos_token_id=eos_token_id, torch_mod=torch_mod, np_mod=np_mod
                )
            )
        return out

    text = _extract_text(item, text_key=text_key)
    return encode_text_to_tokens(tokenizer, eos_token_id, text) if text else []


__all__ = [
    "encode_text_to_tokens",
    "encode_text_batch",
    "try_extract_raw_text",
    "extract_tokens_from_item",
]
