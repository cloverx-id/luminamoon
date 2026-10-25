from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Sequence

class TextCleaner:
    def __init__(
        self,
        trim_whitespace: bool = True,
        normalize_newlines: bool = True,
        min_byte_len: int = 0,
        max_byte_len: int = 0,
        truncate_over_max: bool = False,
        json_extract_key: Optional[str] = None,
        collapse_whitespace: bool = True,
        strip_edges: bool = True,
    ) -> None:
        self.trim_whitespace = trim_whitespace
        self.normalize_newlines = normalize_newlines
        self.min_byte_len = min_byte_len
        self.max_byte_len = max_byte_len
        self.truncate_over_max = truncate_over_max
        self.json_extract_key = json_extract_key
        self.collapse_whitespace = collapse_whitespace
        self.strip_edges = strip_edges

    def clean(self, text: str) -> str:
        out = text.replace("\r\n", "\n").replace("\r", "\n")
        out = "".join(ch if (ch == "\n" or ord(ch) >= 0x20) else " " for ch in out)
        if self.collapse_whitespace:
            out = re.sub(r"[ \t]+", " ", out)
        if self.strip_edges:
            out = out.strip()
        return out

    def clean_batch(self, texts: Sequence[str]) -> List[str]:
        return [self.clean(t) for t in texts]

    def extract_json(self, json_str: str, key: str = "text") -> Optional[str]:
        try:
            obj = json.loads(json_str)
        except Exception:
            return None
        curr: Any = obj
        for part in key.split("."):
            if isinstance(curr, dict) and part in curr:
                curr = curr[part]
            else:
                return None
        return curr if isinstance(curr, str) else None

    def extract_json_batch(self, json_lines: Sequence[str], key: str = "text") -> List[str]:
        out: List[str] = []
        for line in json_lines:
            extracted = self.extract_json(line, key)
            if extracted is not None:
                cleaned = self.clean(extracted)
                if cleaned:
                    out.append(cleaned)
        return out


def clean_single_record(
    text: str,
    trim_whitespace: bool = True,
    normalize_newlines: bool = True,
    min_byte_len: int = 0,
    max_byte_len: int = 0,
    truncate_over_max: bool = False,
    json_extract_key: Optional[str] = None,
) -> Optional[str]:
    tc = TextCleaner(
        trim_whitespace=trim_whitespace,
        normalize_newlines=normalize_newlines,
        min_byte_len=min_byte_len,
        max_byte_len=max_byte_len,
        truncate_over_max=truncate_over_max,
        json_extract_key=json_extract_key,
    )
    t: Optional[str] = text
    if json_extract_key:
        t = tc.extract_json(text, json_extract_key)
        if t is None:
            return None
    res = tc.clean(t)
    raw_len = len(res.encode("utf-8"))
    if min_byte_len > 0 and raw_len < min_byte_len:
        return None
    if max_byte_len > 0 and raw_len > max_byte_len:
        if truncate_over_max:
            res = res.encode("utf-8")[:max_byte_len].decode("utf-8", errors="ignore")
        else:
            return None
    return res


def clean_batch_records(
    texts: Sequence[str],
    trim_whitespace: bool = True,
    normalize_newlines: bool = True,
    min_byte_len: int = 0,
    max_byte_len: int = 0,
    truncate_over_max: bool = False,
    json_extract_key: Optional[str] = None,
) -> List[str]:
    out = []
    for t in texts:
        c = clean_single_record(
            t,
            trim_whitespace=trim_whitespace,
            normalize_newlines=normalize_newlines,
            min_byte_len=min_byte_len,
            max_byte_len=max_byte_len,
            truncate_over_max=truncate_over_max,
            json_extract_key=json_extract_key,
        )
        if c is not None:
            out.append(c)
    return out

__all__ = ["TextCleaner", "clean_batch_records", "clean_single_record"]
