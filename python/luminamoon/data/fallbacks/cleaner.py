from __future__ import annotations

import json
from typing import Any, List, Optional, Sequence


def _extract_json_value(root: Any, key_path: str) -> Any:
    if isinstance(root, dict) and key_path in root:
        return root[key_path]
    if isinstance(root, list) and "." not in key_path:
        try:
            return root[int(key_path)]
        except (ValueError, IndexError):
            return None
    if "." not in key_path:
        return None
    curr: Any = root
    for segment in key_path.split("."):
        if not segment:
            continue
        if isinstance(curr, dict):
            if segment not in curr:
                return None
            curr = curr[segment]
        elif isinstance(curr, list):
            try:
                idx = int(segment)
                curr = curr[idx]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return curr


def _json_value_to_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, str):
        return val
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    return json.dumps(val, separators=(",", ":"), ensure_ascii=False)


class TextCleaner:
    def __init__(
        self,
        trim_whitespace: bool = True,
        normalize_newlines: bool = True,
        min_byte_len: int = 0,
        max_byte_len: int = 0,
        truncate_over_max: bool = False,
        json_extract_key: Optional[str] = None,
    ) -> None:
        self.trim_whitespace = trim_whitespace
        self.normalize_newlines = normalize_newlines
        self.min_byte_len = min_byte_len
        self.max_byte_len = max_byte_len
        self.truncate_over_max = truncate_over_max
        self.json_extract_key = json_extract_key

    def clean(self, text: str) -> str:
        out = text
        if self.normalize_newlines and "\r" in out:
            out = out.replace("\r\n", "\n").replace("\r", "\n")
        if self.trim_whitespace:
            out = out.strip()
        return out

    def clean_batch(self, texts: Sequence[str]) -> List[str]:
        return [self.clean(t) for t in texts]

    def extract_json(self, json_str: str, key: str = "text") -> Optional[str]:
        try:
            obj = json.loads(json_str)
        except Exception:
            return None
        return _json_value_to_str(_extract_json_value(obj, key))

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
    raw: str,
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
    t: Optional[str] = raw
    if json_extract_key:
        t = tc.extract_json(raw, json_extract_key)
        if t is None:
            return None
    res = tc.clean(t)
    raw_bytes = res.encode("utf-8")
    if min_byte_len > 0 and len(raw_bytes) < min_byte_len:
        return None
    if max_byte_len > 0 and len(raw_bytes) > max_byte_len:
        if truncate_over_max:
            res = raw_bytes[:max_byte_len].decode("utf-8", errors="ignore")
            if trim_whitespace:
                res = res.strip()
            if min_byte_len > 0 and len(res.encode("utf-8")) < min_byte_len:
                return None
        else:
            return None
    return res


def clean_batch_records(
    records: Sequence[str],
    trim_whitespace: bool = True,
    normalize_newlines: bool = True,
    min_byte_len: int = 0,
    max_byte_len: int = 0,
    truncate_over_max: bool = False,
    json_extract_key: Optional[str] = None,
) -> List[str]:
    out = []
    for t in records:
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
