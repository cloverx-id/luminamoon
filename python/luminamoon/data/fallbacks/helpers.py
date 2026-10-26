from __future__ import annotations

from typing import List, Sequence


def dataloader_cpu_isa() -> str:
    return "SCALAR"


def _damerau_levenshtein(a: str, b: str) -> int:
    len_a = len(a)
    len_b = len(b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    width = len_b + 1
    matrix = [0] * ((len_a + 1) * width)

    for i in range(len_a + 1):
        matrix[i * width] = i
    for j in range(len_b + 1):
        matrix[j] = j

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            deletion = matrix[(i - 1) * width + j] + 1
            insertion = matrix[i * width + (j - 1)] + 1
            substitution = matrix[(i - 1) * width + (j - 1)] + cost
            best = min(deletion, insertion, substitution)

            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                transposition = matrix[(i - 2) * width + (j - 2)] + 1
                best = min(best, transposition)
            matrix[i * width + j] = best

    return matrix[len_a * width + len_b]


def suggest_closest_matches(
    target: str,
    candidates: Sequence[str],
    max_suggestions: int = 3,
) -> List[str]:
    target_trim = target.strip()
    if not target_trim or not candidates or max_suggestions <= 0:
        return []
    target_lower = target_trim.lower()

    scored: list[tuple[int, int, str]] = []
    for cand in candidates:
        c_str = str(cand)
        if not c_str:
            continue
        c_lower = c_str.lower()
        if c_lower == target_lower:
            scored.append((0, 0, c_str))
            continue
        dist = _damerau_levenshtein(target_lower, c_lower)
        max_len = max(len(target_lower), len(c_lower))
        if max_len <= 3:
            threshold = 1
        elif max_len <= 6:
            threshold = 2
        elif max_len <= 11:
            threshold = 3
        else:
            threshold = (max_len * 2) // 5
        is_sub = (target_lower in c_lower) or (c_lower in target_lower)
        if dist <= threshold or (len(target_lower) >= 3 and is_sub):
            rank_bonus = 0 if is_sub else 1
            scored.append((dist, rank_bonus, c_str))

    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    out: List[str] = []
    for _, _, s in scored:
        if not out or out[-1] != s:
            out.append(s)
            if len(out) >= max_suggestions:
                break
    return out


__all__ = ["dataloader_cpu_isa", "suggest_closest_matches"]
