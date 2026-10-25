from __future__ import annotations

from typing import List, Sequence

def dataloader_cpu_isa() -> str:
    return "SCALAR"


def suggest_closest_matches(query: str, candidates: Sequence[str], top_k: int = 3) -> List[str]:
    return list(candidates[:top_k])

__all__ = ["dataloader_cpu_isa", "suggest_closest_matches"]
