from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
import json
import os
from pathlib import Path
import random
import threading
from typing import Any


def resolve_distributed_context(
    rank: int | None,
    world_size: int | None,
) -> tuple[int, int]:
    r = rank
    ws = world_size
    if r is None or ws is None:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                if r is None:
                    r = dist.get_rank()
                if ws is None:
                    ws = dist.get_world_size()
        except Exception:
            pass
    eff_rank = 0 if r is None else int(r)
    eff_world = 1 if ws is None else int(ws)

    # Fold PyTorch DataLoader worker sharding into effective (rank, world_size) automatically.
    try:
        from torch.utils.data import get_worker_info

        wi = get_worker_info()
        if wi is not None and wi.num_workers > 1:
            eff_rank = eff_rank * wi.num_workers + wi.id
            eff_world = eff_world * wi.num_workers
    except Exception:
        pass

    if eff_world <= 0 or not (0 <= eff_rank < eff_world):
        raise ValueError(f"Invalid distributed sharding: rank={eff_rank}, world_size={eff_world}.")
    return eff_rank, eff_world


def normalize_sources(
    source: Iterable[Any] | Mapping[str, Any],
) -> tuple[list[tuple[str, Iterable[Any], float]], bool]:
    if isinstance(source, Mapping):
        if not source:
            raise ValueError("source mapping cannot be empty.")
        entries: list[tuple[str, Iterable[Any], float]] = []
        for name, spec in source.items():
            if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[1], (int, float)):
                src_iter, weight = spec[0], float(spec[1])
            else:
                src_iter, weight = spec, 1.0
            if weight <= 0.0:
                raise ValueError(f"Mixture weight for source '{name}' must be > 0, got {weight}.")
            entries.append((str(name), src_iter, weight))
        return entries, True
    return [("default", source, 1.0)], False


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{path}.tmp.{os.getpid()}.{threading.get_ident()}")
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def iter_sharded_source(
    name: str,
    src: Iterable[Any],
    *,
    rank: int,
    world_size: int,
    start_row: int,
    already_emitted: set[int],
) -> Iterator[tuple[str, int, Any]]:
    for idx, item in enumerate(src):
        if idx < start_row:
            continue
        if world_size > 1 and (idx % world_size) != rank:
            continue
        if idx in already_emitted:
            continue
        yield name, idx, item


def iter_mixed_records(
    sources: list[tuple[str, Iterable[Any], float]],
    *,
    is_mixture: bool,
    rank: int,
    world_size: int,
    committed_cursors: dict[str, int],
    committed_above: dict[str, set[int]],
    seed: int | None,
    shuffle_buffer: int,
) -> Iterator[tuple[str, int, Any]]:
    rng = random.Random((seed or 0) + rank * 1009)
    if not is_mixture:
        name, src, _ = sources[0]
        raw_iter = iter_sharded_source(
            name,
            src,
            rank=rank,
            world_size=world_size,
            start_row=committed_cursors.get(name, 0),
            already_emitted=committed_above.get(name, set()),
        )
    else:
        active_iters: list[tuple[str, Iterator[tuple[str, int, Any]], float]] = [
            (
                name,
                iter_sharded_source(
                    name,
                    src,
                    rank=rank,
                    world_size=world_size,
                    start_row=committed_cursors.get(name, 0),
                    already_emitted=committed_above.get(name, set()),
                ),
                w,
            )
            for name, src, w in sources
        ]

        def _mix() -> Iterator[tuple[str, int, Any]]:
            while active_iters:
                weights = [w for _, _, w in active_iters]
                chosen_idx = rng.choices(range(len(active_iters)), weights=weights, k=1)[0]
                _, it, _ = active_iters[chosen_idx]
                try:
                    yield next(it)
                except StopIteration:
                    active_iters.pop(chosen_idx)

        raw_iter = _mix()

    if shuffle_buffer <= 1:
        yield from raw_iter
        return

    buf: list[tuple[str, int, Any]] = []
    for rec in raw_iter:
        buf.append(rec)
        if len(buf) >= shuffle_buffer:
            pop_idx = rng.randrange(len(buf))
            buf[pop_idx], buf[-1] = buf[-1], buf[pop_idx]
            yield buf.pop()
    rng.shuffle(buf)
    yield from buf


__all__ = [
    "atomic_write_json",
    "iter_mixed_records",
    "normalize_sources",
    "resolve_distributed_context",
]
