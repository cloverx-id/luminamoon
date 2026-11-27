from __future__ import annotations

from pathlib import Path
import random
import sys
import tempfile

import pytest

ROOT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(ROOT_PYTHON) not in sys.path:
    sys.path.insert(0, str(ROOT_PYTHON))

from luminamoon.data import (
    DataPipeline,
    MinHashDeduplicator,
    PIIMasker,
    ResumablePipeline,
    StreamDeduplicator,
    StringTensor,
    TokenBatch,
    TokenRingBuffer,
)


class _WordTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [10 + (sum(b.encode("utf-8")) % 900) for b in text.split()]


def test_data_pipeline_best_fit_packing_and_doc_isolation() -> None:
    torch = pytest.importorskip("torch")
    from luminamoon.nn import RaAttention

    docs = [
        " ".join(f"doc{i}_tok{j}" for j in range(4 + (i % 7)))
        for i in range(24)
    ]
    loader = DataPipeline(
        source=docs,
        tokenizer=_WordTokenizer(),
        batch_size=2,
        seq_len=16,
        packing="best_fit",
        pack_window=24,
        clean=True,
    )
    batches = list(loader)
    assert len(batches) > 0
    b0 = batches[0]
    assert isinstance(b0, TokenBatch)
    assert b0.input_ids.shape == (2, 16)
    assert b0.labels.shape == (2, 16)
    assert b0.doc_ids.shape == (2, 16)
    assert b0.position_ids.shape == (2, 16)

    # Verify labels never leak across document boundaries or into padding
    for row_idx in range(b0.input_ids.shape[0]):
        d = b0.doc_ids[row_idx].tolist()
        lbl = b0.labels[row_idx].tolist()
        pos = b0.position_ids[row_idx].tolist()
        for t in range(len(d)):
            if d[t] == 0:
                assert lbl[t] == -100
            elif t == len(d) - 1 or d[t + 1] != d[t]:
                assert lbl[t] == -100
            if t > 0 and d[t] > 0 and d[t] != d[t - 1]:
                assert pos[t] == 0

    # Verify causal_doc_mask works directly with luminamoon.nn.RaAttention
    mask = b0.causal_doc_mask(as_bool=True)
    assert mask.shape == (2, 1, 16, 16)
    attn = RaAttention(dim=32, num_heads=4, qk_norm=True)
    hidden = torch.randn(2, 16, 32)
    out = attn(hidden, attn_mask=mask)
    assert out.shape == (2, 16, 32)
    assert torch.isfinite(out).all()


def test_data_pipeline_exact_once_crash_resume_with_best_fit_and_dedup() -> None:
    """Proves zero cursor skew & zero lost/duplicated docs across crash/resume."""
    docs = [f"unique_document_{i:03d} " + ("word " * (3 + (i % 5))) for i in range(40)]

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "train_state"
        p1 = DataPipeline(
            source=docs,
            tokenizer=_WordTokenizer(),
            batch_size=2,
            seq_len=16,
            packing="best_fit",
            pack_window=12,
            dedup="exact",
            checkpoint=ckpt,
        )
        seen_run1: list[int] = []
        for step, batch in enumerate(p1):
            assert isinstance(batch, TokenBatch)
            seen_run1.extend(row_idx for _, row_idx in batch.source_refs)
            p1.commit()
            p1.commit()  # Idempotent double commit must never crash
            if step == 2:
                break

        assert len(seen_run1) > 0
        assert len(seen_run1) == len(set(seen_run1))

        # Simulate process restart from disk checkpoint
        p2 = DataPipeline(
            source=docs,
            tokenizer=_WordTokenizer(),
            batch_size=2,
            seq_len=16,
            packing="best_fit",
            pack_window=12,
            dedup="exact",
            checkpoint=ckpt,
        )
        seen_run2: list[int] = []
        for batch in p2:
            assert isinstance(batch, TokenBatch)
            seen_run2.extend(row_idx for _, row_idx in batch.source_refs)
            p2.commit()

        assert set(seen_run1).isdisjoint(set(seen_run2)), "Duplicate docs emitted after resume!"
        assert set(seen_run1) | set(seen_run2) == set(range(len(docs))), "Documents lost across resume!"


def test_data_pipeline_multi_source_mixing_and_distributed_sharding() -> None:
    web = [f"web_{i} content text alpha beta gamma" for i in range(60)]
    code = [f"code_{i} fn main() {{ std::abs(1); }}" for i in range(60)]

    # Multi-source weighted mixing + code-safe PII masking + deterministic seed
    mix = DataPipeline(
        source={"web": (web, 0.75), "code": (code, 0.25)},
        batch_size=10,
        mask_pii=True,
        seed=123,
    )
    all_texts = [t for batch in mix for t in batch]
    assert len(all_texts) == 120
    assert any("std::abs" in t for t in all_texts)

    # Distributed sharding across 4 ranks covers 100% of records with zero overlap
    shards: list[set[str]] = []
    for rk in range(4):
        p = DataPipeline(source=web, batch_size=8, rank=rk, world_size=4)
        shards.append({t for batch in p for t in batch})
    union = set().union(*shards)
    assert len(union) == len(web)
    assert sum(len(s) for s in shards) == len(web)

    # PyTorch Stateful state_dict / load_state_dict roundtrip
    p_src = DataPipeline(source=web, batch_size=5)
    it = iter(p_src)
    b1 = next(it)
    p_src.commit()
    sd = p_src.state_dict()
    p_dst = DataPipeline(source=web, batch_size=5)
    p_dst.load_state_dict(sd)
    rest = [t for batch in p_dst for t in batch]
    assert b1 + rest == web


def test_primitives_bugfixes_arrow_pycapsule_minhash_and_ring_buffer() -> None:
    # 1. Arrow PyCapsule zero-copy roundtrip
    pa = pytest.importorskip("pyarrow")
    samples = ["hi", "twelve_bytes", "a much longer string stored in buffer pool", ""]
    st = StringTensor(samples)
    arr = pa.array(st)
    assert arr.to_pylist() == samples
    st_back = StringTensor.from_arrow(arr)
    assert st_back == st
    assert st[-1] == ""

    # 2. MinHash auto-tuned LSH + Rust Rayon filter_batch + threshold=0.0
    mh0 = MinHashDeduplicator(threshold=0.0)
    assert mh0.is_unique("first doc") is True
    assert mh0.count == 1
    mh50 = MinHashDeduplicator(threshold=0.50, num_perm=64)
    assert mh50.num_bands == 16 and mh50.rows_per_band == 4
    rng = random.Random(42)
    vocab = [f"w{i}" for i in range(400)]
    corpus = [" ".join(rng.choices(vocab, k=60)) for _ in range(150)]
    kept = mh50.filter_batch(corpus)
    assert len(kept) == len(corpus)

    # 3. ResumablePipeline multi-batch commit + idempotent double commit + early-close rollback
    with tempfile.TemporaryDirectory() as tmp:
        meta = Path(tmp) / "rp.meta"
        wal = Path(tmp) / "rp.dedup"
        docs = [f"doc_{i}" for i in range(16)]
        dedup = StreamDeduplicator(wal_path=wal)
        rp = ResumablePipeline(source=docs, checkpoint_path=meta, deduplicator=dedup, batch_size=4)
        it = iter(rp)
        _ = next(it)
        _ = next(it)
        rp.commit()
        rp.commit()  # must be idempotent
        assert dedup.count == 8  # both batches committed, not just the last 4

    # 4. TokenRingBuffer directly consuming ResumablePipeline batches (list[str] & StringTensor)
    docs_small = [f"item_{i:02d}!" for i in range(16)]
    rp_list = ResumablePipeline(source=docs_small, batch_size=8)
    with TokenRingBuffer(
        source=rp_list,
        capacity_tokens=2048,
        auto_prefill_tokens=100,
        batch_size=1,
        seq_len=16,
        pin_memory=False,
    ) as ring:
        assert ring.available_tokens == sum(len(d) for d in docs_small)

    rp_st = ResumablePipeline(source=docs_small, batch_size=8, return_string_tensor=True)
    with TokenRingBuffer(
        source=rp_st,
        capacity_tokens=2048,
        auto_prefill_tokens=100,
        batch_size=1,
        seq_len=16,
        pin_memory=False,
    ) as ring:
        assert ring.available_tokens == sum(len(d) for d in docs_small)

    # 5. PIIMasker callable on both str and Sequence[str]
    masker = PIIMasker()
    assert masker(["std::abs", "admin@example.com"]) == ["std::abs", "[EMAIL]"]
