from __future__ import annotations

from pathlib import Path
import sys
import tempfile

ROOT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(ROOT_PYTHON) not in sys.path:
    sys.path.insert(0, str(ROOT_PYTHON))

from luminamoon.data import (
    FusedPipeline,
    MinHashDeduplicator,
    PIIMasker,
    QualityFilter,
    ResumablePipeline,
    StateTracker,
    StreamDeduplicator,
    StringTensor,
    TextCleaner,
    TokenRingBuffer,
    create_resumable_collate,
)


def test_adapters_collate_and_pipeline() -> None:
    cleaner = TextCleaner()
    qf = QualityFilter.for_pretraining()
    dedup = StreamDeduplicator()
    masker = PIIMasker()

    class MockTokenizer:
        def __call__(self, texts, **kw):
            return {
                "input_ids": [[len(t)] for t in texts],
                "texts": texts,
            }

    collate = create_resumable_collate(
        cleaner=cleaner,
        quality_filter=qf,
        deduplicator=dedup,
        masker=masker,
        tokenizer=MockTokenizer(),
    )

    batch_items = [
        {"text": "A high quality training sample with email test@example.com.\r\n"},
        {"text": "A high quality training sample with email test@example.com."},
        {"text": "spam spam spam spam spam spam spam spam spam spam spam spam"},
    ]

    res = collate(batch_items)
    assert "input_ids" in res
    assert len(res["input_ids"]) == 1
    assert "[EMAIL]" in res["texts"][0]

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipe_meta = Path(tmp_dir) / "pipe.meta"
        pipe_dedup = Path(tmp_dir) / "pipe.dedup"

        data_source = [
            "Valid sample 1 with content.",
            "Valid sample 2 with content.",
            "Valid sample 1 with content.",
            "spam spam spam spam spam spam spam spam spam spam spam",
            "Valid sample 3 with content.",
        ]

        pipe = ResumablePipeline(
            source=data_source,
            checkpoint_path=pipe_meta,
            cleaner=cleaner,
            quality_filter=qf,
            deduplicator=StreamDeduplicator(wal_path=pipe_dedup),
            batch_size=2,
        )

        batches = list(pipe)
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1
        pipe.commit()
        assert pipe_meta.stat().st_size == 128

        try:
            collate([{"txet": "typo column"}])
            raise AssertionError("Expected KeyError on missing column")
        except KeyError as exc:
            assert "Target column 'text' not found" in str(exc)
            assert "txet" in str(exc)

        class MockHFTokenizerNoPad:
            pad_token = None
            eos_token = "<|endoftext|>"

            def __call__(self, texts, **kw):
                return {"texts": texts, "pad_token": self.pad_token}

        hf_tok = MockHFTokenizerNoPad()
        collate_hf = create_resumable_collate(tokenizer=hf_tok, padding=True)
        hf_out = collate_hf([b"raw utf8 bytes sample"])
        assert hf_tok.pad_token == "<|endoftext|>"
        assert hf_out["texts"] == ["raw utf8 bytes sample"]


def test_hf_style_dataset_simulation() -> None:
    """Simulate streaming dataset resume across shards without re-processing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        meta_file = Path(tmp_dir) / "sim.meta"
        wal_file = Path(tmp_dir) / "sim.dedup"

        mock_shards = [
            ["Text 0", "Text 1", "Text 1", "Text 2"],
            ["Text 3", "Text 4", "Text 2"],
            ["Text 5", "Text 6", "Text 7"],
        ]

        cleaner = TextCleaner()
        dedup = StreamDeduplicator(wal_path=wal_file)
        tracker = StateTracker(meta_path=meta_file)

        processed_records: list[str] = []

        for shard_idx, shard_data in enumerate(mock_shards):
            for row_idx, raw_sample in enumerate(shard_data):
                cleaned = cleaner.clean_single(raw_sample)
                if cleaned is None or not dedup.is_unique(cleaned):
                    continue

                processed_records.append(cleaned)
                tracker.update(
                    shard_id=shard_idx,
                    row_offset_in_page=row_idx,
                    global_row_idx=len(processed_records),
                )
                tracker.commit()
                dedup.commit()

            if shard_idx == 0:
                break

        assert processed_records == ["Text 0", "Text 1", "Text 2"]
        assert dedup.count == 3

        resume_coord = tracker.read()
        resumed_shard = resume_coord.shard_id
        resumed_row = resume_coord.row_offset_in_page

        resumed_records: list[str] = []

        for shard_idx, shard_data in enumerate(mock_shards):
            if shard_idx < resumed_shard:
                continue

            for row_idx, raw_sample in enumerate(shard_data):
                if shard_idx == resumed_shard and row_idx <= resumed_row:
                    continue

                cleaned = cleaner.clean_single(raw_sample)
                if cleaned is None or not dedup.is_unique(cleaned):
                    continue

                resumed_records.append(cleaned)
                tracker.update(
                    shard_id=shard_idx,
                    row_offset_in_page=row_idx,
                    global_row_idx=len(resumed_records),
                )
                tracker.commit()
                dedup.commit()

        assert resumed_records == ["Text 3", "Text 4", "Text 5", "Text 6", "Text 7"]


def test_data_edge_cases() -> None:
    masker = PIIMasker(mask_ips=True)
    ipv6_text = "Connected to 2001:0db8:85a3:0000:0000:8a2e:0370:7334 securely."
    assert "[IP]" in masker(ipv6_text)
    assert "2001:0db8" not in masker(ipv6_text)

    qf = QualityFilter(min_words=50)
    collate = create_resumable_collate(quality_filter=qf)
    empty_result = collate([{"text": "too short text"}])
    assert empty_result == []

    with tempfile.TemporaryDirectory() as tmp_dir:
        meta = Path(tmp_dir) / "empty.meta"
        pipe = ResumablePipeline(source=[], checkpoint_path=meta)
        pipe.commit()

    dedup = MinHashDeduplicator(threshold=0.85, k_shingle=4, num_perm=64)
    topics = [
        "Astrophysics explores stars, galaxies, black holes, and the origins of cosmic radiation.",
        "Organic chemistry investigates carbon compounds, reaction mechanisms, and molecular catalysts.",
        "Classical economics analyzes market equilibrium, supply, demand, and fiscal policies.",
        "Microbiology studies bacteria, viruses, fungi, and immune system antibody responses.",
        "Marine biology focuses on coral reefs, oceanic currents, and abyssal trench biodiversity.",
    ]
    for topic in topics:
        assert dedup.is_unique(topic) is True
    assert dedup.count == 5

    near_topic_0 = "Astrophysics explores stars, galaxies, black holes, and the origins of cosmic radiation!"
    assert dedup.is_unique(near_topic_0) is False


def test_string_tensor_cpu_and_gpu() -> None:
    samples = [
        "short",
        "exactly12byte",
        "12_bytes_ok!",
        "this is a much longer sample text that exceeds 12 bytes",
        "pref_alpha",
        "pref_beta_very_long_sentence_with_prefix",
        "",
    ]

    st = StringTensor(samples)
    assert len(st) == 7
    assert st[0] == "short"
    assert st[1] == "exactly12byte"
    assert st[-1] == ""
    assert st.to_list() == samples

    st_sub = st[1:4]
    assert isinstance(st_sub, StringTensor)
    assert len(st_sub) == 3
    assert st_sub[0] == "exactly12byte"

    mask = st.is_inlined_mask()
    assert mask == [True, False, True, False, True, False, True]
    assert st.inlined_count == 4
    assert st.heap_count == 3
    assert st.total_bytes() == sum(len(s.encode("utf-8")) for s in samples)

    st_pref = st.filter_by_prefix("pref_")
    assert st_pref.to_list() == ["pref_alpha", "pref_beta_very_long_sentence_with_prefix"]

    st_len = st.filter_by_length(min_len=5, max_len=12)
    assert st_len.to_list() == ["short", "12_bytes_ok!", "pref_alpha"]

    hashes = st.hash_all()
    assert len(hashes) == len(samples)
    assert hashes[0] != hashes[1]

    offsets = st.to_offsets()
    payload = st.to_payload()
    assert len(offsets) == len(samples) + 1
    assert offsets[0] == 0
    assert offsets[-1] == len(payload)
    for i in range(len(samples)):
        assert payload[offsets[i] : offsets[i + 1]].decode("utf-8") == samples[i]

    st_reconstructed = StringTensor.from_gpu_buffers(offsets, payload)
    assert st_reconstructed.to_list() == samples
    assert st_reconstructed == st

    try:
        StringTensor.from_gpu_buffers(b"\x00\x01\x02", b"abc")
        raise AssertionError("Expected ValueError for corrupted GPU offsets buffer")
    except ValueError as exc:
        assert "multiple of 4" in str(exc)

    import numpy as np

    off_arr, pay_arr = st.to_numpy_gpu()
    assert isinstance(off_arr, np.ndarray) and off_arr.dtype == np.uint32
    assert isinstance(pay_arr, np.ndarray) and pay_arr.dtype == np.uint8
    assert StringTensor.from_gpu_buffers(off_arr, pay_arr) == st

    try:
        import torch

        off_t, pay_t = st.to_torch_gpu()
        assert isinstance(off_t, torch.Tensor) and off_t.dtype == torch.int32
        assert isinstance(pay_t, torch.Tensor) and pay_t.dtype == torch.uint8
        assert StringTensor.from_gpu_buffers(off_t, pay_t) == st

        for empty_items in ([], [""]):
            eo, ep = StringTensor(empty_items).to_torch_gpu()
            assert ep.numel() == 0
            assert eo.numel() == len(empty_items) + 1
    except ImportError:
        pass

    try:
        import pyarrow as pa

        arrow_arr_zc = st.to_arrow(zero_copy=True)
        assert isinstance(arrow_arr_zc, pa.Array)
        assert arrow_arr_zc.to_pylist() == samples

        arrow_arr_sv = st.to_arrow(zero_copy=False)
        assert isinstance(arrow_arr_sv, pa.Array)
        assert StringTensor.from_arrow(arrow_arr_sv) == st
    except ImportError:
        pass


def test_fused_pipeline_integration() -> None:
    raw_batch = [
        "  Sample One with CRLF\r\n",
        "Sample One with CRLF",
        "Sample Two with proper prose text",
        "x" * 60,
        "",
    ]

    dedup = StreamDeduplicator()
    pipeline = FusedPipeline(
        trim_whitespace=True,
        normalize_newlines=True,
        min_byte_len=1,
        filter_quality=True,
        min_entropy=1.5,
        max_repetition_ratio=0.4,
        deduplicator=dedup,
        commit_dedup=True,
    )

    st = pipeline(raw_batch)
    assert isinstance(st, StringTensor)
    assert len(st) == 2
    assert st[0] == "Sample One with CRLF"
    assert st[1] == "Sample Two with proper prose text"
    assert dedup.count == 2
    assert dedup.is_seen("Sample One with CRLF") is True

    with tempfile.TemporaryDirectory() as tmp_dir:
        wal_file = Path(tmp_dir) / "fused.dedup"
        dedup_wal = StreamDeduplicator(wal_path=wal_file)
        pipeline_wal = FusedPipeline(
            trim_whitespace=True,
            normalize_newlines=True,
            deduplicator=dedup_wal,
            commit_dedup=True,
        )
        _ = pipeline_wal(["item_1", "item_2", "item_1"])
        assert wal_file.is_file()
        assert wal_file.stat().st_size == 16


def test_ecosystem_and_prefetch_pipeline() -> None:
    try:
        import polars as pl

        st = StringTensor(["row_alpha", "row_beta", "row_gamma"])
        series = st.to_polars()
        assert isinstance(series, pl.Series)
        assert series.to_list() == ["row_alpha", "row_beta", "row_gamma"]
    except ImportError:
        pass

    fused = FusedPipeline(trim_whitespace=True, normalize_newlines=True)
    collate = create_resumable_collate(fused_pipeline=fused, return_string_tensor=True)
    batch_raw = [{"text": "  alpha text\r\n"}, {"text": "beta text "}]
    collate_out = collate(batch_raw)
    assert isinstance(collate_out, StringTensor)
    assert collate_out.to_list() == ["alpha text", "beta text"]

    data_source = [f"streaming record #{i}" for i in range(25)]
    pipe = ResumablePipeline(
        source=data_source,
        fused_pipeline=fused,
        batch_size=5,
        prefetch_batches=2,
        return_string_tensor=True,
    )

    batches = list(pipe)
    assert len(batches) == 5
    assert isinstance(batches[0], StringTensor)
    assert len(batches[0]) == 5
    assert batches[0][0] == "streaming record #0"


def test_token_ring_buffer() -> None:
    import luminamoon as lm
    from luminamoon.data.adapters import TokenRingBuffer as AdapterTokenRingBuffer
    from luminamoon.data.ring_buffer import TokenRingBuffer as RingTokenRingBuffer

    assert AdapterTokenRingBuffer is TokenRingBuffer
    assert RingTokenRingBuffer is TokenRingBuffer
    assert lm.TokenRingBuffer is TokenRingBuffer
    assert not hasattr(lm, "LumiRingBuffer")

    token_chunks = [list(range(i * 20, (i + 1) * 20)) for i in range(10)]
    with TokenRingBuffer(
        source=token_chunks,
        capacity_tokens=64,
        auto_prefill_tokens=40,
        batch_size=2,
        seq_len=8,
        producer_chunk_tokens=16,
    ) as ring:
        assert ring.available_tokens >= 40
        batches = list(ring)
        assert len(batches) == 11
        x0, y0 = batches[0]
        assert tuple(x0.shape) == (2, 8)
        assert tuple(y0.shape) == (2, 8)
        if hasattr(x0, "tolist"):
            assert x0[0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
            assert y0[0].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
            assert x0[1].tolist() == [9, 10, 11, 12, 13, 14, 15, 16]
            assert y0[1].tolist() == [10, 11, 12, 13, 14, 15, 16, 17]

    short_source = [[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
    with TokenRingBuffer(
        source=short_source,
        capacity_tokens=100,
        auto_prefill_tokens=80,
        batch_size=1,
        seq_len=4,
    ) as short_ring:
        assert short_ring.is_eof is True
        assert short_ring.available_tokens == 10
        bx, by = short_ring.next_batch()
        assert tuple(bx.shape) == (1, 4)
        assert tuple(by.shape) == (1, 4)

    class WordLenTokenizer:
        def encode(self, text: str) -> list[int]:
            return [len(w) for w in text.split()]

    text_records = [{"text": "alpha beta gamma"}, {"text": "delta epsilon"}]
    with TokenRingBuffer(
        source=text_records,
        capacity_tokens=32,
        auto_prefill_tokens=6,
        batch_size=1,
        seq_len=5,
        tokenizer=WordLenTokenizer(),
        eos_token_id=999,
    ) as tok_ring:
        tx, ty = tok_ring.next_batch()
        toks_x = tx.reshape(-1).tolist()
        toks_y = ty.reshape(-1).tolist()
        assert toks_x == [5, 4, 5, 999, 5]
        assert toks_y == [4, 5, 999, 5, 7]

    def broken_stream():
        yield [1, 2, 3]
        raise RuntimeError("simulated network stream disconnect")

    try:
        with TokenRingBuffer(
            source=broken_stream(),
            capacity_tokens=64,
            auto_prefill_tokens=32,
        ):
            pass
        raise AssertionError("Expected RuntimeError from broken producer stream")
    except RuntimeError as exc:
        assert "simulated network stream disconnect" in str(exc)


def test_resumable_pipeline_fused_and_prefetch_zero_data_loss() -> None:
    import time

    # Auto-enable quality filter when quality bounds are passed to FusedPipeline
    fp_auto = FusedPipeline(min_chars=15, min_entropy=2.0)
    assert fp_auto(["aaa", "Natural prose sentence with plenty of characters."]).to_list() == [
        "Natural prose sentence with plenty of characters."
    ]
    fp_off = FusedPipeline(filter_quality=False, min_chars=15)
    assert fp_off(["aaa"]).to_list() == ["aaa"]

    # 1. FusedPipeline resume must not drop batch 2 from the 2*batch_size chunk
    with tempfile.TemporaryDirectory() as tmp_dir:
        meta = Path(tmp_dir) / "fused_resume.meta"
        src = [f"item_{i}" for i in range(6)]
        p1 = ResumablePipeline(source=src, checkpoint_path=meta, fused_pipeline=FusedPipeline(), batch_size=2)
        it1 = iter(p1)
        b1 = next(it1)
        p1.commit()
        assert b1 == ["item_0", "item_1"]

        p2 = ResumablePipeline(source=src, checkpoint_path=meta, fused_pipeline=FusedPipeline(), batch_size=2)
        assert list(p2) == [["item_2", "item_3"], ["item_4", "item_5"]]

    # 2. Prefetch + Deduplicator resume must only commit consumed batch_id, never prefetched batches
    with tempfile.TemporaryDirectory() as tmp_dir:
        meta = Path(tmp_dir) / "prefetch_resume.meta"
        wal = Path(tmp_dir) / "prefetch_resume.dedup"
        src = [f"doc_{i}" for i in range(6)]
        ded1 = StreamDeduplicator(wal_path=wal)
        p1 = ResumablePipeline(
            source=src,
            checkpoint_path=meta,
            deduplicator=ded1,
            batch_size=2,
            prefetch_batches=2,
        )
        it1 = iter(p1)
        b1 = next(it1)
        time.sleep(0.08)
        p1.commit()
        it1.close()
        assert b1 == ["doc_0", "doc_1"]
        assert ded1.count == 2

        ded2 = StreamDeduplicator(wal_path=wal)
        p2 = ResumablePipeline(source=src, checkpoint_path=meta, deduplicator=ded2, batch_size=2)
        assert list(p2) == [["doc_2", "doc_3"], ["doc_4", "doc_5"]]
