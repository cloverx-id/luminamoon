from __future__ import annotations

from pathlib import Path
import sys
import tempfile

# Ensure python workspace is on sys.path
ROOT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(ROOT_PYTHON) not in sys.path:
    sys.path.insert(0, str(ROOT_PYTHON))

from luminamoon.data import (
    MinHashDeduplicator,
    PIIMasker,
    QualityFilter,
    ResumablePipeline,
    StateTracker,
    StreamCoordinate,
    StreamDeduplicator,
    StringTensor,
    TextCleaner,
    clean_batch,
    clean_text,
    create_resumable_collate,
    data_cpu_isa,
    filter_quality,
    hash_record,
    is_high_quality,
    jaccard_similarity,
    mask_pii,
    serialize_to_arrow,
    serialize_to_jsonl,
    serialize_to_parquet,
)


def test_cpu_isa_detection() -> None:
    isa = data_cpu_isa()
    assert isa in ("AVX-512", "AVX2", "NEON", "SCALAR")
    print(f"[TEST PASS] CPU ISA detected: {isa}")


def test_text_cleaner_features() -> None:
    cleaner = TextCleaner(
        trim_whitespace=True,
        normalize_newlines=True,
        min_byte_len=4,
        max_byte_len=30,
        truncate_over_max=True,
        json_extract_key="payload.content",
    )

    # 1. JSON extraction + CRLF + trim
    raw_json = '{"payload": {"content": "   First sample text\\r\\nwith newline   "}}'
    cleaned = cleaner.clean_single(raw_json)
    assert cleaned == "First sample text\nwith newline"

    # 2. Too short item should be filtered out
    short_json = '{"payload": {"content": "hi"}}'
    assert cleaner.clean_single(short_json) is None

    # 3. Truncation over max
    long_json = '{"payload": {"content": "This is a remarkably long sentence that must be truncated"}}'
    truncated = cleaner.clean_single(long_json)
    assert truncated is not None
    assert len(truncated.encode("utf-8")) <= 30

    # 4. Batch cleaning via __call__
    batch = [
        '{"payload": {"content": "Row Alpha\\r\\n"}}',
        '{"payload": {"content": "Row Beta"}}',
        '{"payload": {"content": ""}}',
    ]
    batch_res = cleaner(batch)
    assert len(batch_res) == 2
    assert batch_res[0] == "Row Alpha"
    assert batch_res[1] == "Row Beta"

    # 5. Module convenience functions
    assert clean_text("  hello world  ") == "hello world"
    assert clean_batch([" line1\r\n", "line2 "]) == ["line1", "line2"]
    print("[TEST PASS] TextCleaner verified.")


def test_stream_deduplicator_and_wal() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        wal_file = Path(tmp_dir) / "test.dedup"

        # 1. Basic unique and staging
        dedup = StreamDeduplicator(wal_path=wal_file)
        assert dedup.count == 0

        assert dedup.is_unique("sample_one") is True
        assert dedup.is_unique("sample_one") is False  # already staged
        assert dedup.is_unique("sample_two") is True

        # Commit batch
        committed_count = dedup.commit()
        assert committed_count == 2
        assert dedup.count == 2
        assert dedup.is_seen("sample_one") is True
        assert dedup.is_seen("sample_three") is False

        # Verify WAL file was created and written
        assert wal_file.is_file()
        assert wal_file.stat().st_size == 16  # 2 * 8 bytes

        # 2. Test rollback
        assert dedup.is_unique("sample_three") is True
        assert dedup.rollback() == 1
        assert dedup.is_seen("sample_three") is False

        # 3. Test reload in a new instance (simulating crash recovery)
        dedup_reloaded = StreamDeduplicator(wal_path=wal_file)
        assert dedup_reloaded.count == 2
        assert dedup_reloaded.is_seen("sample_one") is True
        assert dedup_reloaded.is_seen("sample_two") is True

        # 4. Filter batch
        batch = ["sample_one", "sample_new_1", "sample_two", "sample_new_2"]
        filtered = dedup_reloaded.filter_batch(batch)
        assert filtered == ["sample_new_1", "sample_new_2"]
        dedup_reloaded.commit()
        assert dedup_reloaded.count == 4

        # Hash record
        h = hash_record("sample_one")
        assert isinstance(h, int)
        assert h > 0
        print("[TEST PASS] StreamDeduplicator & WAL verified.")


def test_state_tracker_ping_pong() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        meta_file = Path(tmp_dir) / "checkpoint.meta"

        # 1. Fresh state tracker
        tracker = StateTracker(meta_file)
        initial_coord = tracker.read()
        assert initial_coord.shard_id == 0
        assert initial_coord.global_row_idx == 0

        # 2. Update and commit batch 1
        tracker.update(
            shard_id=1,
            byte_offset=4096,
            row_group_idx=2,
            page_idx=5,
            row_offset_in_page=32,
            global_row_idx=100,
            batch_id=1,
        )
        c1 = tracker.commit(batch_id=1)
        assert c1.shard_id == 1
        assert c1.byte_offset == 4096
        assert c1.global_row_idx == 100

        # File size must be exactly 128 bytes (Slot A 64B + Slot B 64B)
        assert meta_file.stat().st_size == 128

        # 3. Simulate recovery on new tracker instance
        tracker_recovered = StateTracker(meta_file)
        rec_coord = tracker_recovered.read()
        assert rec_coord.shard_id == 1
        assert rec_coord.byte_offset == 4096
        assert rec_coord.global_row_idx == 100

        # 4. Commit batch 2 (toggles to Slot B)
        tracker_recovered.update(
            shard_id=2,
            byte_offset=8192,
            global_row_idx=200,
            batch_id=2,
        )
        c2 = tracker_recovered.commit(batch_id=2)
        assert c2.shard_id == 2
        assert c2.global_row_idx == 200

        # 5. StreamCoordinate byte roundtrip
        coord_bytes = c2.to_bytes()
        assert len(coord_bytes) == 32
        reconstructed = StreamCoordinate.from_bytes(coord_bytes)
        assert reconstructed.shard_id == 2
        assert reconstructed.byte_offset == 8192
        assert reconstructed.global_row_idx == 200

        print("[TEST PASS] StateTracker Ping-Pong verified.")


def test_serialization() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        pq_path = Path(tmp_dir) / "out.parquet"
        jsonl_path = Path(tmp_dir) / "out.jsonl"
        arrow_path = Path(tmp_dir) / "out.arrow"

        table = {
            "text": ["row 1", "row 2", "row 3"],
            "score": [0.99, 0.85, 0.72],
            "label": [1, 0, 1],
        }

        # 1. Parquet
        pq_bytes = serialize_to_parquet(table, output_path=pq_path)
        assert len(pq_bytes) > 0
        assert pq_bytes.startswith(b"PAR1")
        assert pq_path.is_file()

        # 2. JSONL
        jsonl_bytes = serialize_to_jsonl(table, output_path=jsonl_path)
        assert len(jsonl_bytes) > 0
        assert b'"text":"row 1"' in jsonl_bytes
        assert jsonl_path.is_file()

        # 3. Arrow IPC
        arrow_bytes = serialize_to_arrow(table, output_path=arrow_path)
        assert len(arrow_bytes) > 0
        assert arrow_bytes.startswith(b"ARROW1")
        assert arrow_path.is_file()

        print("[TEST PASS] Serialization (Parquet, JSONL, Arrow) verified.")


def test_quality_filter() -> None:
    # 1. Default QualityFilter without rules: Zero silent dropping!
    qf_empty = QualityFilter()
    rep = "test test test test test test test test test test test test test test"
    assert qf_empty.is_high_quality(rep) is True  # No consent given -> passes!

    # 2. QualityFilter.for_pretraining(): Explicit LLM pre-training heuristics
    qf = QualityFilter.for_pretraining()
    good = "Natural language processing models require high quality cleaned data and robust systems."
    assert qf.is_high_quality(good) is True

    metrics = qf.metrics(good)
    assert metrics["is_passed"] is True
    assert metrics["entropy"] > 2.5
    assert metrics["word_count"] > 5

    assert qf.is_high_quality(rep) is False

    sym = "!@#$%^&*()_+{}[]:;\"'<>?,./" * 3
    assert qf.is_high_quality(sym) is False

    batch = [good, rep, sym, "Another valid sentence for deep learning and neural network training."]
    filtered = qf.filter_batch(batch)
    assert len(filtered) == 2
    assert filtered[0] == good

    # 3. Targeted rule: only repetition ratio
    qf_rep = QualityFilter(max_repetition_ratio=0.35)
    assert qf_rep.is_high_quality(good) is True
    assert qf_rep.is_high_quality(rep) is False
    assert qf_rep.is_high_quality(sym) is True  # Only repetition rule active!

    print("[TEST PASS] QualityFilter verified.")


def test_minhash_deduplicator() -> None:
    minhash = MinHashDeduplicator(threshold=0.80)
    doc_1 = "The quick brown fox jumps over the lazy dog and runs across the green meadow."
    doc_1_near = "The quick brown fox jumps over the lazy dog and runs across the green meadow!"
    doc_2 = "Quantum mechanical entanglement produces non-local correlations in physics experiments."

    assert minhash.is_unique(doc_1) is True
    assert minhash.is_unique(doc_1_near) is False  # near-duplicate detected and dropped!
    assert minhash.is_unique(doc_2) is True
    assert minhash.count == 2

    sim = jaccard_similarity(doc_1, doc_1_near)
    assert sim >= 0.80
    print("[TEST PASS] MinHashDeduplicator verified.")


def test_pii_masker() -> None:
    masker = PIIMasker(mask_emails=True, mask_ips=True, mask_api_keys=True, mask_cards=True)
    text = (
        "Contact me at user.support@example.org or server 192.168.1.100. "
        "Use secret key Bearer sk-1234567890abcdef1234567890abcdef and card 4111 2222 3333 4444."
    )
    masked = masker.mask(text)
    assert "[EMAIL]" in masked
    assert "user.support@example.org" not in masked
    assert "[IP]" in masked
    assert "192.168.1.100" not in masked
    assert "[API_KEY]" in masked
    assert "sk-1234567890abcdef" not in masked
    assert "[CARD]" in masked
    assert "4111 2222 3333 4444" not in masked

    assert mask_pii("admin@company.com") == "[EMAIL]"
    print("[TEST PASS] PIIMasker verified.")


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
        {"text": "A high quality training sample with email test@example.com."},  # duplicate!
        {"text": "spam spam spam spam spam spam spam spam spam spam spam spam"},  # low quality!
    ]

    res = collate(batch_items)
    assert "input_ids" in res
    assert len(res["input_ids"]) == 1
    assert "[EMAIL]" in res["texts"][0]

    # Test ResumablePipeline streaming & commit
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipe_meta = Path(tmp_dir) / "pipe.meta"
        pipe_dedup = Path(tmp_dir) / "pipe.dedup"

        data_source = [
            "Valid sample 1 with content.",
            "Valid sample 2 with content.",
            "Valid sample 1 with content.",  # duplicate
            "spam spam spam spam spam spam spam spam spam spam spam",  # bad quality
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

        # Checkpoint exists
        assert pipe_meta.stat().st_size == 128

    print("[TEST PASS] Adapters collate and ResumablePipeline verified.")


def test_hf_style_dataset_simulation() -> None:
    """Simulates processing a streaming dataset with resume without re-tokenizing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        meta_file = Path(tmp_dir) / "sim.meta"
        wal_file = Path(tmp_dir) / "sim.dedup"

        # Mock dataset with shards and duplicated texts
        mock_shards = [
            ["Text 0", "Text 1", "Text 1", "Text 2"],  # Shard 0 (has duplicate)
            ["Text 3", "Text 4", "Text 2"],            # Shard 1 (Text 2 duplicate across shards)
            ["Text 5", "Text 6", "Text 7"],            # Shard 2
        ]

        cleaner = TextCleaner()
        dedup = StreamDeduplicator(wal_path=wal_file)
        tracker = StateTracker(meta_path=meta_file)

        processed_records: list[str] = []

        # Run Phase A: Process Shard 0 and crash at Shard 1
        for shard_idx, shard_data in enumerate(mock_shards):
            for row_idx, raw_sample in enumerate(shard_data):
                cleaned = cleaner.clean_single(raw_sample)
                if cleaned is None or not dedup.is_unique(cleaned):
                    continue

                processed_records.append(cleaned)
                tracker.update(shard_id=shard_idx, row_offset_in_page=row_idx, global_row_idx=len(processed_records))
                tracker.commit()
                dedup.commit()

            if shard_idx == 0:
                # "Simulated crash" right after Shard 0 is complete
                break

        assert processed_records == ["Text 0", "Text 1", "Text 2"]
        assert dedup.count == 3

        # Phase B: Resume from checkpoint
        resume_coord = tracker.read()
        resumed_shard = resume_coord.shard_id
        resumed_row = resume_coord.row_offset_in_page

        resumed_records: list[str] = []

        # Resume exactly where we left off: Shard 1
        for shard_idx, shard_data in enumerate(mock_shards):
            if shard_idx < resumed_shard:
                continue  # O(1) shard skip! Zero download, zero re-tokenization!

            for row_idx, raw_sample in enumerate(shard_data):
                if shard_idx == resumed_shard and row_idx <= resumed_row:
                    continue  # skip inside the active shard

                cleaned = cleaner.clean_single(raw_sample)
                if cleaned is None or not dedup.is_unique(cleaned):
                    continue

                resumed_records.append(cleaned)
                tracker.update(shard_id=shard_idx, row_offset_in_page=row_idx, global_row_idx=len(resumed_records))
                tracker.commit()
                dedup.commit()

        # Text 2 in Shard 1 was filtered out because dedup remembered it from Shard 0!
        assert resumed_records == ["Text 3", "Text 4", "Text 5", "Text 6", "Text 7"]
        print("[TEST PASS] HF-style streaming dataset resume simulation verified.")


def test_audit_edge_cases() -> None:
    # 1. IPv6 masking
    masker = PIIMasker(mask_ips=True)
    ipv6_text = "Connected to 2001:0db8:85a3:0000:0000:8a2e:0370:7334 securely."
    assert "[IP]" in masker(ipv6_text)
    assert "2001:0db8" not in masker(ipv6_text)

    # 2. Collate where every item is filtered out
    qf = QualityFilter(min_words=50)  # very high threshold
    collate = create_resumable_collate(quality_filter=qf)
    empty_result = collate([{"text": "too short text"}])
    assert empty_result == []

    # 3. ResumablePipeline calling commit before running
    with tempfile.TemporaryDirectory() as tmp_dir:
        meta = Path(tmp_dir) / "empty.meta"
        pipe = ResumablePipeline(source=[], checkpoint_path=meta)
        pipe.commit()  # must not raise exception even with 0 batches!

    # 4. MinHash LSH indexing over distinct documents
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

    # Near duplicate of topic 0 (only 1 char changed, >95% identical)
    near_topic_0 = "Astrophysics explores stars, galaxies, black holes, and the origins of cosmic radiation!"
    assert dedup.is_unique(near_topic_0) is False  # near duplicate caught!

    print("[TEST PASS] Audit edge cases verified.")


def test_string_tensor_cpu_and_gpu() -> None:
    samples = [
        "short",  # 5 bytes: inlined
        "exactly12byte",  # 13 bytes: heap pool
        "12_bytes_ok!",  # 12 bytes: inlined
        "this is a much longer sample text that exceeds 12 bytes",  # heap pool
        "pref_alpha",  # 10 bytes: inlined
        "pref_beta_very_long_sentence_with_prefix",  # heap pool
        "",  # 0 bytes: inlined
    ]

    # 1. CPU Layout (Pola 2 - Umbra / DuckDB / Arrow 2.0 StringView)
    st = StringTensor(samples)
    assert len(st) == 7
    assert st[0] == "short"
    assert st[1] == "exactly12byte"
    assert st[-1] == ""
    assert st.to_list() == samples

    # Test slice
    st_sub = st[1:4]
    assert isinstance(st_sub, StringTensor)
    assert len(st_sub) == 3
    assert st_sub[0] == "exactly12byte"

    # Inlining mask check (<= 12 bytes inlined in 16-byte struct without heap deref)
    mask = st.is_inlined_mask()
    assert mask[0] is True  # "short" (5B)
    assert mask[1] is False  # "exactly12byte" (13B)
    assert mask[2] is True  # "12_bytes_ok!" (12B)
    assert mask[3] is False  # >12B
    assert mask[4] is True  # "pref_alpha" (10B)
    assert mask[5] is False  # >12B
    assert mask[6] is True  # "" (0B)

    assert st.inlined_count == 4
    assert st.heap_count == 3
    assert st.total_bytes() == sum(len(s.encode("utf-8")) for s in samples)

    # Prefix filtering using SIMD 4-byte register check
    st_pref = st.filter_by_prefix("pref_")
    assert st_pref.to_list() == ["pref_alpha", "pref_beta_very_long_sentence_with_prefix"]

    # Length filtering
    st_len = st.filter_by_length(min_len=5, max_len=12)
    assert st_len.to_list() == ["short", "12_bytes_ok!", "pref_alpha"]

    # Parallel XXH3-64 hashing without GIL
    hashes = st.hash_all()
    assert len(hashes) == len(samples)
    assert hashes[0] != hashes[1]

    # 2. GPU Layout (Pola 1 - Contiguous Offset + Payload Buffer)
    offsets = st.to_offsets()
    payload = st.to_payload()
    assert len(offsets) == len(samples) + 1
    assert offsets[0] == 0
    assert offsets[-1] == len(payload)
    for i in range(len(samples)):
        assert payload[offsets[i]:offsets[i + 1]].decode("utf-8") == samples[i]

    # Reconstruct from raw buffers
    st_reconstructed = StringTensor.from_gpu_buffers(offsets, payload)
    assert st_reconstructed.to_list() == samples
    assert st_reconstructed == st

    # 3. NumPy interop
    import numpy as np
    off_arr, pay_arr = st.to_numpy_gpu()
    assert isinstance(off_arr, np.ndarray) and off_arr.dtype == np.uint32
    assert isinstance(pay_arr, np.ndarray) and pay_arr.dtype == np.uint8
    st_from_np = StringTensor.from_gpu_buffers(off_arr, pay_arr)
    assert st_from_np == st

    # 4. PyTorch GPU-ready tensor interop (Optional)
    try:
        import torch
        off_t, pay_t = st.to_torch_gpu()
        assert isinstance(off_t, torch.Tensor) and off_t.dtype == torch.int32
        assert isinstance(pay_t, torch.Tensor) and pay_t.dtype == torch.uint8
        st_from_torch = StringTensor.from_gpu_buffers(off_t, pay_t)
        assert st_from_torch == st
    except ImportError:
        pass

    # 5. Apache Arrow interop (Zero-Copy & StringView) (Optional)
    try:
        import pyarrow as pa
        arrow_arr_zc = st.to_arrow(zero_copy=True)
        assert isinstance(arrow_arr_zc, pa.Array)
        assert arrow_arr_zc.to_pylist() == samples

        arrow_arr_sv = st.to_arrow(zero_copy=False)
        assert isinstance(arrow_arr_sv, pa.Array)
        st_from_arrow = StringTensor.from_arrow(arrow_arr_sv)
        assert st_from_arrow == st
    except ImportError:
        pass

    print("[TEST PASS] StringTensor CPU (StringView) & GPU (Offset+Payload) dual-layout verified.")


def test_fused_pipeline_integration() -> None:
    from luminamoon.data import FusedPipeline, fused_process_batch

    raw_batch = [
        "  Sample One with CRLF\r\n",
        "Sample One with CRLF",  # Duplicate -> should be dropped
        "Sample Two with proper prose text",
        "x" * 60,  # Low entropy degenerate repetition -> should be dropped
        "",  # Empty -> should be dropped
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

    # Deduplicator committed 2 items
    assert dedup.count == 2
    assert dedup.is_seen("Sample One with CRLF") is True

    print("[TEST PASS] Fused single-pass pipeline verified.")


def test_ecosystem_and_prefetch_pipeline() -> None:
    from luminamoon.data import FusedPipeline, ResumablePipeline, StringTensor, create_resumable_collate
    import polars as pl

    # 1. Test native Polars zero-copy export (graceful if optional dep missing)
    try:
        import polars as pl

        st = StringTensor(["row_alpha", "row_beta", "row_gamma"])
        series = st.to_polars()
        assert isinstance(series, pl.Series)
        assert series.to_list() == ["row_alpha", "row_beta", "row_gamma"]
    except ImportError:
        pass

    # 2. Test create_resumable_collate with FusedPipeline & return_string_tensor
    fused = FusedPipeline(trim_whitespace=True, normalize_newlines=True)
    collate = create_resumable_collate(fused_pipeline=fused, return_string_tensor=True)
    batch_raw = [{"text": "  alpha text\r\n"}, {"text": "beta text "}]
    collate_out = collate(batch_raw)
    assert isinstance(collate_out, StringTensor)
    assert collate_out.to_list() == ["alpha text", "beta text"]

    # 3. Test ResumablePipeline with async prefetcher
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

    print("[TEST PASS] Ecosystem Polars, Fused Collate, and Async Prefetcher verified.")


if __name__ == "__main__":
    test_cpu_isa_detection()
    test_text_cleaner_features()
    test_stream_deduplicator_and_wal()
    test_state_tracker_ping_pong()
    test_serialization()
    test_quality_filter()
    test_minhash_deduplicator()
    test_pii_masker()
    test_string_tensor_cpu_and_gpu()
    test_fused_pipeline_integration()
    test_adapters_collate_and_pipeline()
    test_hf_style_dataset_simulation()
    test_ecosystem_and_prefetch_pipeline()
    test_audit_edge_cases()
    print("\nALL MODULAR & EXPANDED TESTS PASSED!")
