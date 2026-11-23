from __future__ import annotations

from pathlib import Path
import sys
import tempfile

ROOT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(ROOT_PYTHON) not in sys.path:
    sys.path.insert(0, str(ROOT_PYTHON))

from luminamoon.data import (
    MinHashDeduplicator,
    PIIMasker,
    QualityFilter,
    StateTracker,
    StreamCoordinate,
    StreamDeduplicator,
    TextCleaner,
    clean_batch,
    clean_text,
    data_cpu_isa,
    hash_record,
    jaccard_similarity,
    mask_pii,
    serialize_to_arrow,
    serialize_to_jsonl,
    serialize_to_parquet,
)


def test_cpu_isa_detection() -> None:
    isa = data_cpu_isa()
    assert isa in ("AVX-512", "AVX2", "NEON", "SCALAR")


def test_text_cleaner_features() -> None:
    cleaner = TextCleaner(
        trim_whitespace=True,
        normalize_newlines=True,
        min_byte_len=4,
        max_byte_len=30,
        truncate_over_max=True,
        json_extract_key="payload.content",
    )

    raw_json = '{"payload": {"content": "   First sample text\\r\\nwith newline   "}}'
    cleaned = cleaner.clean_single(raw_json)
    assert cleaned == "First sample text\nwith newline"

    short_json = '{"payload": {"content": "hi"}}'
    assert cleaner.clean_single(short_json) is None

    long_json = '{"payload": {"content": "This is a remarkably long sentence that must be truncated"}}'
    truncated = cleaner.clean_single(long_json)
    assert truncated is not None
    assert len(truncated.encode("utf-8")) <= 30

    batch = [
        '{"payload": {"content": "Row Alpha\\r\\n"}}',
        '{"payload": {"content": "Row Beta"}}',
        '{"payload": {"content": ""}}',
    ]
    batch_res = cleaner(batch)
    assert len(batch_res) == 2
    assert batch_res[0] == "Row Alpha"
    assert batch_res[1] == "Row Beta"

    assert clean_text("  hello world  ") == "hello world"
    assert clean_batch([" line1\r\n", "line2 "]) == ["line1", "line2"]


def test_stream_deduplicator_and_wal() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        wal_file = Path(tmp_dir) / "test.dedup"

        dedup = StreamDeduplicator(wal_path=wal_file)
        assert dedup.count == 0

        assert dedup.is_unique("sample_one") is True
        assert dedup.is_unique("sample_one") is False
        assert dedup.is_unique("sample_two") is True

        committed_count = dedup.commit()
        assert committed_count == 2
        assert dedup.count == 2
        assert dedup.is_seen("sample_one") is True
        assert dedup.is_seen("sample_three") is False

        assert wal_file.is_file()
        assert wal_file.stat().st_size == 8 + 2 * 16

        assert dedup.is_unique("sample_three") is True
        assert dedup.rollback() == 1
        assert dedup.is_seen("sample_three") is False

        dedup_reloaded = StreamDeduplicator(wal_path=wal_file)
        assert dedup_reloaded.count == 2
        assert dedup_reloaded.is_seen("sample_one") is True
        assert dedup_reloaded.is_seen("sample_two") is True

        batch = ["sample_one", "sample_new_1", "sample_two", "sample_new_2"]
        filtered = dedup_reloaded.filter_batch(batch)
        assert filtered == ["sample_new_1", "sample_new_2"]
        dedup_reloaded.commit()
        assert dedup_reloaded.count == 4

        h = hash_record("sample_one")
        assert isinstance(h, int)
        assert h > 0

        corrupt_wal = Path(tmp_dir) / "corrupt.dedup"
        corrupt_wal.write_bytes(b"12345")
        try:
            StreamDeduplicator(wal_path=corrupt_wal)
            raise AssertionError("Expected ValueError for corrupted WAL file length")
        except ValueError as exc:
            assert "Corrupted WAL" in str(exc)


def test_state_tracker_ping_pong() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        meta_file = Path(tmp_dir) / "checkpoint.meta"

        tracker = StateTracker(meta_file)
        initial_coord = tracker.read()
        assert initial_coord.shard_id == 0
        assert initial_coord.global_row_idx == 0

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
        assert meta_file.stat().st_size == 256

        tracker_recovered = StateTracker(meta_file)
        rec_coord = tracker_recovered.read()
        assert rec_coord.shard_id == 1
        assert rec_coord.byte_offset == 4096
        assert rec_coord.global_row_idx == 100

        tracker_recovered.update(
            shard_id=2,
            byte_offset=8192,
            global_row_idx=200,
            batch_id=2,
        )
        c2 = tracker_recovered.commit(batch_id=2)
        assert c2.shard_id == 2
        assert c2.global_row_idx == 200

        coord_bytes = c2.to_bytes()
        assert len(coord_bytes) == 32
        reconstructed = StreamCoordinate.from_bytes(coord_bytes)
        assert reconstructed.shard_id == 2
        assert reconstructed.byte_offset == 8192
        assert reconstructed.global_row_idx == 200


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

        pq_bytes = serialize_to_parquet(table, output_path=pq_path)
        assert len(pq_bytes) > 0
        assert pq_bytes.startswith(b"PAR1")
        assert pq_path.is_file()

        jsonl_bytes = serialize_to_jsonl(table, output_path=jsonl_path)
        assert len(jsonl_bytes) > 0
        assert b'"text":"row 1"' in jsonl_bytes
        assert jsonl_path.is_file()

        arrow_bytes = serialize_to_arrow(table, output_path=arrow_path)
        assert len(arrow_bytes) > 0
        assert arrow_bytes.startswith(b"ARROW1")
        assert arrow_path.is_file()


def test_quality_filter() -> None:
    qf_empty = QualityFilter()
    rep = "test test test test test test test test test test test test test test"
    assert qf_empty.is_high_quality(rep) is True

    qf = QualityFilter.for_pretraining()
    good = "Natural language processing models require cleaned text data and reliable training pipelines."
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

    qf_rep = QualityFilter(max_repetition_ratio=0.35)
    assert qf_rep.is_high_quality(good) is True
    assert qf_rep.is_high_quality(rep) is False
    assert qf_rep.is_high_quality(sym) is True


def test_minhash_deduplicator() -> None:
    minhash = MinHashDeduplicator(threshold=0.80)
    doc_1 = "The quick brown fox jumps over the lazy dog and runs across the green meadow."
    doc_1_near = "The quick brown fox jumps over the lazy dog and runs across the green meadow!"
    doc_2 = "Quantum mechanical entanglement produces non-local correlations in physics experiments."

    assert minhash.is_unique(doc_1) is True
    assert minhash.is_unique(doc_1_near) is False
    assert minhash.is_unique(doc_2) is True
    assert minhash.count == 2

    sim = jaccard_similarity(doc_1, doc_1_near)
    assert sim >= 0.80


def test_pii_masker() -> None:
    masker = PIIMasker(mask_emails=True, mask_ips=True, mask_api_keys=True, mask_cards=True)
    text = (
        "Contact me at user.support@example.org or server 192.168.1.100. "
        "Use secret key Bearer sk-1234567890abcdef1234567890abcdef and card 4111 1111 1111 1111."
    )
    masked = masker.mask(text)
    assert "[EMAIL]" in masked
    assert "user.support@example.org" not in masked
    assert "[IP]" in masked
    assert "192.168.1.100" not in masked
    assert "[API_KEY]" in masked
    assert "sk-1234567890abcdef" not in masked
    assert "[CARD]" in masked
    assert "4111 1111 1111 1111" not in masked

    assert mask_pii("admin@company.com") == "[EMAIL]"

    # Code-safety: C++/Rust namespaces, snake_case identifiers, non-Luhn IDs, and versions stay intact
    code_sample = "std::abs torch::cuda core::cell dead::beef skeleton_implementation hf_cache_directory_path order_id=1234567890123456 version=1.2.3.4"
    assert masker.mask(code_sample) == code_sample

    # Verify compressed IPv6, IPv4-mapped IPv6, no false positive on invalid octets, and '+' phone masking
    full_masker = PIIMasker(mask_emails=True, mask_ips=True, mask_api_keys=True, mask_cards=True, mask_phones=True)
    assert full_masker.mask("ver 999.999.999.999") == "ver 999.999.999.999"
    assert full_masker.mask("ipv6 2001:db8::1 and ::1 and ::ffff:192.168.1.5") == "ipv6 [IP] and [IP] and [IP]"
    assert full_masker.mask("call +1-555-123-4567 now") == "call [PHONE] now"


def test_dedup_nested_wal_and_batch_isolation_and_nested_json_serialize() -> None:
    import json
    from luminamoon.data._ffi import serialize_table_to_bytes

    with tempfile.TemporaryDirectory() as tmp_dir:
        nested_wal = Path(tmp_dir) / "nested" / "sub" / "stream.dedup"
        dedup = StreamDeduplicator(wal_path=nested_wal)
        assert dedup.is_unique("batch1_doc", batch_id=1) is True
        assert dedup.is_unique("batch2_doc", batch_id=2) is True
        assert dedup.commit(batch_id=1) == 1
        assert dedup.count == 1
        assert nested_wal.is_file()
        assert nested_wal.stat().st_size == 8 + 16

        reloaded = StreamDeduplicator(wal_path=nested_wal)
        assert reloaded.is_seen("batch1_doc") is True
        assert reloaded.is_seen("batch2_doc") is False

    raw_jsonl = serialize_to_jsonl({"nested": [{"a": 1, "b": "ok"}, ["x", "y"]]}).decode("utf-8").splitlines()
    assert json.loads(raw_jsonl[0]) == {"nested": {"a": 1, "b": "ok"}}
    assert json.loads(raw_jsonl[1]) == {"nested": ["x", "y"]}

    try:
        serialize_table_to_bytes({"text": ["a"]}, format="unsupported_fmt")
        raise AssertionError("Expected ValueError for unsupported format")
    except ValueError:
        pass


def test_data_v2_exports_and_line_limits() -> None:
    import luminamoon.data as data_mod

    for removed in ("LumiRingBuffer", "get_version", "list_versions", "set_version"):
        assert not hasattr(data_mod, removed), f"Removed symbol still exposed on luminamoon.data: {removed}"

    repo_root = Path(__file__).resolve().parents[1]
    data_py_root = repo_root / "python" / "luminamoon" / "data"
    assert not (data_py_root / "version.py").exists()
    for py_file in sorted(data_py_root.rglob("*.py")):
        lines = len(py_file.read_text(encoding="utf-8").splitlines())
        assert lines <= 450, f"{py_file.relative_to(repo_root)} has {lines} lines (limit 450)"

    data_rs_root = repo_root / "data" / "src"
    for rs_file in sorted(data_rs_root.rglob("*.rs")):
        lines = len(rs_file.read_text(encoding="utf-8").splitlines())
        assert lines <= 450, f"{rs_file.relative_to(repo_root)} has {lines} lines (limit 450)"

    tests_root = Path(__file__).resolve().parent
    for test_file in sorted(tests_root.glob("test_data_*.py")):
        lines = len(test_file.read_text(encoding="utf-8").splitlines())
        assert lines <= 650, f"{test_file.name} has {lines} lines (limit 650)"


def test_data_v2_edge_cases_and_fallback_thread_safety() -> None:
    from concurrent.futures import ThreadPoolExecutor
    import pytest
    from luminamoon.data.fallbacks.dedup import StreamDeduplicator as PyDedup
    from luminamoon.data.fallbacks.minhash import MinHashDeduplicator as PyMinHash
    from luminamoon.data.fallbacks.tracker import StateTracker as PyTracker
    from luminamoon.data.ring_buffer import TokenRingBuffer
    from luminamoon.data.string_tensor import StringTensor

    # [L21] filter_by_length(max_len=0) keeps only empty strings; max_len=None is unconstrained
    st = StringTensor(["", "a", "hello", "longer_string_12345"])
    assert st.filter_by_length(max_len=0).to_list() == [""]
    assert st.filter_by_length(min_len=1, max_len=0).to_list() == []
    assert st.filter_by_length(min_len=5, max_len=None).to_list() == ["hello", "longer_string_12345"]

    # [L22] RingBuffer seq_len <= 0 validation and empty dict input_ids EOS consistency
    rb = TokenRingBuffer(capacity_tokens=64, batch_size=2, seq_len=4, eos_token_id=999)
    assert rb._extract_tokens_from_item({"input_ids": []}) == []
    assert rb._extract_tokens_from_item([]) == []
    rb.push(list(range(20)))
    for bad_bs, bad_sl in ((0, 4), (2, 0), (-1, 4), (2, -3)):
        with pytest.raises(ValueError):
            rb.next_batch(batch_size=bad_bs, seq_len=bad_sl)
    assert rb.available_tokens == 20

    # [L23] Fallback MinHash respects explicit num_bands and implements jaccard_similarity
    mh = PyMinHash(threshold=0.8, num_perm=64, num_bands=16)
    assert mh.num_bands == 16 and mh.rows_per_band == 4
    with pytest.raises(ValueError, match="num_bands"):
        PyMinHash(num_perm=64, num_bands=7)
    s_same = mh.jaccard_similarity("luminamoon data pipeline", "luminamoon data pipeline")
    s_diff = mh.jaccard_similarity("aaaaaaaaaa", "zzzzzzzzzz")
    assert s_same == 1.0 and s_diff < 0.2

    # [M17] Thread-safe Python fallback StateTracker & StreamDeduplicator + concurrent save_wal
    with tempfile.TemporaryDirectory() as tmp_dir:
        meta_path = str(Path(tmp_dir) / "meta.bin")
        wal_path = str(Path(tmp_dir) / "dedup.wal")
        tracker = PyTracker(meta_path)
        dedup = PyDedup()

        def _worker(tid: int) -> None:
            for i in range(10):
                bid = tid * 100 + i + 1
                tracker.update(shard_id=tid, byte_offset=bid, batch_id=bid)
                tracker.commit(batch_id=bid)
                dedup.try_add(f"doc_{tid}_{i}".encode("utf-8"), batch_id=bid)
                dedup.commit(batch_id=bid)
            dedup.save_wal(wal_path)

        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(_worker, range(4)))

        assert dedup.count() == 40
        assert tracker.epoch() == 40
        reloaded = PyDedup()
        assert reloaded.load_wal(wal_path) > 0
