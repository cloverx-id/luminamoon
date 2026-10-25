from __future__ import annotations

from pathlib import Path
import sys
import tempfile

ROOT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(ROOT_PYTHON) not in sys.path:
    sys.path.insert(0, str(ROOT_PYTHON))

from luminamoon.data.fallbacks import (
    MinHashDeduplicator,
    PIIMasker,
    QualityFilter,
    RustStringTensor,
    StateTracker,
    StreamCoordinate,
    StreamDeduplicator,
    StringTensor,
    TextCleaner,
    clean_batch_records,
    clean_single_record,
    dataloader_cpu_isa,
    fused_pipeline_process_batch,
    postprocess_and_dedup_records,
    serialize_table_to_bytes,
    serialize_to_arrow_ipc,
    serialize_to_csv,
    serialize_to_jsonl,
    serialize_to_parquet,
    suggest_closest_matches,
)


def test_fallback_cleaner() -> None:
    cleaner = TextCleaner(
        trim_whitespace=True,
        normalize_newlines=True,
        min_byte_len=5,
        max_byte_len=25,
        truncate_over_max=True,
        json_extract_key="text",
    )
    raw = "   hello\r\nworld   "
    res = cleaner.clean(raw)
    assert res == "hello\nworld"

    # Test standalone functions
    assert clean_single_record("  test  ") == "test"
    assert clean_batch_records([" a ", " b "]) == ["a", "b"]


def test_fallback_quality_filter() -> None:
    qf = QualityFilter(
        min_chars=5,
        max_chars=100,
        min_words=2,
        min_entropy=1.5,
        max_repetition_ratio=0.5,
        max_symbol_ratio=0.2,
    )
    good_text = "This is a clean and high quality sentence for testing."
    assert qf.is_high_quality(good_text) is True

    bad_text = "a" * 50
    assert qf.is_high_quality(bad_text) is False


def test_fallback_deduplicator_and_wal() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        wal_path = Path(tmp_dir) / "test_fallback.wal"
        dedup = StreamDeduplicator(wal_path=str(wal_path))

        assert dedup.is_seen("item_a") is False
        assert dedup.is_unique("item_a") is True
        assert dedup.is_unique("item_a") is False

        committed = dedup.commit()
        assert committed == 1
        assert dedup.is_seen("item_a") is True

        # Test batch filter
        filtered = dedup.filter_batch(["item_a", "item_b", "item_c"])
        assert filtered == ["item_b", "item_c"]
        dedup.rollback()


def test_fallback_minhash() -> None:
    minhash = MinHashDeduplicator(num_perm=64, threshold=0.7)
    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "The quick brown fox jumps over the lazy dog"
    text3 = "Completely unrelated text with different vocabulary"

    sim_same = minhash.similarity(text1, text2)
    sim_diff = minhash.similarity(text1, text3)

    assert sim_same == 1.0
    assert sim_diff < 0.3


def test_fallback_pii_masker() -> None:
    masker = PIIMasker(mask_email=True, mask_ip=True, mask_phone=True)
    text = "Reach admin@example.com or 192.168.1.1 or 08123456789"
    masked = masker.mask(text)
    assert "[EMAIL]" in masked
    assert "[IP_ADDRESS]" in masked
    assert "[PHONE]" in masked


def test_fallback_coordinate_and_tracker() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        meta_path = Path(tmp_dir) / "state.meta"
        tracker = StateTracker(str(meta_path))

        coord = tracker.read()
        assert isinstance(coord, StreamCoordinate)
        assert coord.shard_id == 0

        tracker.update(shard_id=3, byte_offset=1024, global_row_idx=50, batch_id=1)
        committed = tracker.commit(batch_id=1)
        assert committed.shard_id == 3
        assert committed.global_row_idx == 50


def test_fallback_string_tensor() -> None:
    st = RustStringTensor(["alpha", "beta", "gamma"])
    assert len(st) == 3
    assert st[0] == "alpha"
    assert st.to_list() == ["alpha", "beta", "gamma"]


def test_fallback_fused_pipeline() -> None:
    records = ["  first line\r\n  ", "second line", ""]
    st = fused_pipeline_process_batch(
        raw_lines=records,
        trim_whitespace=True,
        normalize_newlines=True,
        min_byte_len=3,
    )
    assert len(st) == 2
    assert st[0] == "first line"


def test_fallback_helpers() -> None:
    assert dataloader_cpu_isa() in ("AVX-512", "AVX2", "NEON", "SCALAR")
    matches = suggest_closest_matches("target", ["target_one", "tar", "unrelated"])
    assert isinstance(matches, list)


def test_fallback_serialization() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        jsonl_path = str(Path(tmp_dir) / "test.jsonl")
        csv_path = str(Path(tmp_dir) / "test.csv")

        strings = ["record_1", "record_2"]
        n_jsonl = serialize_to_jsonl(strings, jsonl_path)
        assert n_jsonl == 2
        assert Path(jsonl_path).is_file()

        n_csv = serialize_to_csv(strings, csv_path)
        assert n_csv == 2
        assert Path(csv_path).is_file()
