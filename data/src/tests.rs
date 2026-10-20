use std::fs;
use std::path::PathBuf;

use crate::coord::StreamCoordinate;
use crate::dedup::TwoStageDeduplicator;
use crate::metadata::DualMetadataManager;
use crate::postprocess::{PostProcessConfig, process_batch_parallel, process_single_record};
use crate::serialize::{
    DataValue, build_arrow_record_batch, classify_string_or_json,
    serialize_columns_to_arrow_ipc_bytes, serialize_columns_to_jsonl_bytes,
    serialize_columns_to_parquet_bytes,
};
use crate::simd_cpu::{contains_byte_fast, detect_cpu_isa, find_newline_fast};

fn temp_test_path(name: &str, ext: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "luminamoon_test_{}_{}.{}",
        name,
        std::process::id(),
        ext
    ));
    p
}

#[test]
fn test_stream_coordinate_roundtrip() {
    let c = StreamCoordinate {
        shard_id: 42,
        byte_offset: 1024 * 1024,
        row_group_idx: 3,
        page_idx: 7,
        row_offset_in_page: 128,
        global_row_idx: 99999,
    };
    let bytes = c.to_bytes();
    assert_eq!(bytes.len(), StreamCoordinate::BYTE_LEN);
    let decoded = StreamCoordinate::from_bytes(&bytes);
    assert_eq!(c, decoded);
}

#[test]
fn test_dual_metadata_ping_pong_and_dedup_wal() {
    let meta_path = temp_test_path("ping_pong_meta", "meta");
    let _ = fs::remove_file(&meta_path);
    let _ = fs::remove_file(format!("{}.dedup", meta_path.display()));

    {
        let mut mgr = DualMetadataManager::open(&meta_path).unwrap();
        assert_eq!(mgr.committed_batch_id(), 0);

        assert!(mgr.deduplicator.try_stage(1, b"line_alpha").is_some());
        assert!(mgr.deduplicator.try_stage(1, b"line_alpha").is_none());

        let c1 = StreamCoordinate {
            shard_id: 1,
            byte_offset: 512,
            row_group_idx: 0,
            page_idx: 0,
            row_offset_in_page: 0,
            global_row_idx: 10,
        };
        mgr.stage_batch_in_ram(1, c1);
        let committed = mgr.commit_gpu_batch(1).unwrap();
        assert_eq!(committed, c1);
        assert_eq!(mgr.committed_batch_id(), 1);
    }

    // Re-open and verify recovery
    {
        let mut mgr = DualMetadataManager::open(&meta_path).unwrap();
        assert_eq!(mgr.committed_batch_id(), 1);
        assert_eq!(mgr.committed_coordinate().shard_id, 1);
        assert_eq!(mgr.committed_coordinate().byte_offset, 512);

        // "line_alpha" must still be marked as committed
        assert!(mgr.deduplicator.try_stage(2, b"line_alpha").is_none());
        assert!(mgr.deduplicator.try_stage(2, b"line_beta").is_some());

        let c2 = StreamCoordinate {
            shard_id: 1,
            byte_offset: 1024,
            row_group_idx: 0,
            page_idx: 1,
            row_offset_in_page: 0,
            global_row_idx: 20,
        };
        mgr.stage_batch_in_ram(2, c2);
        mgr.commit_gpu_batch(2).unwrap();
    }

    let _ = fs::remove_file(&meta_path);
    let _ = fs::remove_file(format!("{}.dedup", meta_path.display()));
}

#[test]
fn test_two_stage_deduplicator() {
    let mut dedup = TwoStageDeduplicator::new();
    assert_eq!(dedup.committed_len(), 0);

    assert!(dedup.try_stage(1, b"apple").is_some());
    assert!(dedup.try_stage(1, b"apple").is_none());
    assert!(dedup.try_stage(1, b"banana").is_some());
    assert_eq!(dedup.staged_len(), 2);

    let committed = dedup.commit_batch(1);
    assert_eq!(committed.len(), 2);
    assert_eq!(dedup.committed_len(), 2);
    assert_eq!(dedup.staged_len(), 0);

    // Staging the same in batch 2 should be rejected
    assert!(dedup.try_stage(2, b"apple").is_none());
    assert!(dedup.try_stage(2, b"orange").is_some());

    // Rollback batch 2
    let rolled = dedup.rollback_batch(2);
    assert_eq!(rolled, 1);
    assert!(dedup.try_stage(3, b"orange").is_some());
}

#[test]
fn test_simd_cpu_detection_and_scans() {
    let isa = detect_cpu_isa();
    assert!(!isa.as_str().is_empty());

    let text = b"hello\r\nworld\nfrom\rsimd";
    assert!(contains_byte_fast(text, b'\r'));
    assert!(contains_byte_fast(text, b'\n'));
    assert!(!contains_byte_fast(text, b'z'));

    let pos = find_newline_fast(text);
    assert!(pos.is_some());
}

#[test]
fn test_postprocess_cleaning_and_json_extraction() {
    let cfg = PostProcessConfig {
        trim_whitespace: true,
        normalize_newlines: true,
        min_byte_len: 3,
        max_byte_len: 20,
        truncate_over_max: true,
        json_extract_key: Some("data.text".to_owned()),
    };

    let raw_json = r#"{"data": {"text": "  Hello\r\nWorld!   "}}"#;
    let cleaned = process_single_record(raw_json, &cfg);
    assert_eq!(cleaned, Some("Hello\nWorld!".to_owned()));

    // Truncation test
    let long_json = r#"{"data": {"text": "A very long sentence that will exceed twenty bytes"}}"#;
    let truncated = process_single_record(long_json, &cfg).unwrap();
    assert!(truncated.len() <= 20);

    // Parallel batch test
    let batch = vec![
        r#"{"data": {"text": "row 1\r\n"}}"#.to_owned(),
        r#"{"data": {"text": "row 2"}}"#.to_owned(),
        r#"{"data": {"text": ""}}"#.to_owned(), // filtered due to min_byte_len
    ];
    let batch_cleaned = process_batch_parallel(&batch, &cfg);
    assert_eq!(batch_cleaned.len(), 2);
    assert_eq!(batch_cleaned[0], "row 1");
    assert_eq!(batch_cleaned[1], "row 2");
}

#[test]
fn test_classify_string_or_json() {
    let s1 = "just plain text";
    assert_eq!(classify_string_or_json(s1), DataValue::Text(s1.to_owned()));

    let s2 = r#"{"key": "value", "id": 123}"#;
    assert_eq!(classify_string_or_json(s2), DataValue::Json(s2.to_owned()));
}

#[test]
fn test_serialization_parquet_and_jsonl() {
    let columns = vec![
        (
            "text".to_owned(),
            vec![
                DataValue::Text("first".to_owned()),
                DataValue::Text("second".to_owned()),
            ],
        ),
        (
            "score".to_owned(),
            vec![DataValue::Float(0.95), DataValue::Float(0.42)],
        ),
        (
            "label".to_owned(),
            vec![DataValue::Int(1), DataValue::Int(0)],
        ),
    ];

    let (schema, batch) = build_arrow_record_batch(&columns, false).unwrap();
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(schema.fields().len(), 3);

    // Parquet serialization
    let pq_bytes = serialize_columns_to_parquet_bytes(&columns, false).unwrap();
    assert!(!pq_bytes.is_empty());
    assert_eq!(&pq_bytes[0..4], b"PAR1");

    // Arrow IPC serialization
    let ipc_bytes = serialize_columns_to_arrow_ipc_bytes(&columns, false).unwrap();
    assert!(!ipc_bytes.is_empty());
    assert_eq!(&ipc_bytes[0..6], b"ARROW1");

    // JSONL serialization
    let jsonl_bytes = serialize_columns_to_jsonl_bytes(&columns).unwrap();
    assert!(!jsonl_bytes.is_empty());
    let jsonl_str = String::from_utf8(jsonl_bytes).unwrap();
    assert!(jsonl_str.contains("\"text\":\"first\""));
    assert!(jsonl_str.contains("\"score\":0.95"));
}

#[test]
fn test_quality_filter() {
    use crate::quality::{QualityConfig, evaluate_quality};

    let cfg = QualityConfig {
        min_chars: 20,
        max_chars: 0,
        min_words: 3,
        min_entropy: 2.5,
        max_repetition_ratio: 0.35,
        max_symbol_ratio: 0.30,
    };

    // 1. High quality normal text
    let good_text =
        "The quick brown fox jumps over the lazy dog and runs across the wide green meadow.";
    let res = evaluate_quality(good_text, &cfg);
    assert!(res.is_passed);
    assert!(res.entropy > 2.5);
    assert!(res.repetition_ratio < 0.2);
    assert!(res.symbol_ratio < 0.1);

    // 2. Degenerate repetition
    let repeated =
        "spam spam spam spam spam spam spam spam spam spam spam spam spam spam spam spam";
    let res_rep = evaluate_quality(repeated, &cfg);
    assert!(!res_rep.is_passed);
    assert!(res_rep.repetition_ratio > 0.35);

    // 3. Code / symbol dump
    let symbol_dump = "{{{[(((@@@###$$$%%%^^^&&&***)))}}}];;;===+++///<<<>>>!!!???";
    let res_sym = evaluate_quality(symbol_dump, &cfg);
    assert!(!res_sym.is_passed);
}

#[test]
fn test_minhash_fuzzy_deduplicator() {
    use crate::minhash::MinHashDeduplicator;

    let mut dedup = MinHashDeduplicator::new(0.80, 4, 64);

    let doc_a =
        "Artificial intelligence and deep learning models require efficient data pipelines.";
    // Near duplicate (just 1 typo/word difference)
    let doc_a_near =
        "Artificial intelligence and deep learning models require efficient data pipelining.";
    // Completely different document
    let doc_b = "Quantum mechanical systems exhibit wave particle duality in atomic physics.";

    assert!(dedup.is_unique(doc_a));
    // doc_a_near should be caught as near-duplicate (> 80% similarity)
    assert!(!dedup.is_unique(doc_a_near));
    // doc_b is different, should pass
    assert!(dedup.is_unique(doc_b));
    assert_eq!(dedup.count(), 2);
}

#[test]
fn test_string_tensor_cpu_and_gpu() {
    use crate::string_tensor::StringTensor;

    let raw = vec![
        "apple",                                        // 5B <= 12B (inlined)
        "cat",                                          // 3B <= 12B (inlined)
        "twelve_bytes",                                 // 12B <= 12B (inlined boundary)
        "thirteen_bytes!",                              // 15B > 12B (referenced into pool)
        "The quick brown fox jumps over the lazy dog.", // 44B > 12B
    ];

    let st = StringTensor::from_strings(&raw);
    assert_eq!(st.len(), 5);

    // Verify inlining
    assert!(st.views[0].is_inlined());
    assert!(st.views[1].is_inlined());
    assert!(st.views[2].is_inlined());
    assert!(!st.views[3].is_inlined());
    assert!(!st.views[4].is_inlined());

    // Verify string content recovery
    assert_eq!(st.get(0).unwrap(), "apple");
    assert_eq!(st.get(1).unwrap(), "cat");
    assert_eq!(st.get(2).unwrap(), "twelve_bytes");
    assert_eq!(st.get(3).unwrap(), "thirteen_bytes!");
    assert_eq!(
        st.get(4).unwrap(),
        "The quick brown fox jumps over the lazy dog."
    );

    // Fast equality check
    let v0 = st.views[0];
    let v1 = st.views[1];
    assert!(v0.fast_eq(&v0, &st.pool));
    assert!(!v0.fast_eq(&v1, &st.pool));

    // Convert to GPU buffers: (offsets, payload)
    let (offsets, payload) = st.to_gpu_buffers();
    assert_eq!(offsets.len(), 6);
    assert_eq!(offsets[0], 0);
    assert_eq!(offsets[5] as usize, payload.len());

    // Reconstruct from GPU buffers
    let st_gpu = StringTensor::from_gpu_buffers(&offsets, &payload);
    assert_eq!(st_gpu.len(), 5);
    assert_eq!(st_gpu.to_vec(), raw);

    // Filtering
    let long_only = st.filter_by_length(13, 0);
    assert_eq!(long_only.len(), 2);
    assert_eq!(long_only.get(0).unwrap(), "thirteen_bytes!");

    let prefix_filtered = st.filter_by_prefix("thirt");
    assert_eq!(prefix_filtered.len(), 1);
    assert_eq!(prefix_filtered.get(0).unwrap(), "thirteen_bytes!");

    // Parallel hashing
    let hashes = st.hash_all_parallel();
    assert_eq!(hashes.len(), 5);
    assert!(hashes[0] != hashes[1]);
}

#[test]
fn test_fused_pipeline_core() {
    let raw = vec![
        "  Valid prose sentence for LLM training alpha\r\n".to_string(),
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(), // degenerate repetition -> reject
        "Valid prose sentence for LLM training beta".to_string(),
        "Valid prose sentence for LLM training alpha\r\n".to_string(), // duplicate -> reject
        "".to_string(),                                                // empty -> reject
    ];

    let post_cfg = PostProcessConfig {
        min_byte_len: 1,
        ..Default::default()
    };
    let qual_cfg = crate::quality::QualityConfig {
        min_entropy: 1.5,
        max_repetition_ratio: 0.4,
        ..Default::default()
    };
    let mut dedup = TwoStageDeduplicator::new();

    let mut survivors = Vec::new();
    for item in raw {
        if let Some(cleaned) = process_single_record(&item, &post_cfg) {
            let eval = crate::quality::evaluate_quality(&cleaned, &qual_cfg);
            if eval.is_passed && dedup.try_stage(1, cleaned.as_bytes()).is_some() {
                survivors.push(cleaned);
            }
        }
    }

    assert_eq!(survivors.len(), 2);
    assert_eq!(survivors[0], "Valid prose sentence for LLM training alpha");
    assert_eq!(survivors[1], "Valid prose sentence for LLM training beta");

    let st = crate::string_tensor::StringTensor::from_strings(survivors);
    assert_eq!(st.len(), 2);
    assert_eq!(
        st.get(0).unwrap(),
        "Valid prose sentence for LLM training alpha"
    );
    assert_eq!(
        st.get(1).unwrap(),
        "Valid prose sentence for LLM training beta"
    );
}
