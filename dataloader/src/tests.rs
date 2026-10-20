use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use crate::coord::StreamCoordinate;
use crate::dedup::TwoStageDeduplicator;
use crate::error::DataLoaderError;
use crate::metadata::DualMetadataManager;
use crate::postprocess::{PostProcessConfig, process_batch_parallel};
use crate::source_hf::{RemoteLineSource, RemoteParquetSource, resolve_hf_uri};
use crate::source_local::{LocalMmapLineSource, LocalParquetSource};

fn temp_path(name: &str, ext: &str) -> PathBuf {
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
fn test_dual_metadata_ram_vs_gpu_crash_recovery() {
    let path = temp_path("crash_recovery", "meta");
    let _ = fs::remove_file(&path);
    let _ = fs::remove_file(format!("{}.dedup", path.display()));

    {
        let mut mgr = DualMetadataManager::open(&path).unwrap();

        assert!(mgr.deduplicator.try_stage(1, b"row_1").is_some());
        assert!(mgr.deduplicator.try_stage(1, b"row_1").is_none());
        let c1 = StreamCoordinate {
            shard_id: 0,
            byte_offset: 4096,
            row_group_idx: 0,
            page_idx: 1,
            row_offset_in_page: 0,
            global_row_idx: 64,
        };
        mgr.stage_batch_in_ram(1, c1);
        mgr.commit_gpu_batch(1).unwrap();

        assert!(
            mgr.deduplicator
                .try_stage(2, b"row_2_only_in_ram")
                .is_some()
        );
        let c2 = StreamCoordinate {
            shard_id: 0,
            byte_offset: 8192,
            row_group_idx: 0,
            page_idx: 2,
            row_offset_in_page: 0,
            global_row_idx: 128,
        };
        mgr.stage_batch_in_ram(2, c2);
        assert_eq!(mgr.in_flight_coordinate().byte_offset, 8192);
        assert_eq!(mgr.committed_coordinate().byte_offset, 4096);
    }

    {
        let mut mgr = DualMetadataManager::open(&path).unwrap();
        assert_eq!(mgr.committed_batch_id(), 1);
        assert_eq!(mgr.committed_coordinate().byte_offset, 4096);
        assert_eq!(mgr.in_flight_coordinate().byte_offset, 4096);

        assert!(mgr.deduplicator.try_stage(2, b"row_1").is_none());
        assert!(
            mgr.deduplicator
                .try_stage(2, b"row_2_only_in_ram")
                .is_some()
        );
    }

    let _ = fs::remove_file(&path);
    let _ = fs::remove_file(format!("{}.dedup", path.display()));
}

#[test]
fn test_single_batch_rollback_and_corrupted_dedup_guard() {
    let path = temp_path("single_rollback_and_guard", "meta");
    let dedup_path = PathBuf::from(format!("{}.dedup", path.display()));
    let _ = fs::remove_file(&path);
    let _ = fs::remove_file(&dedup_path);

    {
        let mut mgr = DualMetadataManager::open(&path).unwrap();
        let c1 = StreamCoordinate {
            shard_id: 0,
            byte_offset: 2048,
            row_group_idx: 0,
            page_idx: 1,
            row_offset_in_page: 0,
            global_row_idx: 32,
        };
        let c2 = StreamCoordinate {
            shard_id: 0,
            byte_offset: 4096,
            row_group_idx: 0,
            page_idx: 2,
            row_offset_in_page: 0,
            global_row_idx: 64,
        };

        assert!(mgr.deduplicator.try_stage(1, b"batch_1_row").is_some());
        mgr.stage_batch_in_ram(1, c1);

        assert!(mgr.deduplicator.try_stage(2, b"batch_2_row").is_some());
        mgr.stage_batch_in_ram(2, c2);
        assert_eq!(mgr.in_flight_coordinate().byte_offset, 4096);

        let restored_coord = mgr.rollback_staged_batch(2).unwrap();
        assert_eq!(restored_coord.byte_offset, 2048);
        assert!(mgr.deduplicator.try_stage(1, b"batch_1_row").is_none());
        assert!(mgr.deduplicator.try_stage(3, b"batch_2_row").is_some());

        mgr.commit_gpu_batch(1).unwrap();
    }

    {
        let f = OpenOptions::new().write(true).open(&dedup_path).unwrap();
        f.set_len(0).unwrap();
    }

    assert!(matches!(
        DualMetadataManager::open(&path),
        Err(DataLoaderError::CorruptedMetadata)
    ));

    let _ = fs::remove_file(&path);
    let _ = fs::remove_file(&dedup_path);
}

#[test]
fn test_local_mmap_o1_skip_and_json_dedup() {
    let path = temp_path("mmap_jsonl", "jsonl");
    {
        let mut f = File::create(&path).unwrap();
        writeln!(f, "{{\"text\": \"alpha\"}}").unwrap();
        writeln!(f, "{{\"text\": \"alpha\"}}").unwrap();
        writeln!(f, "{{\"text\": \"beta\"}}").unwrap();
        writeln!(f, "{{\"text\": \"gamma\"}}").unwrap();
        writeln!(f, "{{\"text\": \"delta\"}}").unwrap();
    }

    let src = LocalMmapLineSource::open(0, &path).unwrap();
    let mut dedup = TwoStageDeduplicator::new();
    let cfg = PostProcessConfig {
        json_extract_key: Some("text".to_owned()),
        ..Default::default()
    };

    let b1 = src
        .read_batch(1, StreamCoordinate::default(), 2, &cfg, Some(&mut dedup))
        .unwrap();
    assert_eq!(b1.records, vec!["alpha", "beta"]);
    assert!(!b1.is_eof);

    let b2 = src
        .read_batch(2, b1.next_coord, 2, &cfg, Some(&mut dedup))
        .unwrap();
    assert_eq!(b2.records, vec!["gamma", "delta"]);
    assert!(b2.is_eof);

    let _ = fs::remove_file(&path);
}

#[test]
fn test_local_parquet_o1_row_skip() {
    let path = temp_path("local_pq", "parquet");
    {
        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None).unwrap();
        let arr = StringArray::from(vec!["r0", "r1", "r2", "r3", "r4"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    let src = LocalParquetSource::open(0, &path, "text").unwrap();
    let cfg = PostProcessConfig::default();

    let jump_coord = StreamCoordinate {
        shard_id: 0,
        byte_offset: 0,
        row_group_idx: 0,
        page_idx: 0,
        row_offset_in_page: 2,
        global_row_idx: 2,
    };

    let b = src.read_batch(1, jump_coord, 2, &cfg, None).unwrap();
    assert_eq!(b.records, vec!["r2", "r3"]);
    assert_eq!(b.next_coord.row_offset_in_page, 4);

    let par_res = process_batch_parallel(&["  hello\r\n ".to_owned()], &cfg);
    assert_eq!(par_res, vec!["hello"]);

    let _ = fs::remove_file(&path);
}

#[tokio::test]
async fn test_hf_uri_and_http_range_parquet_and_oversized_line_streaming() {
    let resolved = resolve_hf_uri(
        "hf://datasets/luminamoon/corpus@refs/convert/parquet/train/000.parquet",
    )
    .unwrap();
    assert_eq!(
        resolved,
        "https://huggingface.co/datasets/luminamoon/corpus/resolve/refs/convert/parquet/train/000.parquet"
    );

    let pq_path = temp_path("remote_pq", "parquet");
    {
        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let file = File::create(&pq_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None).unwrap();
        let arr = StringArray::from(vec!["net0", "net1", "net2", "net3"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
    let pq_payload = Arc::new(fs::read(&pq_path).unwrap());
    let _ = fs::remove_file(&pq_path);

    let long_line_1 = "A".repeat(150);
    let long_line_2 = "B".repeat(150);
    let jsonl_payload = Arc::new(format!("{long_line_1}\n{long_line_2}\n").into_bytes());

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server_pq = pq_payload.clone();
    let server_jsonl = jsonl_payload.clone();

    tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            let pq = server_pq.clone();
            let jl = server_jsonl.clone();
            tokio::spawn(async move {
                let mut buf = [0u8; 2048];
                let n = stream.read(&mut buf).await.unwrap_or(0);
                if n == 0 {
                    return;
                }
                let req_str = String::from_utf8_lossy(&buf[..n]);
                let is_jsonl = req_str
                    .lines()
                    .next()
                    .unwrap_or("")
                    .contains("/lines.jsonl");
                let data = if is_jsonl { jl } else { pq };

                let mut start = 0usize;
                let mut end = data.len() - 1;
                for line in req_str.lines() {
                    if let Some(r) = line
                        .strip_prefix("Range: bytes=")
                        .or_else(|| line.strip_prefix("range: bytes="))
                        && let Some((s, e)) = r.trim().split_once('-')
                    {
                        start = s.parse().unwrap_or(0);
                        end = e
                            .parse::<usize>()
                            .unwrap_or(data.len() - 1)
                            .min(data.len() - 1);
                    }
                }
                let slice = &data[start..=end];
                let header = format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\nConnection: close\r\n\r\n",
                    slice.len(),
                    start,
                    end,
                    data.len()
                );
                let _ = stream.write_all(header.as_bytes()).await;
                let _ = stream.write_all(slice).await;
            });
        }
    });

    let pq_url = format!("http://{addr}/train.parquet");
    let remote_pq = RemoteParquetSource::connect(0, &pq_url, "text", None)
        .await
        .unwrap();
    let skip_coord = StreamCoordinate {
        shard_id: 0,
        byte_offset: 0,
        row_group_idx: 0,
        page_idx: 0,
        row_offset_in_page: 2,
        global_row_idx: 2,
    };
    let batch = remote_pq
        .read_batch(1, skip_coord, 2, &PostProcessConfig::default(), None)
        .await
        .unwrap();
    assert_eq!(batch.records, vec!["net2", "net3"]);
    assert!(batch.is_eof);

    let jsonl_url = format!("http://{addr}/lines.jsonl");
    let remote_lines = RemoteLineSource::connect(0, &jsonl_url, None, 64)
        .await
        .unwrap();
    let line_batch = remote_lines
        .read_batch(
            1,
            StreamCoordinate::default(),
            2,
            &PostProcessConfig::default(),
            None,
        )
        .await
        .unwrap();
    assert_eq!(line_batch.records, vec![long_line_1, long_line_2]);
    assert!(line_batch.is_eof);
}

#[test]
fn test_preram_watermark_refill_and_uncommitted_crash_recovery() {
    use crate::preram::{PreRamConfig, PreRamDataLoader, ShardSourceKind};

    let data_path = temp_path("preram_lines", "txt");
    let meta_path = temp_path("preram_state", "meta");
    let _ = fs::remove_file(&data_path);
    let _ = fs::remove_file(&meta_path);
    let _ = fs::remove_file(format!("{}.dedup", meta_path.display()));

    {
        let mut f = File::create(&data_path).unwrap();
        for i in 0..100 {
            writeln!(f, "sample_line_{i:03}").unwrap();
        }
    }

    let cfg = PreRamConfig {
        capacity_batches: 10,
        refill_trigger: 3,
        batch_size: 2,
        deduplicate: true,
        auto_refill: true,
        download_workers: 2,
        postprocess: PostProcessConfig::default(),
        min_score_filter: None,
    };

    let batch3_expected: Vec<String>;
    {
        let loader = PreRamDataLoader::open(
            &meta_path,
            vec![ShardSourceKind::LocalLine {
                path: data_path.clone(),
            }],
            cfg.clone(),
        )
        .unwrap();

        let filled = loader.wait_for_refill().unwrap();
        assert_eq!(filled, 10);

        let b1 = loader.next_batch().unwrap().unwrap();
        assert_eq!(b1.batch_id, 1);
        assert_eq!(loader.buffered_batches(), 9);

        let b2 = loader.next_batch().unwrap().unwrap();
        assert_eq!(b2.batch_id, 2);
        assert_eq!(loader.buffered_batches(), 8);

        loader.commit_gpu_batch(b1.batch_id).unwrap();
        loader.commit_gpu_batch(b2.batch_id).unwrap();

        let b3 = loader.next_batch().unwrap().unwrap();
        assert_eq!(b3.batch_id, 3);
        batch3_expected = b3.records.clone();

        let refilled = loader.wait_for_refill().unwrap();
        assert_eq!(refilled, 10);
        assert!(
            loader.in_flight_coordinate().byte_offset
                > loader.committed_coordinate().byte_offset
        );
    }

    {
        let loader_resumed = PreRamDataLoader::open(
            &meta_path,
            vec![ShardSourceKind::LocalLine {
                path: data_path.clone(),
            }],
            cfg,
        )
        .unwrap();

        assert_eq!(loader_resumed.committed_batch_id(), 2);
        let b3_recovered = loader_resumed.next_batch().unwrap().unwrap();
        assert_eq!(b3_recovered.batch_id, 3);
        assert_eq!(b3_recovered.records, batch3_expected);
    }

    let _ = fs::remove_file(&data_path);
    let _ = fs::remove_file(&meta_path);
    let _ = fs::remove_file(format!("{}.dedup", meta_path.display()));
}

#[test]
fn test_preram_cross_shard_coalescing_and_empty_shard_transition() {
    use crate::preram::{PreRamConfig, PreRamDataLoader, ShardSourceKind};

    let s0_path = temp_path("preram_s0", "txt");
    let s1_empty_path = temp_path("preram_s1_empty", "txt");
    let s2_path = temp_path("preram_s2", "txt");
    let meta_path = temp_path("preram_coalesce", "meta");
    for p in [&s0_path, &s1_empty_path, &s2_path, &meta_path] {
        let _ = fs::remove_file(p);
    }
    let _ = fs::remove_file(format!("{}.dedup", meta_path.display()));

    {
        let mut f0 = File::create(&s0_path).unwrap();
        writeln!(f0, "s0_0").unwrap();
        writeln!(f0, "s0_1").unwrap();
        writeln!(f0, "s0_2").unwrap();

        let _f1 = File::create(&s1_empty_path).unwrap();

        let mut f2 = File::create(&s2_path).unwrap();
        writeln!(f2, "s2_0").unwrap();
        writeln!(f2, "s2_1").unwrap();
        writeln!(f2, "s2_2").unwrap();
    }

    let cfg = PreRamConfig {
        capacity_batches: 5,
        refill_trigger: 2,
        batch_size: 4,
        deduplicate: true,
        auto_refill: true,
        download_workers: 2,
        postprocess: PostProcessConfig::default(),
        min_score_filter: None,
    };

    {
        let loader = PreRamDataLoader::open(
            &meta_path,
            vec![
                ShardSourceKind::LocalLine {
                    path: s0_path.clone(),
                },
                ShardSourceKind::LocalLine {
                    path: s1_empty_path.clone(),
                },
                ShardSourceKind::LocalLine {
                    path: s2_path.clone(),
                },
            ],
            cfg,
        )
        .unwrap();

        let b1 = loader.next_batch().unwrap().unwrap();
        assert_eq!(b1.batch_id, 1);
        assert_eq!(b1.records, vec!["s0_0", "s0_1", "s0_2", "s2_0"]);
        assert_eq!(b1.next_coord.shard_id, 2);
        assert!(!b1.is_eof);
        loader.commit_gpu_batch(b1.batch_id).unwrap();

        let b2 = loader.next_batch().unwrap().unwrap();
        assert_eq!(b2.batch_id, 2);
        assert_eq!(b2.records, vec!["s2_1", "s2_2"]);
        assert!(b2.is_eof);
        loader.commit_gpu_batch(b2.batch_id).unwrap();

        assert!(loader.next_batch().unwrap().is_none());
    }

    for p in [&s0_path, &s1_empty_path, &s2_path, &meta_path] {
        let _ = fs::remove_file(p);
    }
    let _ = fs::remove_file(format!("{}.dedup", meta_path.display()));
}

#[test]
fn test_parse_shard_specs_routing() {
    use crate::preram::ShardSourceKind;
    use crate::py_bridge::parse_shard_specs;

    let specs = parse_shard_specs(
        &[
            "hf://datasets/org/repo@main/part-0.parquet".to_owned(),
            "https://example.org/stream.jsonl".to_owned(),
            "local_shard.parquet".to_owned(),
            "local_lines.txt".to_owned(),
        ],
        "content",
        Some("hf_secret"),
        65536,
    )
    .unwrap();

    assert_eq!(specs.len(), 4);
    assert!(matches!(specs[0], ShardSourceKind::RemoteParquet { .. }));
    assert!(matches!(specs[1], ShardSourceKind::RemoteLine { .. }));
    assert!(matches!(specs[2], ShardSourceKind::LocalParquet { .. }));
    assert!(matches!(specs[3], ShardSourceKind::LocalLine { .. }));
}

#[test]
fn test_multimodal_parquet_image_text_json_and_dedup() {
    use crate::multimodal::{LocalMultimodalParquetSource, MultimodalValue};
    use arrow_array::{BinaryArray, Int64Array, StructArray};

    let pq_path = temp_path("multimodal_test", "parquet");
    {
        let img_bytes_field = Arc::new(Field::new("bytes", DataType::Binary, true));
        let img_path_field = Arc::new(Field::new("path", DataType::Utf8, true));
        let img_struct_type =
            DataType::Struct(vec![img_bytes_field.clone(), img_path_field.clone()].into());

        let schema = Arc::new(Schema::new(vec![
            Field::new("text", DataType::Utf8, false),
            Field::new("image", img_struct_type, false),
            Field::new("meta", DataType::Utf8, false),
            Field::new("score", DataType::Int64, false),
        ]));

        let text_arr = Arc::new(StringArray::from(vec!["hello", "hello", "world"]));
        let raw_img: Vec<Option<&[u8]>> =
            vec![Some(b"\x89PNG_1"), Some(b"\x89PNG_1"), Some(b"\x89PNG_2")];
        let bin_arr: Arc<dyn arrow_array::Array> = Arc::new(BinaryArray::from(raw_img));
        let path_arr: Arc<dyn arrow_array::Array> = Arc::new(StringArray::from(vec![
            Some("1.png"),
            Some("1.png"),
            Some("2.png"),
        ]));

        let img_struct =
            StructArray::from(vec![(img_bytes_field, bin_arr), (img_path_field, path_arr)]);

        let meta_arr = Arc::new(StringArray::from(vec![
            "{\"lang\":\"id\"}",
            "{\"lang\":\"id\"}",
            "{\"lang\":\"en\"}",
        ]));
        let score_arr = Arc::new(Int64Array::from(vec![10, 10, 20]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![text_arr, Arc::new(img_struct), meta_arr, score_arr],
        )
        .unwrap();

        let file = File::create(&pq_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    let src = LocalMultimodalParquetSource::open(0, &pq_path, Vec::new()).unwrap();
    let mut dedup = TwoStageDeduplicator::new();

    let batch = src
        .read_multimodal_batch(1, StreamCoordinate::default(), 10, Some(&mut dedup))
        .unwrap();

    assert_eq!(batch.num_rows, 2);
    assert_eq!(
        batch.columns["text"],
        vec![
            MultimodalValue::Text("hello".to_owned()),
            MultimodalValue::Text("world".to_owned())
        ]
    );
    assert_eq!(
        batch.columns["image"],
        vec![
            MultimodalValue::Bytes(b"\x89PNG_1".to_vec()),
            MultimodalValue::Bytes(b"\x89PNG_2".to_vec())
        ]
    );
    assert_eq!(
        batch.columns["meta"],
        vec![
            MultimodalValue::Json("{\"lang\":\"id\"}".to_owned()),
            MultimodalValue::Json("{\"lang\":\"en\"}".to_owned())
        ]
    );
    assert_eq!(
        batch.columns["score"],
        vec![MultimodalValue::Int(10), MultimodalValue::Int(20)]
    );

    let _ = fs::remove_file(&pq_path);
}

#[tokio::test]
async fn test_hf_credentials_security_and_ndjson_parquet_upload() {
    use crate::hf_hub::{HfCredentials, HfHubClient};
    use crate::multimodal::MultimodalValue;

    let creds = HfCredentials::new(
        Some("hf_super_secret_token_123".to_owned()),
        Some("silvermoon".to_owned()),
        Some("luminamoon-org".to_owned()),
        Some("https://huggingface.co".to_owned()),
    );

    let dbg_str = format!("{creds:?}");
    assert!(!dbg_str.contains("hf_super_secret_token_123"));
    assert!(dbg_str.contains("[REDACTED]"));

    assert_eq!(
        creds.resolve_repo_id("my-dataset").unwrap(),
        ("luminamoon-org".to_owned(), "my-dataset".to_owned())
    );

    assert!(
        creds
            .validate_url_security("http://evil.example.com/api")
            .is_err()
    );
    assert!(
        creds
            .validate_url_security("http://127.0.0.1:8080/api")
            .is_ok()
    );

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        for _ in 0..2 {
            if let Ok((mut stream, _)) = listener.accept().await {
                let mut buf = vec![0u8; 16384];
                let _ = stream.read(&mut buf).await;
                let resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\nConnection: close\r\n\r\n{\"status\":\"ok\"}";
                let _ = stream.write_all(resp.as_bytes()).await;
            }
        }
    });

    let local_creds = HfCredentials::new(
        Some("hf_test_token".to_owned()),
        Some("silvermoon".to_owned()),
        Some("luminamoon-org".to_owned()),
        Some(format!("http://{addr}")),
    );
    let hub = HfHubClient::new(local_creds).unwrap();

    let repo_id = hub
        .create_repo("cleaned-multimodal", "dataset", true, true)
        .await
        .unwrap();
    assert_eq!(repo_id, "luminamoon-org/cleaned-multimodal");

    let cols = vec![
        (
            "text".to_owned(),
            vec![MultimodalValue::Text("sample".to_owned())],
        ),
        (
            "image".to_owned(),
            vec![MultimodalValue::Bytes(vec![1, 2, 3, 4])],
        ),
    ];
    let uploaded_uri = hub
        .upload_multimodal_parquet(
            "cleaned-multimodal",
            "train/part-000.parquet",
            &cols,
            "main",
            "Upload multimodal shard",
        )
        .await
        .unwrap();
    assert_eq!(
        uploaded_uri,
        "hf://datasets/luminamoon-org/cleaned-multimodal@main/train/part-000.parquet"
    );
}

#[test]
fn test_length_prefixed_multimodal_hash_and_hf_struct_roundtrip() {
    use crate::multimodal::{
        LocalMultimodalParquetSource, MultimodalValue, serialize_columns_to_parquet_bytes,
    };

    let mut buf1 = Vec::new();
    MultimodalValue::Text("ab".to_owned()).update_hash_payload(&mut buf1);
    MultimodalValue::Text("c".to_owned()).update_hash_payload(&mut buf1);

    let mut buf2 = Vec::new();
    MultimodalValue::Text("a".to_owned()).update_hash_payload(&mut buf2);
    MultimodalValue::Text("\x01bc".to_owned()).update_hash_payload(&mut buf2);

    assert_ne!(buf1, buf2);

    let cols = vec![
        (
            "caption".to_owned(),
            vec![
                MultimodalValue::Text("first".to_owned()),
                MultimodalValue::Text("second".to_owned()),
            ],
        ),
        (
            "image".to_owned(),
            vec![
                MultimodalValue::Bytes(vec![0x89, 0x50, 0x4E, 0x47]),
                MultimodalValue::Null,
            ],
        ),
    ];

    let pq_bytes = serialize_columns_to_parquet_bytes(&cols).unwrap();
    let pq_path = temp_path("hf_struct_roundtrip", "parquet");
    fs::write(&pq_path, &pq_bytes).unwrap();

    let src = LocalMultimodalParquetSource::open(0, &pq_path, Vec::new()).unwrap();
    let batch = src
        .read_multimodal_batch(1, StreamCoordinate::default(), 10, None)
        .unwrap();

    assert_eq!(batch.num_rows, 2);
    assert_eq!(
        batch.columns["caption"],
        vec![
            MultimodalValue::Text("first".to_owned()),
            MultimodalValue::Text("second".to_owned()),
        ]
    );
    assert_eq!(
        batch.columns["image"],
        vec![
            MultimodalValue::Bytes(vec![0x89, 0x50, 0x4E, 0x47]),
            MultimodalValue::Null,
        ]
    );

    let _ = fs::remove_file(&pq_path);
}

#[tokio::test]
async fn test_step6_dynamic_schema_formats_and_remote_multimodal_parquet() {
    use crate::hf_hub::{HfCredentials, HfHubClient};
    use crate::multimodal::{
        LocalMultimodalParquetSource, MultimodalValue, RemoteMultimodalParquetSource,
        serialize_columns_to_json_bytes, serialize_columns_to_jsonl_bytes,
        serialize_columns_to_parquet_bytes,
    };

    let cols = vec![
        (
            "custom_prompt".to_owned(),
            vec![
                MultimodalValue::Text("alpha".to_owned()),
                MultimodalValue::Text("alpha".to_owned()),
                MultimodalValue::Text("beta".to_owned()),
            ],
        ),
        (
            "camera_frame".to_owned(),
            vec![
                MultimodalValue::Bytes(vec![0x89, 0x50, 0x4E, 0x47, 0x01]),
                MultimodalValue::Bytes(vec![0x89, 0x50, 0x4E, 0x47, 0x01]),
                MultimodalValue::Bytes(vec![0x89, 0x50, 0x4E, 0x47, 0x02]),
            ],
        ),
        (
            "sensor_json".to_owned(),
            vec![
                MultimodalValue::Json("{\"fps\":60}".to_owned()),
                MultimodalValue::Json("{\"fps\":60}".to_owned()),
                MultimodalValue::Json("{\"fps\":120}".to_owned()),
            ],
        ),
        (
            "step_id".to_owned(),
            vec![
                MultimodalValue::Int(100),
                MultimodalValue::Int(100),
                MultimodalValue::Int(200),
            ],
        ),
        (
            "reward_score".to_owned(),
            vec![
                MultimodalValue::Float(0.95),
                MultimodalValue::Float(0.95),
                MultimodalValue::Float(1.5),
            ],
        ),
        (
            "is_verified".to_owned(),
            vec![
                MultimodalValue::Bool(true),
                MultimodalValue::Bool(true),
                MultimodalValue::Bool(false),
            ],
        ),
    ];

    let pq_bytes = serialize_columns_to_parquet_bytes(&cols).unwrap();
    let jsonl_bytes = serialize_columns_to_jsonl_bytes(&cols).unwrap();
    let json_bytes = serialize_columns_to_json_bytes(&cols).unwrap();

    let jsonl_str = std::str::from_utf8(&jsonl_bytes).unwrap();
    assert_eq!(jsonl_str.lines().count(), 3);
    let parsed_json: serde_json::Value = serde_json::from_slice(&json_bytes).unwrap();
    assert_eq!(parsed_json.as_array().unwrap().len(), 3);
    assert_eq!(parsed_json[0]["step_id"], 100);
    assert_eq!(parsed_json[2]["is_verified"], false);
    assert_eq!(parsed_json[0]["sensor_json"]["fps"], 60);

    let pq_path = temp_path("step6_dynamic_schema", "parquet");
    fs::write(&pq_path, &pq_bytes).unwrap();

    let local_src = LocalMultimodalParquetSource::open(0, &pq_path, Vec::new()).unwrap();
    let mut dedup_local = TwoStageDeduplicator::new();
    let local_batch = local_src
        .read_multimodal_batch(1, StreamCoordinate::default(), 10, Some(&mut dedup_local))
        .unwrap();
    assert_eq!(local_batch.num_rows, 2);
    assert_eq!(
        local_batch.columns["step_id"],
        vec![MultimodalValue::Int(100), MultimodalValue::Int(200)]
    );
    assert_eq!(
        local_batch.columns["reward_score"],
        vec![MultimodalValue::Float(0.95), MultimodalValue::Float(1.5)]
    );
    assert_eq!(
        local_batch.columns["is_verified"],
        vec![MultimodalValue::Bool(true), MultimodalValue::Bool(false)]
    );

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let pq_served = Arc::new(pq_bytes);
    let pq_clone = Arc::clone(&pq_served);

    let server_handle = tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            let data = Arc::clone(&pq_clone);
            tokio::spawn(async move {
                let mut buf = vec![0u8; 32768];
                let n = stream.read(&mut buf).await.unwrap_or(0);
                if n == 0 {
                    return;
                }
                let req_str = String::from_utf8_lossy(&buf[..n]);
                if req_str.starts_with("POST ") {
                    let resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\nConnection: close\r\n\r\n{\"status\":\"ok\"}";
                    let _ = stream.write_all(resp.as_bytes()).await;
                    return;
                }
                let total = data.len();
                let mut start = 0usize;
                let mut end = total.saturating_sub(1);
                for line in req_str.lines() {
                    if let Some(r_val) = line
                        .strip_prefix("Range: bytes=")
                        .or_else(|| line.strip_prefix("range: bytes="))
                        && let Some((s, e)) = r_val.trim().split_once('-')
                    {
                        start = s.parse::<usize>().unwrap_or(0);
                        end = e.parse::<usize>().unwrap_or(end).min(total - 1);
                    }
                }
                let slice = &data[start..=end];
                let header = format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{end}/{total}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    slice.len()
                );
                let _ = stream.write_all(header.as_bytes()).await;
                let _ = stream.write_all(slice).await;
            });
        }
    });

    let remote_uri = format!("http://{addr}/dynamic.parquet");
    let remote_src = RemoteMultimodalParquetSource::connect(
        0,
        &remote_uri,
        vec![
            "custom_prompt".to_owned(),
            "camera_frame".to_owned(),
            "step_id".to_owned(),
            "is_verified".to_owned(),
        ],
        None,
    )
    .await
    .unwrap();

    let mut dedup_remote = TwoStageDeduplicator::new();
    let remote_batch = remote_src
        .read_multimodal_batch(1, StreamCoordinate::default(), 10, Some(&mut dedup_remote))
        .await
        .unwrap();

    assert_eq!(remote_batch.num_rows, 2);
    assert_eq!(
        remote_batch.columns["custom_prompt"],
        vec![
            MultimodalValue::Text("alpha".to_owned()),
            MultimodalValue::Text("beta".to_owned()),
        ]
    );
    assert_eq!(
        remote_batch.columns["step_id"],
        vec![MultimodalValue::Int(100), MultimodalValue::Int(200)]
    );
    assert_eq!(
        remote_batch.columns["is_verified"],
        vec![MultimodalValue::Bool(true), MultimodalValue::Bool(false)]
    );

    let creds = HfCredentials::new(
        Some("hf_step6_token".to_owned()),
        Some("silvermoon".to_owned()),
        Some("luminamoon-org".to_owned()),
        Some(format!("http://{addr}")),
    );
    let hub = HfHubClient::new(creds).unwrap();
    let jsonl_uri = hub
        .upload_table_auto_format(
            "multimodal-set",
            "train/shard.jsonl",
            &cols,
            "main",
            "Upload JSONL",
            true,
        )
        .await
        .unwrap();
    assert!(jsonl_uri.ends_with("/train/shard.jsonl"));

    let json_uri = hub
        .upload_table_auto_format(
            "multimodal-set",
            "train/shard.json",
            &cols,
            "main",
            "Upload JSON",
            true,
        )
        .await
        .unwrap();
    assert!(json_uri.ends_with("/train/shard.json"));

    server_handle.abort();
    let _ = fs::remove_file(&pq_path);
}

#[test]
fn test_step7_cpu_simd_avx512_avx2_newline_and_cr_kernels() {
    use crate::simd_cpu::{
        CacheAlignedBuffer, contains_byte_fast, detect_cpu_isa, find_newline_fast,
        rfind_newline_fast,
    };

    let isa = detect_cpu_isa();
    assert!(!isa.as_str().is_empty());

    let aligned = CacheAlignedBuffer::with_capacity(256);
    assert!(aligned.data.capacity() >= 256);
    assert_eq!((&raw const aligned) as usize % 64, 0);

    let mut payload = vec![b'a'; 300];
    payload[0] = b'\n';
    payload[63] = b'\n';
    payload[64] = b'\n';
    payload[195] = b'\r';
    payload[289] = b'\n';

    assert_eq!(find_newline_fast(&payload), Some(0));
    assert_eq!(find_newline_fast(&payload[1..]), Some(62));
    assert_eq!(find_newline_fast(&payload[64..]), Some(0));
    assert_eq!(find_newline_fast(&payload[65..]), Some(289 - 65));
    assert_eq!(rfind_newline_fast(&payload), Some(289));
    assert_eq!(rfind_newline_fast(&payload[..289]), Some(64));
    assert!(contains_byte_fast(&payload, b'\r'));
    assert!(!contains_byte_fast(&payload, b'\t'));
}

#[test]
fn test_step8_concurrent_multi_range_and_multicore_preram() {
    use crate::preram::{PreRamConfig, PreRamDataLoader, ShardSourceKind};
    use crate::source_hf::HttpRangeClient;

    let mut corpus = String::new();
    for idx in 0..40 {
        corpus.push_str(&format!("{{\"text\":\"record_{idx:03}\"}}\n"));
    }
    let payload = Arc::new(corpus.into_bytes());

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();

    let (addr, server_handle) = rt.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let data_ref = Arc::clone(&payload);

        let handle = tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let data = Arc::clone(&data_ref);
                tokio::spawn(async move {
                    let mut buf = [0u8; 2048];
                    let n = stream.read(&mut buf).await.unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    let req_str = String::from_utf8_lossy(&buf[..n]);
                    let total = data.len();
                    let mut start = 0usize;
                    let mut end = total.saturating_sub(1);
                    for line in req_str.lines() {
                        if let Some(r_val) = line
                            .strip_prefix("Range: bytes=")
                            .or_else(|| line.strip_prefix("range: bytes="))
                            && let Some((s, e)) = r_val.trim().split_once('-')
                        {
                            start = s.parse::<usize>().unwrap_or(0);
                            end = e.parse::<usize>().unwrap_or(end).min(total - 1);
                        }
                    }
                    let slice = &data[start..=end];
                    let header = format!(
                        "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{end}/{total}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        slice.len()
                    );
                    let _ = stream.write_all(header.as_bytes()).await;
                    let _ = stream.write_all(slice).await;
                });
            }
        });
        (addr, handle)
    });

    let remote_uri = format!("http://{addr}/concurrent_shard.jsonl");

    rt.block_on(async {
        let client = HttpRangeClient::connect(&remote_uri, None).await.unwrap();
        let parts = client
            .fetch_ranges_concurrent(vec![(0, 64), (64, 128), (128, 192), (192, 256)], 4)
            .await
            .unwrap();
        assert_eq!(parts.len(), 4);
        let mut joined = Vec::new();
        for p in parts {
            joined.extend_from_slice(&p);
        }
        assert_eq!(&joined[..], &payload[0..256]);
    });

    let meta_path = temp_path("step8_preram_multicore", "meta");
    let _ = fs::remove_file(&meta_path);
    let _ = fs::remove_file(format!("{}.dedup", meta_path.display()));

    {
        let cfg = PreRamConfig {
            capacity_batches: 8,
            refill_trigger: 2,
            batch_size: 10,
            deduplicate: true,
            auto_refill: true,
            download_workers: 4,
            postprocess: PostProcessConfig {
                json_extract_key: Some("text".to_owned()),
                ..PostProcessConfig::default()
            },
            min_score_filter: None,
        };

        let loader = PreRamDataLoader::open(
            &meta_path,
            vec![ShardSourceKind::RemoteLine {
                uri: remote_uri,
                auth_token: None,
                chunk_bytes: 128,
            }],
            cfg,
        )
        .unwrap();

        assert_eq!(loader.download_workers(), 4);

        let mut all_records = Vec::new();
        while let Some(batch) = loader.next_batch().unwrap() {
            all_records.extend(batch.records);
            loader.commit_gpu_batch(batch.batch_id).unwrap();
        }

        assert_eq!(all_records.len(), 40);
        assert_eq!(all_records[0], "record_000");
        assert_eq!(all_records[39], "record_039");
    }

    server_handle.abort();
    let _ = fs::remove_file(&meta_path);
    let _ = fs::remove_file(format!("{}.dedup", meta_path.display()));
}

#[test]
fn test_step9_parallel_postprocess_dedup_and_multi_format_bytes() {
    use crate::multimodal::{
        MultimodalValue, serialize_columns_to_json_bytes, serialize_columns_to_jsonl_bytes,
        serialize_columns_to_parquet_bytes_with_options,
    };
    use crate::postprocess::process_batch_parallel_indexed;

    let raw_lines = vec![
        "  {\"text\": \"hello\\r\\nworld\"}  ".to_owned(),
        "{\"text\": \"hello\\nworld\"}".to_owned(),
        "{\"text\": \"\"}".to_owned(),
        "{\"text\": \"unique_line\"}".to_owned(),
    ];
    let cfg = PostProcessConfig {
        json_extract_key: Some("text".to_owned()),
        ..PostProcessConfig::default()
    };
    let indexed = process_batch_parallel_indexed(&raw_lines, &cfg);
    assert_eq!(indexed.len(), 4);
    assert_eq!(indexed[0].as_deref(), Some("hello\nworld"));
    assert_eq!(indexed[1].as_deref(), Some("hello\nworld"));
    assert!(indexed[2].is_none());
    assert_eq!(indexed[3].as_deref(), Some("unique_line"));

    let mut dedup = TwoStageDeduplicator::new();
    let mut unique = Vec::new();
    for item in indexed.into_iter().flatten() {
        if dedup.try_stage(1, item.as_bytes()).is_some() {
            unique.push(item);
        }
    }
    let hashes = dedup.commit_batch(1);
    assert_eq!(
        unique,
        vec!["hello\nworld".to_owned(), "unique_line".to_owned()]
    );
    assert_eq!(hashes.len(), 2);

    let cols = vec![
        (
            "text".to_owned(),
            vec![
                MultimodalValue::Text(unique[0].clone()),
                MultimodalValue::Text(unique[1].clone()),
            ],
        ),
        (
            "tokens_len".to_owned(),
            vec![MultimodalValue::Int(2), MultimodalValue::Int(1)],
        ),
    ];
    let pq = serialize_columns_to_parquet_bytes_with_options(&cols, true).unwrap();
    assert!(pq.starts_with(b"PAR1") && pq.ends_with(b"PAR1"));
    let jsl = serialize_columns_to_jsonl_bytes(&cols).unwrap();
    assert_eq!(jsl.iter().filter(|&&b| b == b'\n').count(), 2);
    let jsa = serialize_columns_to_json_bytes(&cols).unwrap();
    assert!(jsa.starts_with(b"[") && jsa.ends_with(b"]"));
}

#[test]
fn test_custom_parquet_columns_and_score_filter() {
    use crate::multimodal::{
        LocalMultimodalParquetSource, MultimodalValue,
        serialize_columns_to_parquet_bytes_with_options,
    };

    let cols = vec![
        (
            "no".to_owned(),
            vec![
                MultimodalValue::Int(1),
                MultimodalValue::Int(2),
                MultimodalValue::Int(3),
            ],
        ),
        (
            "en".to_owned(),
            vec![
                MultimodalValue::Text("Hello world.".to_owned()),
                MultimodalValue::Text("Good morning.".to_owned()),
                MultimodalValue::Text("Thank you.".to_owned()),
            ],
        ),
        (
            "id".to_owned(),
            vec![
                MultimodalValue::Text("Halo dunia.".to_owned()),
                MultimodalValue::Text("Selamat pagi.".to_owned()),
                MultimodalValue::Text("Terima kasih.".to_owned()),
            ],
        ),
        (
            "score".to_owned(),
            vec![
                MultimodalValue::Float(0.92),
                MultimodalValue::Float(0.75),
                MultimodalValue::Float(0.98),
            ],
        ),
    ];

    let pq_bytes = serialize_columns_to_parquet_bytes_with_options(&cols, true).unwrap();
    let pq_path = temp_path("custom_cols", "parquet");
    fs::write(&pq_path, &pq_bytes).unwrap();

    let default_src = LocalParquetSource::open(0, &pq_path, "text").unwrap();
    let batch_en = default_src
        .read_batch(
            1,
            StreamCoordinate::default(),
            10,
            &PostProcessConfig::default(),
            None,
        )
        .unwrap();
    assert!(batch_en.is_eof);
    assert_eq!(
        batch_en.records,
        vec![
            "Hello world.".to_owned(),
            "Good morning.".to_owned(),
            "Thank you.".to_owned(),
        ]
    );

    let id_src = LocalParquetSource::open(0, &pq_path, "id").unwrap();
    let batch_id = id_src
        .read_batch(
            1,
            StreamCoordinate::default(),
            10,
            &PostProcessConfig::default(),
            None,
        )
        .unwrap();
    assert!(batch_id.is_eof);
    assert_eq!(
        batch_id.records,
        vec![
            "Halo dunia.".to_owned(),
            "Selamat pagi.".to_owned(),
            "Terima kasih.".to_owned(),
        ]
    );

    assert!(LocalParquetSource::open(0, &pq_path, "missing_col").is_err());

    let mm_src = LocalMultimodalParquetSource::open(0, &pq_path, Vec::new()).unwrap();
    let mut dedup = TwoStageDeduplicator::new();
    let batch = mm_src
        .read_multimodal_batch(1, StreamCoordinate::default(), 10, Some(&mut dedup))
        .unwrap();
    let committed = dedup.commit_batch(1);
    assert_eq!(committed.len(), 3);
    assert_eq!(batch.num_rows, 3);
    assert!(batch.columns.contains_key("score"));

    let exported = dedup.export_committed_hashes();
    assert_eq!(exported.len(), 3);
    let mut dedup_shard2 = TwoStageDeduplicator::new();
    dedup_shard2.import_committed_hashes(exported);
    let batch_dup = mm_src
        .read_multimodal_batch(2, StreamCoordinate::default(), 10, Some(&mut dedup_shard2))
        .unwrap();
    assert_eq!(batch_dup.num_rows, 0);

    let filtered_src = LocalParquetSource::open(0, &pq_path, "en")
        .unwrap()
        .with_min_score_filter(Some((Some("score".to_owned()), 0.90)))
        .unwrap();
    let batch_filtered = filtered_src
        .read_batch(
            1,
            StreamCoordinate::default(),
            10,
            &PostProcessConfig::default(),
            None,
        )
        .unwrap();
    assert_eq!(
        batch_filtered.records,
        vec!["Hello world.".to_owned(), "Thank you.".to_owned()]
    );

    let mm_filtered_src =
        LocalMultimodalParquetSource::open(0, &pq_path, vec!["en".to_owned(), "id".to_owned()])
            .unwrap()
            .with_min_score_filter(Some((Some("score".to_owned()), 0.90)))
            .unwrap();
    let mm_filtered_batch = mm_filtered_src
        .read_multimodal_batch(1, StreamCoordinate::default(), 10, None)
        .unwrap();
    assert_eq!(mm_filtered_batch.num_rows, 2);
    assert!(!mm_filtered_batch.columns.contains_key("score"));

    let _ = fs::remove_file(&pq_path);
}

#[test]
fn test_universal_formats_arrow_ipc_csv_tar_and_compression() {
    use crate::hf_hub::serialize_columns_by_path_format;
    use crate::multimodal::{
        InMemoryMultimodalTable, MultimodalValue, serialize_columns_to_arrow_ipc_bytes,
        serialize_columns_to_csv_bytes,
    };

    let cols = vec![
        (
            "text".to_owned(),
            vec![
                MultimodalValue::Text("row_one".to_owned()),
                MultimodalValue::Text("row_two".to_owned()),
            ],
        ),
        (
            "score".to_owned(),
            vec![MultimodalValue::Float(0.95), MultimodalValue::Float(0.60)],
        ),
        (
            "tags".to_owned(),
            vec![
                MultimodalValue::List(vec![
                    MultimodalValue::Text("math".to_owned()),
                    MultimodalValue::Text("code".to_owned()),
                ]),
                MultimodalValue::List(vec![MultimodalValue::Text("chat".to_owned())]),
            ],
        ),
    ];

    let ipc_bytes = serialize_columns_to_arrow_ipc_bytes(&cols, true).unwrap();
    let ipc_table =
        InMemoryMultimodalTable::from_bytes_auto(&ipc_bytes, "test.arrow", &[], None).unwrap();
    let ipc_batch = ipc_table.read_batch(0, 1, StreamCoordinate::default(), 10, None);
    assert_eq!(ipc_batch.num_rows, 2);
    assert_eq!(
        ipc_batch.columns["text"][0],
        MultimodalValue::Text("row_one".to_owned())
    );
    assert_eq!(
        ipc_batch.columns["tags"][0],
        MultimodalValue::List(vec![
            MultimodalValue::Text("math".to_owned()),
            MultimodalValue::Text("code".to_owned()),
        ])
    );

    let csv_bytes = serialize_columns_to_csv_bytes(&cols, b',').unwrap();
    let csv_table =
        InMemoryMultimodalTable::from_bytes_auto(&csv_bytes, "test.csv", &[], None).unwrap();
    let csv_batch = csv_table.read_batch(0, 1, StreamCoordinate::default(), 10, None);
    assert_eq!(csv_batch.num_rows, 2);
    assert_eq!(
        csv_batch.columns["text"][1],
        MultimodalValue::Text("row_two".to_owned())
    );

    let gz_bytes = serialize_columns_by_path_format(&cols, "train/data.jsonl.gz", true).unwrap();
    let gz_table =
        InMemoryMultimodalTable::from_bytes_auto(&gz_bytes, "train/data.jsonl.gz", &[], None)
            .unwrap();
    let gz_batch = gz_table.read_batch(0, 1, StreamCoordinate::default(), 10, None);
    assert_eq!(gz_batch.num_rows, 2);

    let zst_bytes =
        serialize_columns_by_path_format(&cols, "train/data.jsonl.zst", true).unwrap();
    let zst_table =
        InMemoryMultimodalTable::from_bytes_auto(&zst_bytes, "train/data.jsonl.zst", &[], None)
            .unwrap();
    let zst_batch = zst_table.read_batch(0, 1, StreamCoordinate::default(), 10, None);
    assert_eq!(zst_batch.num_rows, 2);

    let mut tar_buf = Vec::new();
    let mut write_tar_entry = |name: &str, data: &[u8]| {
        let mut header = [0u8; 512];
        header[..name.len()].copy_from_slice(name.as_bytes());
        let size_octal = format!("{:011o}\0", data.len());
        header[124..136].copy_from_slice(size_octal.as_bytes());
        header[156] = b'0';
        header[148..156].fill(b' ');
        let checksum: u32 = header.iter().map(|&b| u32::from(b)).sum();
        let chk_str = format!("{checksum:06o}\0 ");
        header[148..156].copy_from_slice(chk_str.as_bytes());
        tar_buf.extend_from_slice(&header);
        tar_buf.extend_from_slice(data);
        let rem = data.len() % 512;
        if rem != 0 {
            tar_buf.extend(std::iter::repeat_n(0u8, 512 - rem));
        }
    };
    write_tar_entry("sample_001.txt", b"caption one");
    write_tar_entry("sample_001.png", b"\x89PNG\r\n\x1a\n");
    write_tar_entry("sample_001.json", b"{\"quality\": 9}");
    write_tar_entry("sample_002.txt", b"caption two");
    tar_buf.extend_from_slice(&[0u8; 1024]);

    let tar_table =
        InMemoryMultimodalTable::from_bytes_auto(&tar_buf, "shard_00.tar", &[], None).unwrap();
    let tar_batch = tar_table.read_batch(0, 1, StreamCoordinate::default(), 10, None);
    assert_eq!(tar_batch.num_rows, 2);
    assert_eq!(
        tar_batch.columns["txt"][0],
        MultimodalValue::Text("caption one".to_owned())
    );
    assert_eq!(
        tar_batch.columns["png"][0],
        MultimodalValue::Bytes(b"\x89PNG\r\n\x1a\n".to_vec())
    );
}

#[test]
fn test_typo_diagnostics_and_developer_friendly_errors() {
    use crate::error::{
        damerau_levenshtein, diagnose_uri_typo, find_closest_matches,
        open_local_file_with_diagnostics,
    };
    use crate::hf_hub::serialize_columns_by_path_format;
    use crate::multimodal::{LocalMultimodalParquetSource, MultimodalValue};
    use crate::postprocess::process_single_record;

    assert_eq!(damerau_levenshtein("text", "txet"), 1);
    let candidates = vec!["prompt", "completion", "score", "messages"];
    assert_eq!(
        find_closest_matches("prmopt", &candidates, 3),
        vec!["prompt".to_owned()]
    );

    let err_hf = diagnose_uri_typo("hf://dataset/org/repo/data.parquet").unwrap();
    assert!(err_hf.contains("hf://datasets/"));
    let err_http = diagnose_uri_typo("httpss://huggingface.co/data.parquet").unwrap();
    assert!(err_http.contains("https://"));

    let cfg = PostProcessConfig {
        json_extract_key: Some("messages.0.content".to_owned()),
        ..PostProcessConfig::default()
    };
    let line = r#"{"messages": [{"role": "user", "content": "Hello nested!"}]}"#;
    assert_eq!(
        process_single_record(line, &cfg),
        Some("Hello nested!".to_owned())
    );

    let cols = vec![(
        "text".to_owned(),
        vec![MultimodalValue::Text("abc".to_owned())],
    )];
    let ext_err = serialize_columns_by_path_format(&cols, "train/part.parqet", true)
        .unwrap_err()
        .to_string();
    assert!(ext_err.contains("Did you mean"));

    let lfs_path = temp_path("fake_lfs_shard", "parquet");
    fs::write(
        &lfs_path,
        b"version https://git-lfs.github.com/spec/v1\noid sha256:123456\nsize 999\n",
    )
    .unwrap();
    let lfs_err = LocalMultimodalParquetSource::open(0, &lfs_path, Vec::new())
        .err()
        .unwrap()
        .to_string();
    assert!(lfs_err.contains("Git LFS"));

    let typo_dir = lfs_path.parent().unwrap();
    let typo_target = typo_dir.join("fake_lfs_shrad.parquet");
    let missing_err = open_local_file_with_diagnostics(&typo_target)
        .unwrap_err()
        .to_string();
    let _ = fs::remove_file(&lfs_path);
    assert!(missing_err.contains("not found"));
}
