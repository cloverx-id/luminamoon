pub const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";
pub const DEFAULT_HF_REVISION: &str = "main";
pub const GCS_HTTPS_BASE_URL: &str = "https://storage.googleapis.com";
pub const DEFAULT_GCS_ENDPOINT: &str = GCS_HTTPS_BASE_URL;
pub const S3_HTTPS_DOMAIN_SUFFIX: &str = "s3.amazonaws.com";

pub const URI_HF_DATASETS_PREFIX: &str = "hf://datasets/";
pub const URI_HTTP_PREFIX: &str = "http://";
pub const URI_HTTPS_PREFIX: &str = "https://";
pub const URI_GS_PREFIX: &str = "gs://";
pub const URI_S3_PREFIX: &str = "s3://";

pub const LOOPBACK_HTTP_PREFIXES: &[&str] =
    &["http://127.0.0.1", "http://localhost", "http://[::1]"];

pub const ENV_HF_TOKEN: &str = "HF_TOKEN";
pub const ENV_HUGGING_FACE_HUB_TOKEN: &str = "HUGGING_FACE_HUB_TOKEN";
pub const ENV_HF_HUB_TOKEN_LEGACY: &str = ENV_HUGGING_FACE_HUB_TOKEN;
pub const ENV_HF_USERNAME: &str = "HF_USERNAME";
pub const ENV_HF_ORG: &str = "HF_ORG";
pub const ENV_HF_ENDPOINT: &str = "HF_ENDPOINT";
pub const ENV_HF_HOME: &str = "HF_HOME";
pub const ENV_HOME: &str = "HOME";
pub const ENV_USERPROFILE: &str = "USERPROFILE";
pub const ENV_AWS_ENDPOINT_URL: &str = "AWS_ENDPOINT_URL";
pub const ENV_S3_ENDPOINT_URL: &str = "S3_ENDPOINT_URL";

pub const STREAM_COORD_BYTE_LEN: usize = 32;
pub const STREAM_COORDINATE_BYTE_LEN: usize = STREAM_COORD_BYTE_LEN;

pub const CHECKPOINT_SLOT_MAGIC: [u8; 4] = *b"LMDL";
pub const META_SLOT_MAGIC: [u8; 4] = CHECKPOINT_SLOT_MAGIC;
pub const CHECKPOINT_SLOT_VERSION: u32 = 1;
pub const META_SLOT_VERSION: u32 = CHECKPOINT_SLOT_VERSION;
pub const CHECKPOINT_SLOT_SIZE: usize = 64;
pub const META_SLOT_SIZE: usize = CHECKPOINT_SLOT_SIZE;
pub const CHECKPOINT_TOTAL_META_SIZE: u64 = (CHECKPOINT_SLOT_SIZE * 2) as u64;
pub const TOTAL_META_SIZE: u64 = CHECKPOINT_TOTAL_META_SIZE;
pub const DEDUP_FILE_SUFFIX: &str = ".dedup";

pub const PARQUET_MAGIC_BYTES: &[u8; 4] = b"PAR1";
pub const MAGIC_PARQUET: &[u8; 4] = PARQUET_MAGIC_BYTES;
pub const ARROW_IPC_MAGIC_BYTES: &[u8; 6] = b"ARROW1";
pub const MAGIC_ARROW_IPC: &[u8; 6] = ARROW_IPC_MAGIC_BYTES;
pub const GZIP_MAGIC_BYTES: [u8; 2] = [0x1F, 0x8B];
pub const MAGIC_GZIP: &[u8; 2] = &[0x1F, 0x8B];
pub const ZSTD_MAGIC_BYTES: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];
pub const MAGIC_ZSTD: &[u8; 4] = &[0x28, 0xB5, 0x2F, 0xFD];
pub const LZ4_FRAME_MAGIC_BYTES: [u8; 4] = [0x04, 0x22, 0x4D, 0x18];
pub const MAGIC_LZ4_FRAME: &[u8; 4] = &[0x04, 0x22, 0x4D, 0x18];
pub const GIT_LFS_POINTER_PREFIX: &[u8] = b"version https://git-lfs.github.com/spec/v1";
pub const MAGIC_GIT_LFS_V1: &[u8] = GIT_LFS_POINTER_PREFIX;

pub const SUPPORTED_UPLOAD_EXTENSIONS: &[&str] = &[
    ".parquet", ".pq", ".jsonl", ".ndjson", ".json", ".arrow", ".feather", ".ipc", ".csv", ".tsv",
];

pub const COMPRESSION_EXTENSIONS: &[&str] = &[".gz", ".zst", ".zstd", ".lz4"];

pub const UNIVERSAL_TABLE_EXTENSIONS: &[&str] = &[
    ".arrow", ".feather", ".ipc", ".csv", ".tsv", ".json", ".tar", ".tgz",
];

pub const TEXT_COLUMN_FALLBACKS: &[&str] = &[
    "text",
    "content",
    "prompt",
    "instruction",
    "input",
    "en",
    "id",
    "sentence",
    "document",
];

pub const SCORE_COLUMN_FALLBACKS: &[&str] = &[
    "score",
    "LMSE 1.0 Q-SCORE",
    "quality_score",
    "q_score",
    "reward",
    "rating",
];

pub const JSON_ARRAY_WRAPPER_KEYS: &[&str] = &["data", "rows", "train", "items"];

pub const WEBDATASET_TEXT_EXTENSIONS: &[&str] = &["txt", "text", "cls", "caption", "prompt"];

pub const UNSUPPORTED_HF_URI_SCHEMES: &[&str] =
    &["huggingface://", "hf-hub://", "hf_hub://", "hf://repos/"];
pub const HF_SCHEME_TYPOS: &[&str] = UNSUPPORTED_HF_URI_SCHEMES;

pub const URI_SCHEME_TYPO_MAP: &[(&str, &str)] = &[
    ("httpss://", "https://"),
    ("htpps://", "https://"),
    ("htpp://", "http://"),
    ("https//", "https://"),
    ("http//", "http://"),
    ("https:/", "https://"),
    ("http:/", "http://"),
    ("htps://", "https://"),
    ("htp://", "http://"),
    ("ttps://", "https://"),
];
pub const HTTP_SCHEME_TYPOS: &[(&str, &str)] = URI_SCHEME_TYPO_MAP;

pub const MAX_TYPO_SUGGESTIONS: usize = 3;
pub const MAX_COLUMN_PREVIEW_COUNT: usize = 16;
pub const MAX_SIBLING_SCAN_ENTRIES: usize = 2000;

pub const DEFAULT_BATCH_SIZE: usize = 64;
pub const DEFAULT_PRERAM_BATCHES: usize = 100;
pub const DEFAULT_REFILL_TRIGGER: usize = 10;
pub const DEFAULT_REMOTE_CHUNK_BYTES: usize = 262_144;
pub const MIN_REMOTE_CHUNK_BYTES: u64 = 64;
pub const DEFAULT_DOWNLOAD_WORKERS: usize = 4;
pub const DEFAULT_ZSTD_COMPRESSION_LEVEL: i32 = 3;

pub const HTTP_MAX_CACHED_RANGES: usize = 32;
pub const MAX_CACHED_RANGES: usize = HTTP_MAX_CACHED_RANGES;
pub const HTTP_MAX_CACHED_RANGE_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_CACHED_RANGE_BYTES: usize = HTTP_MAX_CACHED_RANGE_BYTES;
pub const HTTP_CONNECT_TIMEOUT_SECS: u64 = 20;
pub const HTTP_REQUEST_TIMEOUT_SECS: u64 = 120;
pub const HTTP_MAX_RETRIES: u32 = 4;
pub const HF_DIRECT_COMMIT_MAX_BYTES: usize = 25 * 1024 * 1024;
pub const TAR_USTAR_BLOCK_SIZE: usize = 512;
