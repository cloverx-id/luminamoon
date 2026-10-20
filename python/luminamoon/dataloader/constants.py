from __future__ import annotations

DEFAULT_HF_ENDPOINT = "https://huggingface.co"
DEFAULT_HF_REVISION = "main"
DEFAULT_CACHE_DIR = ".luminamoon_cache"

LOOPBACK_HTTP_PREFIXES = (
    "http://127.0.0.1",
    "http://localhost",
    "http://[::1]",
)

HF_TOKEN_ENV_KEYS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
)
HF_USERNAME_ENV_KEYS = ("HF_USERNAME",)
HF_ORG_ENV_KEYS = ("HF_ORG",)
HF_ENDPOINT_ENV_KEYS = ("HF_ENDPOINT",)

UNSUPPORTED_HF_URI_SCHEMES = (
    "huggingface://",
    "hf-hub://",
    "hf_hub://",
    "hf://repos/",
)

URI_SCHEME_TYPO_MAP = (
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
    ("s3:/", "s3://"),
    ("gs:/", "gs://"),
)

KNOWN_SHARD_EXTENSIONS = (
    ".parquet",
    ".pq",
    ".arrow",
    ".feather",
    ".ipc",
    ".jsonl",
    ".ndjson",
    ".json",
    ".csv",
    ".tsv",
    ".tar",
    ".tgz",
    ".txt",
    ".gz",
    ".zst",
    ".zstd",
    ".lz4",
)

HF_DISCOVERY_EXTENSIONS = (
    ".parquet",
    ".pq",
    ".arrow",
    ".feather",
    ".ipc",
    ".jsonl",
    ".ndjson",
    ".json",
    ".csv",
    ".tsv",
    ".tar",
    ".tgz",
    ".jsonl.gz",
    ".jsonl.zst",
    ".jsonl.zstd",
    ".jsonl.lz4",
    ".json.gz",
    ".json.zst",
    ".csv.gz",
    ".csv.zst",
    ".tsv.gz",
    ".tar.gz",
    ".tar.zst",
)

WEBDATASET_TAR_EXTENSIONS = (
    ".tar",
    ".tgz",
    ".tar.gz",
    ".tar.zst",
    ".tar.lz4",
)

TABULAR_MODULE_PREFIXES = (
    "pandas",
    "polars",
    "pyarrow",
    "datasets",
)

VALID_EXPAND_HF_SOURCES_KWARGS = (
    "split",
    "subset",
    "token",
    "endpoint",
)

VALID_DATA_STREAM_LOADER_KWARGS = (
    "sources",
    "checkpoint_path",
    "batch_size",
    "preram_batches",
    "refill_trigger",
    "deduplicate",
    "auto_refill",
    "parquet_column",
    "json_extract_key",
    "trim_whitespace",
    "normalize_newlines",
    "min_byte_len",
    "max_byte_len",
    "truncate_over_max",
    "auth_token",
    "remote_chunk_bytes",
    "download_workers",
    "auto_commit",
    "tokenizer",
    "tokenizer_kwargs",
    "postprocess_fn",
    "split",
    "subset",
    "min_score",
    "score_column",
    "rank",
    "world_size",
    "shuffle",
    "seed",
    "shuffle_buffer_size",
)

VALID_MULTIMODAL_LOADER_KWARGS = (
    "path",
    "columns",
    "batch_size",
    "deduplicate",
    "shard_id",
    "auth_token",
    "download_workers",
    "tokenizer",
    "tokenizer_kwargs",
    "postprocess_fn",
    "min_score",
    "score_column",
    "shuffle",
    "seed",
)

VALID_STREAM_KWARGS = (
    "columns",
    "batch_size",
    "preram_batches",
    "refill_trigger",
    "download_workers",
    "deduplicate",
    "auto_commit",
    "checkpoint_path",
    "json_extract_key",
    "auth_token",
    "tokenizer",
    "tokenizer_kwargs",
    "postprocess_fn",
    "split",
    "subset",
    "min_score",
    "score_column",
    "endpoint",
    "multimodal",
    "coalesce_batches",
    "rank",
    "world_size",
    "shuffle",
    "seed",
    "shuffle_buffer_size",
    "auto_refill",
    "parquet_column",
    "trim_whitespace",
    "normalize_newlines",
    "min_byte_len",
    "max_byte_len",
    "truncate_over_max",
    "remote_chunk_bytes",
)

VALID_INTERLEAVE_KWARGS = (
    "probabilities",
    "seed",
    "stopping_strategy",
)

VALID_INTERLEAVE_STRATEGIES = (
    "first_exhausted",
    "all_exhausted",
)

VALID_FROM_HF_DATASET_KWARGS = (
    "column",
    "batch_size",
    "deduplicate",
    "trim_whitespace",
    "normalize_newlines",
    "min_byte_len",
    "max_byte_len",
    "truncate_over_max",
    "json_extract_key",
    "tokenizer",
    "tokenizer_kwargs",
    "postprocess_fn",
)

VALID_PUSH_TO_HUB_KWARGS = (
    "columns",
    "rename",
    "private",
    "revision",
    "commit_message",
    "hf_image_struct",
    "token",
    "username",
    "org",
    "endpoint",
)

DEFAULT_BATCH_SIZE = 64
DEFAULT_PRERAM_BATCHES = 100
DEFAULT_REFILL_TRIGGER = 10
DEFAULT_REMOTE_CHUNK_BYTES = 262_144
DEFAULT_DOWNLOAD_WORKERS = 4
DEFAULT_SHUFFLE_SEED = 42
DEFAULT_SHUFFLE_BUFFER_SIZE = 256
MAX_HF_TREE_PAGES = 50
HF_TREE_REQUEST_TIMEOUT_SECS = 30
