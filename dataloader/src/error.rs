use thiserror::Error;

#[derive(Debug, Error)]
pub enum DataLoaderError {
    #[error("I/O failure: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP failure: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Parquet failure: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error("Arrow failure: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),

    #[error("Corrupted metadata checkpoint in both slots")]
    CorruptedMetadata,

    #[error("Unknown staged batch id: {0}")]
    UnknownBatchId(u64),

    #[error("Invalid coordinate buffer length")]
    InvalidCoordinate,

    #[error("Target column not found: {0}")]
    ColumnNotFound(String),

    #[error("Invalid source configuration: {0}")]
    InvalidConfig(String),
}

pub type Result<T> = std::result::Result<T, DataLoaderError>;
