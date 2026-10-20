use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, Date32Array, Date64Array,
    FixedSizeBinaryArray, FixedSizeListArray, Float16Array, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array, LargeBinaryArray, LargeListArray, LargeStringArray,
    ListArray, RecordBatch, StringArray, StringViewArray, StructArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt8Array,
    UInt16Array, UInt32Array, UInt64Array,
};
use hashbrown::HashMap;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ParquetRecordBatchReaderBuilder, RowSelection, RowSelector,
};

use crate::constants::{DEFAULT_DOWNLOAD_WORKERS, PARQUET_MAGIC_BYTES};
use crate::coord::StreamCoordinate;
use crate::dedup::TwoStageDeduplicator;
use crate::error::{
    DataLoaderError, Result, column_not_found_with_suggestions, diagnose_parquet_file_error,
    inspect_magic_bytes_hint, open_local_file_with_diagnostics,
};
pub use crate::multimodal_formats::{
    InMemoryMultimodalTable, classify_string_or_json, json_value_to_multimodal,
};
pub use crate::multimodal_serialize::{
    build_arrow_record_batch, decode_base64, encode_base64, multimodal_value_to_json,
    serialize_columns_to_arrow_ipc_bytes, serialize_columns_to_csv_bytes,
    serialize_columns_to_json_bytes, serialize_columns_to_jsonl_bytes,
    serialize_columns_to_parquet_bytes, serialize_columns_to_parquet_bytes_with_options,
};

#[derive(Debug, Clone)]
pub enum MultimodalValue {
    Null,
    Text(String),
    Bytes(Vec<u8>),
    Json(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    List(Vec<MultimodalValue>),
}

impl PartialEq for MultimodalValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Text(a), Self::Text(b)) => a == b,
            (Self::Bytes(a), Self::Bytes(b)) => a == b,
            (Self::Json(a), Self::Json(b)) => a == b,
            (Self::Int(a), Self::Int(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => a.to_bits() == b.to_bits(),
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::List(a), Self::List(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for MultimodalValue {}

impl MultimodalValue {
    #[inline]
    pub fn update_hash_payload(&self, out: &mut Vec<u8>) {
        match self {
            Self::Null => out.push(0),
            Self::Text(s) => {
                out.push(1);
                out.extend_from_slice(&(s.len() as u64).to_le_bytes());
                out.extend_from_slice(s.as_bytes());
            }
            Self::Bytes(b) => {
                out.push(2);
                out.extend_from_slice(&(b.len() as u64).to_le_bytes());
                out.extend_from_slice(b);
            }
            Self::Json(j) => {
                out.push(3);
                out.extend_from_slice(&(j.len() as u64).to_le_bytes());
                out.extend_from_slice(j.as_bytes());
            }
            Self::Int(v) => {
                out.push(4);
                out.extend_from_slice(&v.to_le_bytes());
            }
            Self::Float(v) => {
                out.push(5);
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            Self::Bool(v) => {
                out.push(6);
                out.push(u8::from(*v));
            }
            Self::List(items) => {
                out.push(7);
                out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for item in items {
                    item.update_hash_payload(out);
                }
            }
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(*f),
            Self::Int(i) => Some(*i as f64),
            Self::Text(s) => s.trim().parse::<f64>().ok(),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MultimodalBatch {
    pub batch_id: u64,
    pub column_order: Vec<String>,
    pub columns: HashMap<String, Vec<MultimodalValue>>,
    pub num_rows: usize,
    pub next_coord: StreamCoordinate,
    pub is_eof: bool,
}

impl MultimodalBatch {
    pub fn to_ordered_columns(&self) -> Vec<(String, Vec<MultimodalValue>)> {
        self.column_order
            .iter()
            .filter_map(|name| {
                self.columns
                    .get(name)
                    .map(|vals| (name.clone(), vals.clone()))
            })
            .collect()
    }
}

pub fn extract_multimodal_column(
    batch: &RecordBatch,
    col_name: &str,
) -> Result<Vec<MultimodalValue>> {
    let Some(col) = batch.column_by_name(col_name) else {
        let schema = batch.schema();
        let avail: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        return Err(column_not_found_with_suggestions(col_name, &avail));
    };
    extract_array_values(col.as_ref())
}

pub fn extract_score_mask_from_batch(
    batch: &RecordBatch,
    col_name: &str,
    min_score: f64,
) -> Result<Vec<bool>> {
    let Some(col) = batch.column_by_name(col_name) else {
        let schema = batch.schema();
        let avail: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        return Err(column_not_found_with_suggestions(col_name, &avail));
    };
    let len = col.len();
    let mut mask = Vec::with_capacity(len);

    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        for i in 0..len {
            mask.push(!arr.is_null(i) && arr.value(i) >= min_score);
        }
        return Ok(mask);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
        for i in 0..len {
            mask.push(!arr.is_null(i) && (arr.value(i) as f64) >= min_score);
        }
        return Ok(mask);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Float16Array>() {
        for i in 0..len {
            mask.push(!arr.is_null(i) && arr.value(i).to_f64() >= min_score);
        }
        return Ok(mask);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
        for i in 0..len {
            mask.push(!arr.is_null(i) && (arr.value(i) as f64) >= min_score);
        }
        return Ok(mask);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
        for i in 0..len {
            mask.push(!arr.is_null(i) && (arr.value(i) as f64) >= min_score);
        }
        return Ok(mask);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Int16Array>() {
        for i in 0..len {
            mask.push(!arr.is_null(i) && (arr.value(i) as f64) >= min_score);
        }
        return Ok(mask);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Int8Array>() {
        for i in 0..len {
            mask.push(!arr.is_null(i) && (arr.value(i) as f64) >= min_score);
        }
        return Ok(mask);
    }
    if let Some(arr) = col.as_any().downcast_ref::<UInt64Array>() {
        for i in 0..len {
            mask.push(!arr.is_null(i) && (arr.value(i) as f64) >= min_score);
        }
        return Ok(mask);
    }
    if let Some(arr) = col.as_any().downcast_ref::<UInt32Array>() {
        for i in 0..len {
            mask.push(!arr.is_null(i) && (arr.value(i) as f64) >= min_score);
        }
        return Ok(mask);
    }
    if let Some(arr) = col.as_any().downcast_ref::<UInt16Array>() {
        for i in 0..len {
            mask.push(!arr.is_null(i) && (arr.value(i) as f64) >= min_score);
        }
        return Ok(mask);
    }
    if let Some(arr) = col.as_any().downcast_ref::<UInt8Array>() {
        for i in 0..len {
            mask.push(!arr.is_null(i) && (arr.value(i) as f64) >= min_score);
        }
        return Ok(mask);
    }

    Err(DataLoaderError::InvalidConfig(format!(
        "Score filter column '{col_name}' has type {:?}, which is not a numeric column",
        col.data_type()
    )))
}

fn extract_array_values(col: &dyn Array) -> Result<Vec<MultimodalValue>> {
    let len = col.len();
    let mut out = Vec::with_capacity(len);

    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        for i in 0..len {
            if arr.is_null(i) {
                out.push(MultimodalValue::Null);
            } else {
                out.push(classify_string_or_json(arr.value(i)));
            }
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
        for i in 0..len {
            if arr.is_null(i) {
                out.push(MultimodalValue::Null);
            } else {
                out.push(classify_string_or_json(arr.value(i)));
            }
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<StringViewArray>() {
        for i in 0..len {
            if arr.is_null(i) {
                out.push(MultimodalValue::Null);
            } else {
                out.push(classify_string_or_json(arr.value(i)));
            }
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<BinaryArray>() {
        for i in 0..len {
            if arr.is_null(i) {
                out.push(MultimodalValue::Null);
            } else {
                out.push(MultimodalValue::Bytes(arr.value(i).to_vec()));
            }
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<LargeBinaryArray>() {
        for i in 0..len {
            if arr.is_null(i) {
                out.push(MultimodalValue::Null);
            } else {
                out.push(MultimodalValue::Bytes(arr.value(i).to_vec()));
            }
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<BinaryViewArray>() {
        for i in 0..len {
            if arr.is_null(i) {
                out.push(MultimodalValue::Null);
            } else {
                out.push(MultimodalValue::Bytes(arr.value(i).to_vec()));
            }
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<FixedSizeBinaryArray>() {
        for i in 0..len {
            if arr.is_null(i) {
                out.push(MultimodalValue::Null);
            } else {
                out.push(MultimodalValue::Bytes(arr.value(i).to_vec()));
            }
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<ListArray>() {
        for i in 0..len {
            if arr.is_null(i) {
                out.push(MultimodalValue::Null);
            } else {
                let slice = arr.value(i);
                let items = extract_array_values(slice.as_ref())?;
                out.push(MultimodalValue::List(items));
            }
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<LargeListArray>() {
        for i in 0..len {
            if arr.is_null(i) {
                out.push(MultimodalValue::Null);
            } else {
                let slice = arr.value(i);
                let items = extract_array_values(slice.as_ref())?;
                out.push(MultimodalValue::List(items));
            }
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<FixedSizeListArray>() {
        for i in 0..len {
            if arr.is_null(i) {
                out.push(MultimodalValue::Null);
            } else {
                let slice = arr.value(i);
                let items = extract_array_values(slice.as_ref())?;
                out.push(MultimodalValue::List(items));
            }
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<StructArray>() {
        if let Some(bytes_col) = arr.column_by_name("bytes") {
            let extracted_bytes = extract_array_values(bytes_col.as_ref())?;
            let path_vals = if let Some(p_col) = arr.column_by_name("path") {
                Some(extract_array_values(p_col.as_ref())?)
            } else {
                None
            };

            for i in 0..len {
                if arr.is_null(i) {
                    out.push(MultimodalValue::Null);
                } else if !matches!(extracted_bytes[i], MultimodalValue::Null) {
                    out.push(extracted_bytes[i].clone());
                } else if let Some(paths) = &path_vals {
                    out.push(paths[i].clone());
                } else {
                    out.push(MultimodalValue::Null);
                }
            }
            return Ok(out);
        }

        let field_names: Vec<String> = arr.fields().iter().map(|f| f.name().clone()).collect();
        let child_cols: Vec<Vec<MultimodalValue>> = arr
            .columns()
            .iter()
            .map(|c| extract_array_values(c.as_ref()))
            .collect::<Result<_>>()?;

        for i in 0..len {
            if arr.is_null(i) {
                out.push(MultimodalValue::Null);
            } else {
                let mut map = serde_json::Map::with_capacity(field_names.len());
                for (name, col_vals) in field_names.iter().zip(child_cols.iter()) {
                    map.insert(name.clone(), multimodal_value_to_json(&col_vals[i]));
                }
                out.push(MultimodalValue::Json(
                    serde_json::Value::Object(map).to_string(),
                ));
            }
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
        for i in 0..len {
            out.push(if arr.is_null(i) {
                MultimodalValue::Null
            } else {
                MultimodalValue::Int(arr.value(i))
            });
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
        for i in 0..len {
            out.push(if arr.is_null(i) {
                MultimodalValue::Null
            } else {
                MultimodalValue::Int(i64::from(arr.value(i)))
            });
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<Int16Array>() {
        for i in 0..len {
            out.push(if arr.is_null(i) {
                MultimodalValue::Null
            } else {
                MultimodalValue::Int(i64::from(arr.value(i)))
            });
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<Int8Array>() {
        for i in 0..len {
            out.push(if arr.is_null(i) {
                MultimodalValue::Null
            } else {
                MultimodalValue::Int(i64::from(arr.value(i)))
            });
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<UInt64Array>() {
        for i in 0..len {
            out.push(if arr.is_null(i) {
                MultimodalValue::Null
            } else {
                MultimodalValue::Int(arr.value(i) as i64)
            });
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<UInt32Array>() {
        for i in 0..len {
            out.push(if arr.is_null(i) {
                MultimodalValue::Null
            } else {
                MultimodalValue::Int(i64::from(arr.value(i)))
            });
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<UInt16Array>() {
        for i in 0..len {
            out.push(if arr.is_null(i) {
                MultimodalValue::Null
            } else {
                MultimodalValue::Int(i64::from(arr.value(i)))
            });
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<UInt8Array>() {
        for i in 0..len {
            out.push(if arr.is_null(i) {
                MultimodalValue::Null
            } else {
                MultimodalValue::Int(i64::from(arr.value(i)))
            });
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        for i in 0..len {
            out.push(if arr.is_null(i) {
                MultimodalValue::Null
            } else {
                MultimodalValue::Float(arr.value(i))
            });
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
        for i in 0..len {
            out.push(if arr.is_null(i) {
                MultimodalValue::Null
            } else {
                MultimodalValue::Float(f64::from(arr.value(i)))
            });
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<Float16Array>() {
        for i in 0..len {
            out.push(if arr.is_null(i) {
                MultimodalValue::Null
            } else {
                MultimodalValue::Float(arr.value(i).to_f64())
            });
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<BooleanArray>() {
        for i in 0..len {
            out.push(if arr.is_null(i) {
                MultimodalValue::Null
            } else {
                MultimodalValue::Bool(arr.value(i))
            });
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<Date32Array>() {
        for i in 0..len {
            out.push(if arr.is_null(i) {
                MultimodalValue::Null
            } else {
                MultimodalValue::Int(i64::from(arr.value(i)))
            });
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<Date64Array>() {
        for i in 0..len {
            out.push(if arr.is_null(i) {
                MultimodalValue::Null
            } else {
                MultimodalValue::Int(arr.value(i))
            });
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<TimestampSecondArray>() {
        for i in 0..len {
            out.push(if arr.is_null(i) {
                MultimodalValue::Null
            } else {
                MultimodalValue::Int(arr.value(i))
            });
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<TimestampMillisecondArray>() {
        for i in 0..len {
            out.push(if arr.is_null(i) {
                MultimodalValue::Null
            } else {
                MultimodalValue::Int(arr.value(i))
            });
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<TimestampMicrosecondArray>() {
        for i in 0..len {
            out.push(if arr.is_null(i) {
                MultimodalValue::Null
            } else {
                MultimodalValue::Int(arr.value(i))
            });
        }
        return Ok(out);
    }

    if let Some(arr) = col.as_any().downcast_ref::<TimestampNanosecondArray>() {
        for i in 0..len {
            out.push(if arr.is_null(i) {
                MultimodalValue::Null
            } else {
                MultimodalValue::Int(arr.value(i))
            });
        }
        return Ok(out);
    }

    Err(DataLoaderError::InvalidConfig(format!(
        "Unsupported Arrow column data type: {:?}",
        col.data_type()
    )))
}

enum LocalMultimodalInner {
    Parquet {
        file: File,
        arrow_metadata: ArrowReaderMetadata,
        column_names: Vec<String>,
        root_indices: Vec<usize>,
        row_group_counts: Vec<usize>,
        min_score_filter: Option<(String, f64)>,
    },
    Table {
        raw_bytes: Vec<u8>,
        path_hint: String,
        requested_cols: Vec<String>,
        table: InMemoryMultimodalTable,
    },
}

pub struct LocalMultimodalParquetSource {
    shard_id: u32,
    inner: LocalMultimodalInner,
}

impl LocalMultimodalParquetSource {
    pub fn open(shard_id: u32, path: impl AsRef<Path>, columns: Vec<String>) -> Result<Self> {
        let p = path.as_ref();
        let mut file = open_local_file_with_diagnostics(p)?;
        let path_str = p.to_string_lossy().to_string();
        let lower_ext = path_str.to_ascii_lowercase();

        let mut magic = [0u8; 8];
        let n_magic = file.read(&mut magic).unwrap_or(0);
        let is_parquet_ext = lower_ext.ends_with(".parquet") || lower_ext.ends_with(".pq");
        let is_parquet_magic = n_magic >= 4 && &magic[..4] == PARQUET_MAGIC_BYTES;

        if is_parquet_magic
            || (is_parquet_ext
                && n_magic >= 4
                && &magic[..4] != b"ARRO"
                && magic[0] != b'{'
                && magic[0] != b'[')
        {
            let arrow_metadata = ArrowReaderMetadata::load(&file, Default::default())
                .map_err(|e| diagnose_parquet_file_error(p, e))?;
            let parquet_meta = arrow_metadata.metadata();
            let schema_descr = parquet_meta.file_metadata().schema_descr();
            let root_fields = schema_descr.root_schema().get_fields();
            let avail_names: Vec<&str> = root_fields.iter().map(|f| f.name()).collect();

            let selected_cols = if columns.is_empty() {
                root_fields.iter().map(|f| f.name().to_owned()).collect()
            } else {
                columns
            };

            let mut root_indices = Vec::with_capacity(selected_cols.len());
            for col_name in &selected_cols {
                let idx = root_fields
                    .iter()
                    .position(|f| f.name() == col_name)
                    .ok_or_else(|| column_not_found_with_suggestions(col_name, &avail_names))?;
                root_indices.push(idx);
            }

            let row_group_counts = parquet_meta
                .row_groups()
                .iter()
                .map(|rg| rg.num_rows() as usize)
                .collect();

            return Ok(Self {
                shard_id,
                inner: LocalMultimodalInner::Parquet {
                    file,
                    arrow_metadata,
                    column_names: selected_cols,
                    root_indices,
                    row_group_counts,
                    min_score_filter: None,
                },
            });
        }

        if is_parquet_ext {
            let mut preview = [0u8; 128];
            let mut f2 = File::open(p)?;
            let n = f2.read(&mut preview).unwrap_or(0);
            if let Some(hint) = inspect_magic_bytes_hint(&preview[..n], "Parquet", &path_str) {
                return Err(DataLoaderError::InvalidConfig(hint));
            }
        }

        let raw_bytes = std::fs::read(p)?;
        let table =
            InMemoryMultimodalTable::from_bytes_auto(&raw_bytes, &path_str, &columns, None)?;
        Ok(Self {
            shard_id,
            inner: LocalMultimodalInner::Table {
                raw_bytes,
                path_hint: path_str,
                requested_cols: columns,
                table,
            },
        })
    }

    pub fn with_min_score_filter(mut self, filter: Option<(Option<String>, f64)>) -> Result<Self> {
        match &mut self.inner {
            LocalMultimodalInner::Parquet {
                arrow_metadata,
                root_indices,
                min_score_filter,
                ..
            } => {
                if let Some((score_col_opt, min_score)) = filter {
                    let schema_descr = arrow_metadata.metadata().file_metadata().schema_descr();
                    let (resolved_col, score_idx) =
                        crate::source_local::resolve_parquet_score_column(
                            schema_descr,
                            score_col_opt.as_deref(),
                        )?;
                    if !root_indices.contains(&score_idx) {
                        root_indices.push(score_idx);
                    }
                    *min_score_filter = Some((resolved_col, min_score));
                } else {
                    *min_score_filter = None;
                }
            }
            LocalMultimodalInner::Table {
                raw_bytes,
                path_hint,
                requested_cols,
                table,
            } => {
                *table = InMemoryMultimodalTable::from_bytes_auto(
                    raw_bytes,
                    path_hint,
                    requested_cols,
                    filter,
                )?;
            }
        }
        Ok(self)
    }

    pub fn read_multimodal_batch(
        &self,
        batch_id: u64,
        start_coord: StreamCoordinate,
        batch_size: usize,
        mut dedup: Option<&mut TwoStageDeduplicator>,
    ) -> Result<MultimodalBatch> {
        match &self.inner {
            LocalMultimodalInner::Table { table, .. } => {
                Ok(table.read_batch(self.shard_id, batch_id, start_coord, batch_size, dedup))
            }
            LocalMultimodalInner::Parquet {
                file,
                arrow_metadata,
                column_names,
                root_indices,
                row_group_counts,
                min_score_filter,
            } => {
                let mut rg_idx = start_coord.row_group_idx as usize;
                let mut row_in_rg = start_coord.row_offset_in_page as usize;
                let mut global_row = start_coord.global_row_idx;

                let mut collected: HashMap<String, Vec<MultimodalValue>> = column_names
                    .iter()
                    .map(|c| (c.clone(), Vec::with_capacity(batch_size)))
                    .collect();
                let mut collected_rows = 0usize;

                while rg_idx < row_group_counts.len() && collected_rows < batch_size {
                    let rg_total = row_group_counts[rg_idx];
                    if row_in_rg >= rg_total {
                        rg_idx += 1;
                        row_in_rg = 0;
                        continue;
                    }

                    let remaining_needed = batch_size - collected_rows;
                    let actual_take = (remaining_needed * 2)
                        .max(remaining_needed)
                        .min(rg_total - row_in_rg);

                    let mut selectors = Vec::with_capacity(2);
                    if row_in_rg > 0 {
                        selectors.push(RowSelector::skip(row_in_rg));
                    }
                    selectors.push(RowSelector::select(actual_take));

                    let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(
                        file.try_clone()?,
                        arrow_metadata.clone(),
                    );
                    let mask = ProjectionMask::roots(
                        arrow_metadata.metadata().file_metadata().schema_descr(),
                        root_indices.iter().copied(),
                    );

                    let reader = builder
                        .with_row_groups(vec![rg_idx])
                        .with_projection(mask)
                        .with_row_selection(RowSelection::from(selectors))
                        .with_batch_size(actual_take.max(1))
                        .build()?;

                    let mut consumed_in_slice = 0usize;
                    let mut hash_buf = crate::simd_cpu::CacheAlignedBuffer::with_capacity(256);

                    for batch_res in reader {
                        let batch = batch_res?;
                        let score_mask = if let Some((ref score_col, min_score)) = *min_score_filter
                        {
                            Some(extract_score_mask_from_batch(&batch, score_col, min_score)?)
                        } else {
                            None
                        };
                        let mut extracted_cols: Vec<Vec<MultimodalValue>> =
                            Vec::with_capacity(column_names.len());
                        for c_name in column_names {
                            extracted_cols.push(extract_multimodal_column(&batch, c_name)?);
                        }

                        for r_idx in 0..batch.num_rows() {
                            if collected_rows >= batch_size {
                                break;
                            }
                            consumed_in_slice += 1;
                            if let Some(ref mask) = score_mask
                                && !mask[r_idx]
                            {
                                continue;
                            }

                            let accepted = if let Some(d) = dedup.as_deref_mut() {
                                hash_buf.data.clear();
                                for col_vals in &extracted_cols {
                                    col_vals[r_idx].update_hash_payload(&mut hash_buf.data);
                                }
                                d.try_stage(batch_id, &hash_buf.data).is_some()
                            } else {
                                true
                            };

                            if accepted {
                                for (c_idx, c_name) in column_names.iter().enumerate() {
                                    if let Some(vec_target) = collected.get_mut(c_name) {
                                        vec_target.push(extracted_cols[c_idx][r_idx].clone());
                                    }
                                }
                                collected_rows += 1;
                            }
                        }

                        if collected_rows >= batch_size {
                            break;
                        }
                    }

                    row_in_rg += consumed_in_slice;
                    global_row += consumed_in_slice as u64;

                    if row_in_rg >= rg_total {
                        rg_idx += 1;
                        row_in_rg = 0;
                    }
                }

                let is_eof = rg_idx >= row_group_counts.len();
                let next_coord = StreamCoordinate {
                    shard_id: self.shard_id,
                    byte_offset: 0,
                    row_group_idx: rg_idx as u32,
                    page_idx: 0,
                    row_offset_in_page: row_in_rg as u32,
                    global_row_idx: global_row,
                };

                Ok(MultimodalBatch {
                    batch_id,
                    column_order: column_names.clone(),
                    columns: collected,
                    num_rows: collected_rows,
                    next_coord,
                    is_eof,
                })
            }
        }
    }
}

enum RemoteMultimodalInner {
    Parquet {
        range_client: crate::source_hf::HttpRangeClient,
        metadata: Arc<parquet::file::metadata::ParquetMetaData>,
        column_names: Vec<String>,
        root_indices: Vec<usize>,
        row_group_counts: Vec<usize>,
        min_score_filter: Option<(String, f64)>,
    },
    Table {
        raw_bytes: Vec<u8>,
        uri_hint: String,
        requested_cols: Vec<String>,
        table: InMemoryMultimodalTable,
    },
}

pub struct RemoteMultimodalParquetSource {
    shard_id: u32,
    download_workers: usize,
    inner: RemoteMultimodalInner,
}

impl RemoteMultimodalParquetSource {
    pub async fn connect(
        shard_id: u32,
        uri: &str,
        columns: Vec<String>,
        auth_token: Option<String>,
    ) -> Result<Self> {
        use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
        let range_client = crate::source_hf::HttpRangeClient::connect(uri, auth_token).await?;
        let clean_uri = uri.split('?').next().unwrap_or(uri).to_ascii_lowercase();
        let is_parquet_uri = clean_uri.ends_with(".parquet") || clean_uri.ends_with(".pq");

        if is_parquet_uri {
            let reader = crate::source_hf::HttpParquetReader::new(range_client.clone(), None);
            let builder = match ParquetRecordBatchStreamBuilder::new(reader).await {
                Ok(b) => b,
                Err(pq_err) => {
                    if let Ok(head) = range_client.fetch_range(0, 128).await
                        && let Some(hint) =
                            inspect_magic_bytes_hint(&head, "Parquet", range_client.url())
                    {
                        return Err(DataLoaderError::InvalidConfig(hint));
                    }
                    return Err(DataLoaderError::Parquet(pq_err));
                }
            };
            let metadata = builder.metadata().clone();
            let schema_descr = metadata.file_metadata().schema_descr();
            let root_fields = schema_descr.root_schema().get_fields();
            let avail_names: Vec<&str> = root_fields.iter().map(|f| f.name()).collect();

            let selected_cols = if columns.is_empty() {
                root_fields.iter().map(|f| f.name().to_owned()).collect()
            } else {
                columns
            };

            let mut root_indices = Vec::with_capacity(selected_cols.len());
            for col_name in &selected_cols {
                let idx = root_fields
                    .iter()
                    .position(|f| f.name() == col_name)
                    .ok_or_else(|| column_not_found_with_suggestions(col_name, &avail_names))?;
                root_indices.push(idx);
            }

            let row_group_counts = metadata
                .row_groups()
                .iter()
                .map(|rg| rg.num_rows() as usize)
                .collect();

            return Ok(Self {
                shard_id,
                download_workers: DEFAULT_DOWNLOAD_WORKERS,
                inner: RemoteMultimodalInner::Parquet {
                    range_client,
                    metadata,
                    column_names: selected_cols,
                    root_indices,
                    row_group_counts,
                    min_score_filter: None,
                },
            });
        }

        let raw_bytes = range_client.fetch_all_bytes().await?.to_vec();
        let table = InMemoryMultimodalTable::from_bytes_auto(&raw_bytes, uri, &columns, None)?;
        Ok(Self {
            shard_id,
            download_workers: DEFAULT_DOWNLOAD_WORKERS,
            inner: RemoteMultimodalInner::Table {
                raw_bytes,
                uri_hint: uri.to_owned(),
                requested_cols: columns,
                table,
            },
        })
    }

    pub fn with_download_workers(mut self, workers: usize) -> Self {
        self.download_workers = if workers == 0 {
            DEFAULT_DOWNLOAD_WORKERS
        } else {
            workers
        };
        self
    }

    pub fn with_min_score_filter(mut self, filter: Option<(Option<String>, f64)>) -> Result<Self> {
        match &mut self.inner {
            RemoteMultimodalInner::Parquet {
                metadata,
                root_indices,
                min_score_filter,
                ..
            } => {
                if let Some((score_col_opt, min_score)) = filter {
                    let schema_descr = metadata.file_metadata().schema_descr();
                    let (resolved_col, score_idx) =
                        crate::source_local::resolve_parquet_score_column(
                            schema_descr,
                            score_col_opt.as_deref(),
                        )?;
                    if !root_indices.contains(&score_idx) {
                        root_indices.push(score_idx);
                    }
                    *min_score_filter = Some((resolved_col, min_score));
                } else {
                    *min_score_filter = None;
                }
            }
            RemoteMultimodalInner::Table {
                raw_bytes,
                uri_hint,
                requested_cols,
                table,
            } => {
                *table = InMemoryMultimodalTable::from_bytes_auto(
                    raw_bytes,
                    uri_hint,
                    requested_cols,
                    filter,
                )?;
            }
        }
        Ok(self)
    }

    pub fn download_workers(&self) -> usize {
        self.download_workers
    }

    pub async fn read_multimodal_batch(
        &self,
        batch_id: u64,
        start_coord: StreamCoordinate,
        batch_size: usize,
        mut dedup: Option<&mut TwoStageDeduplicator>,
    ) -> Result<MultimodalBatch> {
        match &self.inner {
            RemoteMultimodalInner::Table { table, .. } => {
                Ok(table.read_batch(self.shard_id, batch_id, start_coord, batch_size, dedup))
            }
            RemoteMultimodalInner::Parquet {
                range_client,
                metadata,
                column_names,
                root_indices,
                row_group_counts,
                min_score_filter,
            } => {
                use futures::StreamExt;
                use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;

                let mut rg_idx = start_coord.row_group_idx as usize;
                let mut row_in_rg = start_coord.row_offset_in_page as usize;
                let mut global_row = start_coord.global_row_idx;

                let mut collected: HashMap<String, Vec<MultimodalValue>> = column_names
                    .iter()
                    .map(|c| (c.clone(), Vec::with_capacity(batch_size)))
                    .collect();
                let mut collected_rows = 0usize;

                while rg_idx < row_group_counts.len() && collected_rows < batch_size {
                    let rg_total = row_group_counts[rg_idx];
                    if row_in_rg >= rg_total {
                        rg_idx += 1;
                        row_in_rg = 0;
                        continue;
                    }

                    let remaining_needed = batch_size - collected_rows;
                    let actual_take = (remaining_needed * 2)
                        .max(remaining_needed)
                        .min(rg_total - row_in_rg);

                    let mut selectors = Vec::with_capacity(2);
                    if row_in_rg > 0 {
                        selectors.push(RowSelector::skip(row_in_rg));
                    }
                    selectors.push(RowSelector::select(actual_take));

                    let reader = crate::source_hf::HttpParquetReader::new(
                        range_client.clone(),
                        Some(metadata.clone()),
                    )
                    .with_download_workers(self.download_workers);
                    let builder = ParquetRecordBatchStreamBuilder::new(reader).await?;
                    let mask = ProjectionMask::roots(
                        metadata.file_metadata().schema_descr(),
                        root_indices.iter().copied(),
                    );

                    let mut stream = builder
                        .with_row_groups(vec![rg_idx])
                        .with_projection(mask)
                        .with_row_selection(RowSelection::from(selectors))
                        .with_batch_size(actual_take.max(1))
                        .build()?;

                    let mut consumed_in_slice = 0usize;
                    let mut hash_buf = crate::simd_cpu::CacheAlignedBuffer::with_capacity(256);

                    while let Some(batch_res) = stream.next().await {
                        let batch = batch_res?;
                        let score_mask = if let Some((ref score_col, min_score)) = *min_score_filter
                        {
                            Some(extract_score_mask_from_batch(&batch, score_col, min_score)?)
                        } else {
                            None
                        };
                        let mut extracted_cols: Vec<Vec<MultimodalValue>> =
                            Vec::with_capacity(column_names.len());
                        for c_name in column_names {
                            extracted_cols.push(extract_multimodal_column(&batch, c_name)?);
                        }

                        for r_idx in 0..batch.num_rows() {
                            if collected_rows >= batch_size {
                                break;
                            }
                            consumed_in_slice += 1;
                            if let Some(ref mask) = score_mask
                                && !mask[r_idx]
                            {
                                continue;
                            }

                            let accepted = if let Some(d) = dedup.as_deref_mut() {
                                hash_buf.data.clear();
                                for col_vals in &extracted_cols {
                                    col_vals[r_idx].update_hash_payload(&mut hash_buf.data);
                                }
                                d.try_stage(batch_id, &hash_buf.data).is_some()
                            } else {
                                true
                            };

                            if accepted {
                                for (c_idx, c_name) in column_names.iter().enumerate() {
                                    if let Some(vec_target) = collected.get_mut(c_name) {
                                        vec_target.push(extracted_cols[c_idx][r_idx].clone());
                                    }
                                }
                                collected_rows += 1;
                            }
                        }

                        if collected_rows >= batch_size {
                            break;
                        }
                    }

                    row_in_rg += consumed_in_slice;
                    global_row += consumed_in_slice as u64;

                    if row_in_rg >= rg_total {
                        rg_idx += 1;
                        row_in_rg = 0;
                    }
                }

                let is_eof = rg_idx >= row_group_counts.len();
                let next_coord = StreamCoordinate {
                    shard_id: self.shard_id,
                    byte_offset: 0,
                    row_group_idx: rg_idx as u32,
                    page_idx: 0,
                    row_offset_in_page: row_in_rg as u32,
                    global_row_idx: global_row,
                };

                Ok(MultimodalBatch {
                    batch_id,
                    column_order: column_names.clone(),
                    columns: collected,
                    num_rows: collected_rows,
                    next_coord,
                    is_eof,
                })
            }
        }
    }
}
