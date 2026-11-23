use std::sync::Arc;

use arrow_array::builder::{
    BooleanBuilder, Float64Builder, Int64Builder, LargeBinaryBuilder, LargeStringBuilder,
};
use arrow_array::{Array, RecordBatch, StructArray};
use arrow_buffer::NullBuffer;
use arrow_ipc::writer::FileWriter as ArrowIpcFileWriter;
use arrow_schema::{DataType, Field, Fields, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use crate::error::{DataLoaderError, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum DataValue {
    Null,
    Text(String),
    Bytes(Vec<u8>),
    Json(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    List(Vec<DataValue>),
}

pub fn classify_string_or_json(s: &str) -> DataValue {
    let trimmed = s.trim();
    if ((trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']')))
        && serde_json::from_str::<serde_json::Value>(trimmed).is_ok()
    {
        return DataValue::Json(trimmed.to_owned());
    }
    DataValue::Text(s.to_owned())
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

pub fn build_arrow_record_batch(
    columns: &[(String, Vec<DataValue>)],
    hf_image_struct: bool,
) -> Result<(Arc<Schema>, RecordBatch)> {
    if columns.is_empty() {
        return Err(DataLoaderError::InvalidConfig(
            "Cannot serialize empty column list".to_owned(),
        ));
    }

    let num_rows = columns[0].1.len();
    let mut fields = Vec::with_capacity(columns.len());
    let mut arrays: Vec<Arc<dyn Array>> = Vec::with_capacity(columns.len());

    for (name, vals) in columns {
        if vals.len() != num_rows {
            return Err(DataLoaderError::InvalidConfig(format!(
                "Column '{name}' length mismatch: expected {num_rows}, got {}",
                vals.len()
            )));
        }

        let has_non_null = vals.iter().any(|v| !matches!(v, DataValue::Null));
        let has_bytes = vals.iter().any(|v| matches!(v, DataValue::Bytes(_)));
        let pure_binary = has_bytes
            && vals
                .iter()
                .all(|v| matches!(v, DataValue::Null | DataValue::Bytes(_)));
        let hf_image_compatible = has_bytes
            && vals
                .iter()
                .all(|v| matches!(v, DataValue::Null | DataValue::Bytes(_) | DataValue::Text(_)));
        let all_int = has_non_null
            && vals
                .iter()
                .all(|v| matches!(v, DataValue::Null | DataValue::Int(_)));
        let has_float = vals.iter().any(|v| matches!(v, DataValue::Float(_)));
        let all_numeric = has_non_null
            && has_float
            && vals
                .iter()
                .all(|v| matches!(v, DataValue::Null | DataValue::Int(_) | DataValue::Float(_)));
        let all_bool = has_non_null
            && vals
                .iter()
                .all(|v| matches!(v, DataValue::Null | DataValue::Bool(_)));

        if hf_image_struct && hf_image_compatible {
            let mut bytes_builder = LargeBinaryBuilder::new();
            let mut path_builder = LargeStringBuilder::new();
            let mut validity = Vec::with_capacity(num_rows);

            for val in vals {
                match val {
                    DataValue::Bytes(b) => {
                        bytes_builder.append_value(b);
                        path_builder.append_null();
                        validity.push(true);
                    }
                    DataValue::Text(t) => {
                        bytes_builder.append_null();
                        path_builder.append_value(t);
                        validity.push(true);
                    }
                    _ => {
                        bytes_builder.append_null();
                        path_builder.append_null();
                        validity.push(false);
                    }
                }
            }

            let bytes_arr = Arc::new(bytes_builder.finish()) as Arc<dyn Array>;
            let path_arr = Arc::new(path_builder.finish()) as Arc<dyn Array>;
            let struct_fields = Fields::from(vec![
                Field::new("bytes", DataType::LargeBinary, true),
                Field::new("path", DataType::LargeUtf8, true),
            ]);
            let nulls = if validity.iter().all(|&v| v) {
                None
            } else {
                Some(NullBuffer::from(validity))
            };
            let struct_arr =
                StructArray::new(struct_fields.clone(), vec![bytes_arr, path_arr], nulls);

            fields.push(Field::new(name, DataType::Struct(struct_fields), true));
            arrays.push(Arc::new(struct_arr));
        } else if pure_binary {
            let mut b_builder = LargeBinaryBuilder::new();
            for val in vals {
                match val {
                    DataValue::Bytes(b) => b_builder.append_value(b),
                    _ => b_builder.append_null(),
                }
            }
            fields.push(Field::new(name, DataType::LargeBinary, true));
            arrays.push(Arc::new(b_builder.finish()));
        } else if all_int {
            let mut i_builder = Int64Builder::with_capacity(num_rows);
            for val in vals {
                match val {
                    DataValue::Int(i) => i_builder.append_value(*i),
                    _ => i_builder.append_null(),
                }
            }
            fields.push(Field::new(name, DataType::Int64, true));
            arrays.push(Arc::new(i_builder.finish()));
        } else if all_numeric {
            let mut f_builder = Float64Builder::with_capacity(num_rows);
            for val in vals {
                match val {
                    DataValue::Float(f) => f_builder.append_value(*f),
                    DataValue::Int(i) => f_builder.append_value(*i as f64),
                    _ => f_builder.append_null(),
                }
            }
            fields.push(Field::new(name, DataType::Float64, true));
            arrays.push(Arc::new(f_builder.finish()));
        } else if all_bool {
            let mut b_builder = BooleanBuilder::with_capacity(num_rows);
            for val in vals {
                match val {
                    DataValue::Bool(b) => b_builder.append_value(*b),
                    _ => b_builder.append_null(),
                }
            }
            fields.push(Field::new(name, DataType::Boolean, true));
            arrays.push(Arc::new(b_builder.finish()));
        } else {
            let mut s_builder = LargeStringBuilder::new();
            for val in vals {
                match val {
                    DataValue::Null => s_builder.append_null(),
                    DataValue::Text(s) | DataValue::Json(s) => s_builder.append_value(s),
                    DataValue::Int(i) => s_builder.append_value(i.to_string()),
                    DataValue::Float(f) => s_builder.append_value(f.to_string()),
                    DataValue::Bool(b) => s_builder.append_value(b.to_string()),
                    DataValue::Bytes(b) => s_builder.append_value(format!("0x{}", encode_hex(b))),
                    DataValue::List(l) => {
                        let json = serde_json::to_string(l).map_err(|e| {
                            DataLoaderError::InvalidConfig(format!("List serialization error: {e}"))
                        })?;
                        s_builder.append_value(json);
                    }
                }
            }
            fields.push(Field::new(name, DataType::LargeUtf8, true));
            arrays.push(Arc::new(s_builder.finish()));
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    Ok((schema, batch))
}

impl serde::Serialize for DataValue {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            DataValue::Null => serializer.serialize_none(),
            DataValue::Text(s) => serializer.serialize_str(s),
            DataValue::Bytes(b) => serializer.serialize_str(&format!("0x{}", encode_hex(b))),
            DataValue::Json(j) => {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(j) {
                    v.serialize(serializer)
                } else {
                    serializer.serialize_str(j)
                }
            }
            DataValue::Int(i) => serializer.serialize_i64(*i),
            DataValue::Float(f) => {
                if f.is_finite() {
                    serializer.serialize_f64(*f)
                } else {
                    serializer.serialize_none()
                }
            }
            DataValue::Bool(b) => serializer.serialize_bool(*b),
            DataValue::List(l) => l.serialize(serializer),
        }
    }
}

pub fn serialize_columns_to_parquet_bytes(
    columns: &[(String, Vec<DataValue>)],
    hf_image_struct: bool,
) -> Result<Vec<u8>> {
    let (schema, batch) = build_arrow_record_batch(columns, hf_image_struct)?;
    let mut out = Vec::new();
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let mut writer = ArrowWriter::try_new(&mut out, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(out)
}

pub fn serialize_columns_to_arrow_ipc_bytes(
    columns: &[(String, Vec<DataValue>)],
    hf_image_struct: bool,
) -> Result<Vec<u8>> {
    let (schema, batch) = build_arrow_record_batch(columns, hf_image_struct)?;
    let mut out = Vec::new();
    let mut writer = ArrowIpcFileWriter::try_new(&mut out, &schema)?;
    writer.write(&batch)?;
    writer.finish()?;
    Ok(out)
}

pub fn serialize_columns_to_jsonl_bytes(columns: &[(String, Vec<DataValue>)]) -> Result<Vec<u8>> {
    if columns.is_empty() {
        return Ok(Vec::new());
    }
    let num_rows = columns[0].1.len();
    for (name, vals) in columns {
        if vals.len() != num_rows {
            return Err(DataLoaderError::InvalidConfig(format!(
                "Column '{name}' length mismatch: expected {num_rows}, got {}",
                vals.len()
            )));
        }
    }
    let mut out = Vec::new();
    for row_idx in 0..num_rows {
        let mut map = serde_json::Map::with_capacity(columns.len());
        for (col_name, vals) in columns {
            let val = &vals[row_idx];
            let json_val = serde_json::to_value(val)
                .map_err(|e| DataLoaderError::InvalidConfig(e.to_string()))?;
            map.insert(col_name.clone(), json_val);
        }
        let line = serde_json::to_string(&map)
            .map_err(|e| DataLoaderError::InvalidConfig(e.to_string()))?;
        out.extend_from_slice(line.as_bytes());
        out.push(b'\n');
    }
    Ok(out)
}

pub fn serialize_columns_by_format(
    columns: &[(String, Vec<DataValue>)],
    format: &str,
    hf_image_struct: bool,
) -> Result<Vec<u8>> {
    let lower = format.trim().to_ascii_lowercase();
    if lower == "parquet" || lower.ends_with(".parquet") || lower.ends_with(".pq") {
        serialize_columns_to_parquet_bytes(columns, hf_image_struct)
    } else if lower == "arrow" || lower == "feather" || lower == "ipc" || lower.ends_with(".arrow")
    {
        serialize_columns_to_arrow_ipc_bytes(columns, hf_image_struct)
    } else if lower == "jsonl" || lower == "ndjson" || lower.ends_with(".jsonl") {
        serialize_columns_to_jsonl_bytes(columns)
    } else {
        Err(DataLoaderError::InvalidConfig(format!(
            "Unsupported serialization format '{format}'. Expected one of: 'parquet', 'arrow', 'feather', 'ipc', 'jsonl', 'ndjson'."
        )))
    }
}
