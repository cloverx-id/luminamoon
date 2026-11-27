use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyByteArray, PyBytes, PyDict, PyFloat, PyInt, PyList, PySequence, PyString, PyTuple,
};
use rayon::prelude::*;

use crate::dedup::TwoStageDeduplicator;
use crate::error::find_closest_matches;
use crate::postprocess::PostProcessConfig;
use crate::serialize::{DataValue, classify_string_or_json, serialize_columns_by_format};

use super::state::PyStreamDeduplicator;
use super::text::PyStringTensor;
use super::to_py_err;

const MAX_EXTRACT_DEPTH: usize = 32;

fn reject_str_or_bytes(obj: &Bound<'_, PyAny>, ctx: &str) -> PyResult<()> {
    if obj.is_instance_of::<PyString>()
        || obj.is_instance_of::<PyBytes>()
        || obj.is_instance_of::<PyByteArray>()
    {
        return Err(PyValueError::new_err(format!(
            "{ctx} cannot be a str, bytes, or bytearray"
        )));
    }
    Ok(())
}

pub fn extract_py_columns_table(
    py: Python<'_>,
    columns: &Bound<'_, PyAny>,
) -> PyResult<Vec<(String, Vec<DataValue>)>> {
    reject_str_or_bytes(columns, "columns")?;
    let mut rust_cols = Vec::new();

    if let Ok(dict) = columns.cast::<PyDict>() {
        for (k, v) in dict.iter() {
            let col_name = k.extract::<String>()?;
            let vals = extract_py_seq_values(py, &v)?;
            rust_cols.push((col_name, vals));
        }
    } else if let Ok(seq) = columns.cast::<PySequence>() {
        let len = seq.len()?;
        for i in 0..len {
            let item = seq.get_item(i)?;
            reject_str_or_bytes(&item, "column pair")?;
            if let Ok(pair) = item.cast::<PySequence>()
                && pair.len()? == 2
            {
                let col_name = pair.get_item(0)?.extract::<String>()?;
                let col_vals_seq = pair.get_item(1)?;
                let vals = extract_py_seq_values(py, &col_vals_seq)?;
                rust_cols.push((col_name, vals));
                continue;
            }
            return Err(PyValueError::new_err(
                "columns sequence must contain (name, values) pairs or be a dict",
            ));
        }
    } else {
        return Err(PyValueError::new_err(
            "columns must be a dict {name: [values]} or sequence of (name, [values]) pairs",
        ));
    }

    Ok(rust_cols)
}

fn extract_py_data_value(item: &Bound<'_, PyAny>, depth: usize) -> PyResult<DataValue> {
    if depth > MAX_EXTRACT_DEPTH {
        return Err(PyValueError::new_err(
            "Maximum container nesting depth exceeded during serialization (possible cyclic reference)",
        ));
    }
    if item.is_none() {
        Ok(DataValue::Null)
    } else if let Ok(b) = item.cast::<PyBool>() {
        Ok(DataValue::Bool(b.is_true()))
    } else if let Ok(i) = item.cast::<PyInt>() {
        if let Ok(v) = i.extract::<i64>() {
            Ok(DataValue::Int(v))
        } else {
            Ok(DataValue::Text(i.str()?.to_string_lossy().into_owned()))
        }
    } else if let Ok(f) = item.cast::<PyFloat>() {
        Ok(DataValue::Float(f.extract::<f64>()?))
    } else if let Ok(s) = item.cast::<PyString>() {
        Ok(classify_string_or_json(&s.to_string_lossy()))
    } else if let Ok(bytes) = item.cast::<PyBytes>() {
        Ok(DataValue::Bytes(bytes.as_bytes().to_vec()))
    } else if let Ok(bytes) = item.cast::<PyByteArray>() {
        Ok(DataValue::Bytes(bytes.to_vec()))
    } else if let Ok(list) = item.cast::<PyList>() {
        let mut vals = Vec::with_capacity(list.len());
        for elem in list.iter() {
            vals.push(extract_py_data_value(&elem, depth + 1)?);
        }
        Ok(DataValue::List(vals))
    } else if let Ok(tuple) = item.cast::<PyTuple>() {
        let mut vals = Vec::with_capacity(tuple.len());
        for elem in tuple.iter() {
            vals.push(extract_py_data_value(&elem, depth + 1)?);
        }
        Ok(DataValue::List(vals))
    } else if let Ok(dict) = item.cast::<PyDict>() {
        let mut map = serde_json::Map::with_capacity(dict.len());
        for (k, v) in dict.iter() {
            let key = k.str()?.to_string_lossy().into_owned();
            let dv = extract_py_data_value(&v, depth + 1)?;
            let jv = serde_json::to_value(&dv).map_err(|e| PyValueError::new_err(e.to_string()))?;
            map.insert(key, jv);
        }
        let json_str = serde_json::to_string(&serde_json::Value::Object(map))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(DataValue::Json(json_str))
    } else {
        let py_str = item.str()?;
        Ok(classify_string_or_json(&py_str.to_string_lossy()))
    }
}

fn extract_py_seq_values(_py: Python<'_>, seq_obj: &Bound<'_, PyAny>) -> PyResult<Vec<DataValue>> {
    reject_str_or_bytes(seq_obj, "column values")?;
    let seq = seq_obj
        .cast::<PySequence>()
        .map_err(|_| PyValueError::new_err("column values must be a sequence (list/tuple)"))?;
    let len = seq.len()?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push(extract_py_data_value(&seq.get_item(i)?, 0)?);
    }
    Ok(out)
}

#[pyfunction]
pub fn dataloader_cpu_isa() -> &'static str {
    crate::simd_cpu::detect_cpu_isa().as_str()
}

#[pyfunction]
#[pyo3(signature = (target, candidates, max_suggestions=3))]
pub fn suggest_closest_matches(
    target: &str,
    candidates: Vec<String>,
    max_suggestions: usize,
) -> Vec<String> {
    find_closest_matches(target, &candidates, max_suggestions)
}

#[pyfunction]
#[pyo3(signature = (columns, format="parquet", hf_image_struct=true))]
pub fn serialize_table_to_bytes(
    py: Python<'_>,
    columns: &Bound<'_, PyAny>,
    format: &str,
    hf_image_struct: bool,
) -> PyResult<Vec<u8>> {
    let rust_cols = extract_py_columns_table(py, columns)?;
    py.detach(|| serialize_columns_by_format(&rust_cols, format, hf_image_struct))
        .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (
    raw,
    trim_whitespace=true,
    normalize_newlines=true,
    min_byte_len=0,
    max_byte_len=0,
    truncate_over_max=false,
    json_extract_key=None
))]
#[allow(clippy::too_many_arguments)]
pub fn clean_single_record(
    py: Python<'_>,
    raw: &str,
    trim_whitespace: bool,
    normalize_newlines: bool,
    min_byte_len: usize,
    max_byte_len: usize,
    truncate_over_max: bool,
    json_extract_key: Option<String>,
) -> Option<String> {
    let cfg = PostProcessConfig {
        trim_whitespace,
        normalize_newlines,
        min_byte_len,
        max_byte_len,
        truncate_over_max,
        json_extract_key,
    };
    py.detach(|| crate::postprocess::process_single_record(raw, &cfg))
}

#[pyfunction]
#[pyo3(signature = (
    records,
    trim_whitespace=true,
    normalize_newlines=true,
    min_byte_len=0,
    max_byte_len=0,
    truncate_over_max=false,
    json_extract_key=None
))]
#[allow(clippy::too_many_arguments)]
pub fn clean_batch_records(
    py: Python<'_>,
    records: Vec<String>,
    trim_whitespace: bool,
    normalize_newlines: bool,
    min_byte_len: usize,
    max_byte_len: usize,
    truncate_over_max: bool,
    json_extract_key: Option<String>,
) -> Vec<String> {
    let cfg = PostProcessConfig {
        trim_whitespace,
        normalize_newlines,
        min_byte_len,
        max_byte_len,
        truncate_over_max,
        json_extract_key,
    };
    py.detach(|| crate::postprocess::process_batch_parallel(&records, &cfg))
}

fn run_fused_indexed(
    records: &[String],
    post_cfg: &PostProcessConfig,
    filter_quality: bool,
    qual_cfg: &crate::quality::QualityConfig,
    deduplicator: Option<&PyStreamDeduplicator>,
    commit_dedup: bool,
    batch_id: u64,
) -> crate::error::Result<(crate::string_tensor::StringTensor, Vec<usize>)> {
    let need_hash = deduplicator.is_some();
    // Pre-compute 128-bit dedup hashes in parallel BEFORE acquiring the mutex.
    let candidates: Vec<(usize, String, u128)> = records
        .par_iter()
        .enumerate()
        .filter_map(|(idx, raw)| {
            let cleaned = crate::postprocess::process_single_record(raw, post_cfg)?;
            if filter_quality && !crate::quality::evaluate_quality(&cleaned, qual_cfg).is_passed {
                return None;
            }
            let h = if need_hash {
                TwoStageDeduplicator::compute_hash(cleaned.as_bytes())
            } else {
                0
            };
            Some((idx, cleaned, h))
        })
        .collect();

    let (final_strings, survivor_indices) = if let Some(dedup_wrapper) = deduplicator {
        let mut state = dedup_wrapper.inner.lock();
        let mut unique = Vec::with_capacity(candidates.len());
        let mut indices = Vec::with_capacity(candidates.len());
        for (idx, s, h) in candidates {
            if state.dedup.try_stage_hash(batch_id, h).is_some() {
                unique.push(s);
                indices.push(idx);
            }
        }
        if commit_dedup {
            let newly = state.dedup.commit_batch(batch_id);
            state.pending_wal_hashes.extend(newly);
        }
        (unique, indices)
    } else {
        let mut unique = Vec::with_capacity(candidates.len());
        let mut indices = Vec::with_capacity(candidates.len());
        for (idx, s, _) in candidates {
            unique.push(s);
            indices.push(idx);
        }
        (unique, indices)
    };

    let tensor = crate::string_tensor::StringTensor::try_from_strings(final_strings)?;
    Ok((tensor, survivor_indices))
}

#[pyfunction]
#[pyo3(signature = (
    records,
    trim_whitespace = true,
    normalize_newlines = true,
    min_byte_len = 0,
    max_byte_len = 0,
    truncate_over_max = false,
    json_extract_key = None,
    filter_quality = false,
    min_chars = 0,
    max_chars = 0,
    min_words = 0,
    min_entropy = 0.0,
    max_repetition_ratio = 1.0,
    max_symbol_ratio = 1.0,
    deduplicator = None,
    commit_dedup = false,
    batch_id = 1,
))]
#[allow(clippy::too_many_arguments)]
pub fn fused_pipeline_process_batch(
    py: Python<'_>,
    records: Vec<String>,
    trim_whitespace: bool,
    normalize_newlines: bool,
    min_byte_len: usize,
    max_byte_len: usize,
    truncate_over_max: bool,
    json_extract_key: Option<String>,
    filter_quality: bool,
    min_chars: usize,
    max_chars: usize,
    min_words: usize,
    min_entropy: f64,
    max_repetition_ratio: f64,
    max_symbol_ratio: f64,
    deduplicator: Option<&PyStreamDeduplicator>,
    commit_dedup: bool,
    batch_id: u64,
) -> PyResult<PyStringTensor> {
    let post_cfg = PostProcessConfig {
        trim_whitespace,
        normalize_newlines,
        min_byte_len,
        max_byte_len,
        truncate_over_max,
        json_extract_key,
    };
    let qual_cfg = crate::quality::QualityConfig {
        min_chars,
        max_chars,
        min_words,
        min_entropy,
        max_repetition_ratio,
        max_symbol_ratio,
    };
    let (tensor, _) = py
        .detach(|| {
            run_fused_indexed(
                &records,
                &post_cfg,
                filter_quality,
                &qual_cfg,
                deduplicator,
                commit_dedup,
                batch_id,
            )
        })
        .map_err(to_py_err)?;
    Ok(PyStringTensor { inner: tensor })
}

#[pyfunction]
#[pyo3(signature = (
    records,
    trim_whitespace = true,
    normalize_newlines = true,
    min_byte_len = 0,
    max_byte_len = 0,
    truncate_over_max = false,
    json_extract_key = None,
    filter_quality = false,
    min_chars = 0,
    max_chars = 0,
    min_words = 0,
    min_entropy = 0.0,
    max_repetition_ratio = 1.0,
    max_symbol_ratio = 1.0,
    deduplicator = None,
    commit_dedup = false,
    batch_id = 1,
))]
#[allow(clippy::too_many_arguments)]
pub fn fused_pipeline_process_batch_indexed(
    py: Python<'_>,
    records: Vec<String>,
    trim_whitespace: bool,
    normalize_newlines: bool,
    min_byte_len: usize,
    max_byte_len: usize,
    truncate_over_max: bool,
    json_extract_key: Option<String>,
    filter_quality: bool,
    min_chars: usize,
    max_chars: usize,
    min_words: usize,
    min_entropy: f64,
    max_repetition_ratio: f64,
    max_symbol_ratio: f64,
    deduplicator: Option<&PyStreamDeduplicator>,
    commit_dedup: bool,
    batch_id: u64,
) -> PyResult<(PyStringTensor, Vec<usize>)> {
    let post_cfg = PostProcessConfig {
        trim_whitespace,
        normalize_newlines,
        min_byte_len,
        max_byte_len,
        truncate_over_max,
        json_extract_key,
    };
    let qual_cfg = crate::quality::QualityConfig {
        min_chars,
        max_chars,
        min_words,
        min_entropy,
        max_repetition_ratio,
        max_symbol_ratio,
    };
    let (tensor, indices) = py
        .detach(|| {
            run_fused_indexed(
                &records,
                &post_cfg,
                filter_quality,
                &qual_cfg,
                deduplicator,
                commit_dedup,
                batch_id,
            )
        })
        .map_err(to_py_err)?;
    Ok((PyStringTensor { inner: tensor }, indices))
}
