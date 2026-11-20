use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyByteArray, PyBytes, PyDict, PyFloat, PyInt, PySequence, PyString};
use rayon::prelude::*;

use crate::dedup::TwoStageDeduplicator;
use crate::error::find_closest_matches;
use crate::postprocess::PostProcessConfig;
use crate::serialize::{DataValue, classify_string_or_json, serialize_columns_by_format};

use super::state::PyStreamDeduplicator;
use super::text::PyStringTensor;
use super::to_py_err;

pub fn extract_py_columns_table(
    py: Python<'_>,
    columns: &Bound<'_, PyAny>,
) -> PyResult<Vec<(String, Vec<DataValue>)>> {
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

fn extract_py_seq_values(_py: Python<'_>, seq_obj: &Bound<'_, PyAny>) -> PyResult<Vec<DataValue>> {
    let seq = seq_obj
        .cast::<PySequence>()
        .map_err(|_| PyValueError::new_err("column values must be a sequence (list/tuple)"))?;
    let len = seq.len()?;
    let mut out = Vec::with_capacity(len);

    for i in 0..len {
        let item = seq.get_item(i)?;
        if item.is_none() {
            out.push(DataValue::Null);
        } else if let Ok(b) = item.cast::<PyBool>() {
            out.push(DataValue::Bool(b.is_true()));
        } else if let Ok(i) = item.cast::<PyInt>() {
            out.push(DataValue::Int(i.extract::<i64>()?));
        } else if let Ok(f) = item.cast::<PyFloat>() {
            out.push(DataValue::Float(f.extract::<f64>()?));
        } else if let Ok(s) = item.cast::<PyString>() {
            let str_val = s.to_string_lossy();
            out.push(classify_string_or_json(&str_val));
        } else if let Ok(bytes) = item.cast::<PyBytes>() {
            out.push(DataValue::Bytes(bytes.as_bytes().to_vec()));
        } else if let Ok(bytes) = item.cast::<PyByteArray>() {
            out.push(DataValue::Bytes(bytes.to_vec()));
        } else {
            let py_str = item.str()?;
            let str_repr = py_str.to_string_lossy();
            out.push(classify_string_or_json(&str_repr));
        }
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
pub fn clean_single_record(
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
    crate::postprocess::process_single_record(raw, &cfg)
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

#[pyfunction]
#[pyo3(signature = (
    records,
    trim_whitespace=true,
    normalize_newlines=true,
    min_byte_len=1,
    max_byte_len=0,
    truncate_over_max=false,
    json_extract_key=None,
    seen_hashes=None,
    deduplicate=true
))]
#[allow(clippy::too_many_arguments)]
pub fn postprocess_and_dedup_records(
    py: Python<'_>,
    records: Vec<String>,
    trim_whitespace: bool,
    normalize_newlines: bool,
    min_byte_len: usize,
    max_byte_len: usize,
    truncate_over_max: bool,
    json_extract_key: Option<String>,
    seen_hashes: Option<Vec<u64>>,
    deduplicate: bool,
) -> PyResult<(Vec<String>, Vec<u64>)> {
    let cfg = PostProcessConfig {
        trim_whitespace,
        normalize_newlines,
        min_byte_len,
        max_byte_len,
        truncate_over_max,
        json_extract_key,
    };
    Ok(py.detach(|| {
        let processed = crate::postprocess::process_batch_parallel_indexed(&records, &cfg);
        if !deduplicate {
            let cleaned_all: Vec<String> = processed.into_iter().flatten().collect();
            return (cleaned_all, Vec::new());
        }
        let mut dedup = TwoStageDeduplicator::with_committed(seen_hashes.unwrap_or_default());
        let mut cleaned_unique = Vec::with_capacity(processed.len());
        for cleaned in processed.into_iter().flatten() {
            if dedup.try_stage(1, cleaned.as_bytes()).is_some() {
                cleaned_unique.push(cleaned);
            }
        }
        let newly_committed = dedup.commit_batch(1);
        (cleaned_unique, newly_committed)
    }))
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

    let result = py.detach(|| {
        let cleaned_candidates: Vec<String> = records
            .par_iter()
            .filter_map(|raw| {
                let cleaned = crate::postprocess::process_single_record(raw, &post_cfg)?;
                if filter_quality {
                    let eval = crate::quality::evaluate_quality(&cleaned, &qual_cfg);
                    if !eval.is_passed {
                        return None;
                    }
                }
                Some(cleaned)
            })
            .collect();

        let final_strings = if let Some(dedup_wrapper) = deduplicator {
            let mut dedup = dedup_wrapper.inner.lock();
            let mut unique = Vec::with_capacity(cleaned_candidates.len());
            for s in cleaned_candidates {
                if dedup.try_stage(1, s.as_bytes()).is_some() {
                    unique.push(s);
                }
            }
            if commit_dedup {
                dedup.commit_batch(1);
            }
            unique
        } else {
            cleaned_candidates
        };

        crate::string_tensor::StringTensor::from_strings(final_strings)
    });

    Ok(PyStringTensor { inner: result })
}
