use std::fs::File;
use std::io::{Read, Write};

use parking_lot::Mutex;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyByteArray, PyBytes, PyDict, PyFloat, PyInt, PySequence, PyString};
use rayon::prelude::*;

use crate::coord::StreamCoordinate;
use crate::dedup::TwoStageDeduplicator;
use crate::error::{DataLoaderError, find_closest_matches};
use crate::metadata::DualMetadataManager;
use crate::postprocess::PostProcessConfig;
use crate::quality::QualityConfig;
use crate::serialize::{DataValue, classify_string_or_json, serialize_columns_by_format};

fn to_py_err(err: DataLoaderError) -> PyErr {
    match err {
        DataLoaderError::InvalidConfig(msg) | DataLoaderError::ColumnNotFound(msg) => {
            PyValueError::new_err(msg)
        }
        DataLoaderError::InvalidCoordinate => PyValueError::new_err(err.to_string()),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

#[pyclass(frozen, name = "StreamCoordinate")]
#[derive(Clone, Copy)]
pub struct PyStreamCoordinate {
    #[pyo3(get)]
    pub shard_id: u32,
    #[pyo3(get)]
    pub byte_offset: u64,
    #[pyo3(get)]
    pub row_group_idx: u32,
    #[pyo3(get)]
    pub page_idx: u32,
    #[pyo3(get)]
    pub row_offset_in_page: u32,
    #[pyo3(get)]
    pub global_row_idx: u64,
}

#[pymethods]
impl PyStreamCoordinate {
    #[new]
    #[pyo3(signature = (
        shard_id = 0,
        byte_offset = 0,
        row_group_idx = 0,
        page_idx = 0,
        row_offset_in_page = 0,
        global_row_idx = 0,
    ))]
    pub fn new(
        shard_id: u32,
        byte_offset: u64,
        row_group_idx: u32,
        page_idx: u32,
        row_offset_in_page: u32,
        global_row_idx: u64,
    ) -> Self {
        Self {
            shard_id,
            byte_offset,
            row_group_idx,
            page_idx,
            row_offset_in_page,
            global_row_idx,
        }
    }

    pub fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let coord = StreamCoordinate {
            shard_id: self.shard_id,
            byte_offset: self.byte_offset,
            row_group_idx: self.row_group_idx,
            page_idx: self.page_idx,
            row_offset_in_page: self.row_offset_in_page,
            global_row_idx: self.global_row_idx,
        };
        PyBytes::new(py, &coord.to_bytes())
    }

    #[staticmethod]
    pub fn from_bytes(raw: &[u8]) -> PyResult<Self> {
        if raw.len() != StreamCoordinate::BYTE_LEN {
            return Err(PyValueError::new_err(format!(
                "Invalid coordinate length: expected {} bytes, got {}",
                StreamCoordinate::BYTE_LEN,
                raw.len()
            )));
        }
        let arr: &[u8; StreamCoordinate::BYTE_LEN] = raw.try_into().unwrap();
        let c = StreamCoordinate::from_bytes(arr);
        Ok(Self {
            shard_id: c.shard_id,
            byte_offset: c.byte_offset,
            row_group_idx: c.row_group_idx,
            page_idx: c.page_idx,
            row_offset_in_page: c.row_offset_in_page,
            global_row_idx: c.global_row_idx,
        })
    }

    pub fn __repr__(&self) -> String {
        format!(
            "StreamCoordinate(shard_id={}, byte_offset={}, row_group={}, page={}, row_offset={}, global_row={})",
            self.shard_id,
            self.byte_offset,
            self.row_group_idx,
            self.page_idx,
            self.row_offset_in_page,
            self.global_row_idx
        )
    }
}

#[pyclass(name = "StreamDeduplicator")]
pub struct PyStreamDeduplicator {
    inner: Mutex<TwoStageDeduplicator>,
}

#[pymethods]
impl PyStreamDeduplicator {
    #[new]
    #[pyo3(signature = (initial_hashes = None))]
    pub fn new(initial_hashes: Option<Vec<u64>>) -> Self {
        let dedup = match initial_hashes {
            Some(hashes) => TwoStageDeduplicator::with_committed(hashes),
            None => TwoStageDeduplicator::new(),
        };
        Self {
            inner: Mutex::new(dedup),
        }
    }

    #[staticmethod]
    pub fn compute_hash(payload: &[u8]) -> u64 {
        TwoStageDeduplicator::compute_hash(payload)
    }

    pub fn is_seen(&self, payload: &[u8]) -> bool {
        self.inner.lock().is_seen(payload)
    }

    pub fn try_add(&self, payload: &[u8]) -> bool {
        let mut dedup = self.inner.lock();
        dedup.try_stage(1, payload).is_some()
    }

    pub fn filter_batch(&self, records: Vec<String>) -> Vec<String> {
        let mut dedup = self.inner.lock();
        let mut out = Vec::with_capacity(records.len());
        for rec in records {
            if dedup.try_stage(1, rec.as_bytes()).is_some() {
                out.push(rec);
            }
        }
        out
    }

    pub fn commit(&self) -> usize {
        let mut dedup = self.inner.lock();
        dedup.commit_batch(1).len()
    }

    pub fn rollback(&self) -> usize {
        let mut dedup = self.inner.lock();
        dedup.rollback_batch(1)
    }

    pub fn clear(&self) {
        let mut dedup = self.inner.lock();
        dedup.clear_staged();
    }

    pub fn count(&self) -> usize {
        self.inner.lock().committed_len()
    }

    pub fn save_wal(&self, path: &str) -> PyResult<usize> {
        let dedup = self.inner.lock();
        let hashes = dedup.export_committed_hashes();
        let mut buf = Vec::with_capacity(hashes.len() * 8);
        for h in &hashes {
            buf.extend_from_slice(&h.to_le_bytes());
        }
        let mut file = File::create(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        file.write_all(&buf)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        file.sync_data()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(hashes.len())
    }

    pub fn load_wal(&self, path: &str) -> PyResult<usize> {
        let mut file = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(PyRuntimeError::new_err(e.to_string())),
        };
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        if buf.len() % 8 != 0 {
            return Err(PyValueError::new_err(format!(
                "Corrupted WAL file '{}': byte length {} is not a multiple of 8",
                path,
                buf.len()
            )));
        }
        let count = buf.len() / 8;
        let mut dedup = self.inner.lock();
        let (chunks, _) = buf.as_chunks::<8>();
        for chunk in chunks {
            let h = u64::from_le_bytes(*chunk);
            dedup.import_committed_hashes(std::iter::once(h));
        }
        Ok(count)
    }

    pub fn export_hashes(&self) -> Vec<u64> {
        self.inner.lock().export_committed_hashes()
    }
}

#[pyclass(name = "StateTracker")]
pub struct PyStateTracker {
    inner: Mutex<DualMetadataManager>,
}

#[pymethods]
impl PyStateTracker {
    #[new]
    pub fn new(meta_path: &str) -> PyResult<Self> {
        let mgr = DualMetadataManager::open(meta_path).map_err(to_py_err)?;
        Ok(Self {
            inner: Mutex::new(mgr),
        })
    }

    pub fn read(&self) -> PyStreamCoordinate {
        let mgr = self.inner.lock();
        let c = mgr.committed_coordinate();
        PyStreamCoordinate {
            shard_id: c.shard_id,
            byte_offset: c.byte_offset,
            row_group_idx: c.row_group_idx,
            page_idx: c.page_idx,
            row_offset_in_page: c.row_offset_in_page,
            global_row_idx: c.global_row_idx,
        }
    }

    #[pyo3(signature = (
        shard_id = 0,
        byte_offset = 0,
        row_group_idx = 0,
        page_idx = 0,
        row_offset_in_page = 0,
        global_row_idx = 0,
        batch_id = 1
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &self,
        shard_id: u32,
        byte_offset: u64,
        row_group_idx: u32,
        page_idx: u32,
        row_offset_in_page: u32,
        global_row_idx: u64,
        batch_id: u64,
    ) {
        let coord = StreamCoordinate {
            shard_id,
            byte_offset,
            row_group_idx,
            page_idx,
            row_offset_in_page,
            global_row_idx,
        };
        let mut mgr = self.inner.lock();
        mgr.stage_batch_in_ram(batch_id, coord);
    }

    #[pyo3(signature = (batch_id = 1))]
    pub fn commit(&self, batch_id: u64) -> PyResult<PyStreamCoordinate> {
        let mut mgr = self.inner.lock();
        let c = mgr.commit_gpu_batch(batch_id).map_err(to_py_err)?;
        Ok(PyStreamCoordinate {
            shard_id: c.shard_id,
            byte_offset: c.byte_offset,
            row_group_idx: c.row_group_idx,
            page_idx: c.page_idx,
            row_offset_in_page: c.row_offset_in_page,
            global_row_idx: c.global_row_idx,
        })
    }

    pub fn committed_batch_id(&self) -> u64 {
        self.inner.lock().committed_batch_id()
    }
}

#[pyclass(name = "QualityFilter")]
pub struct PyQualityFilter {
    cfg: QualityConfig,
}

#[pymethods]
impl PyQualityFilter {
    #[new]
    #[pyo3(signature = (
        min_chars = 0,
        max_chars = 0,
        min_words = 0,
        min_entropy = 0.0,
        max_repetition_ratio = 1.0,
        max_symbol_ratio = 1.0
    ))]
    pub fn new(
        min_chars: usize,
        max_chars: usize,
        min_words: usize,
        min_entropy: f64,
        max_repetition_ratio: f64,
        max_symbol_ratio: f64,
    ) -> Self {
        Self {
            cfg: QualityConfig {
                min_chars,
                max_chars,
                min_words,
                min_entropy,
                max_repetition_ratio,
                max_symbol_ratio,
            },
        }
    }

    pub fn is_high_quality(&self, text: &str) -> bool {
        crate::quality::evaluate_quality(text, &self.cfg).is_passed
    }

    pub fn metrics(&self, text: &str) -> (usize, usize, f64, f64, f64, bool) {
        let m = crate::quality::evaluate_quality(text, &self.cfg);
        (
            m.char_count,
            m.word_count,
            m.entropy,
            m.repetition_ratio,
            m.symbol_ratio,
            m.is_passed,
        )
    }

    pub fn filter_flags(&self, py: Python<'_>, texts: Vec<String>) -> Vec<bool> {
        let cfg = self.cfg.clone();
        py.detach(|| crate::quality::filter_quality_batch_parallel(&texts, &cfg))
    }

    pub fn filter_batch(&self, py: Python<'_>, texts: Vec<String>) -> Vec<String> {
        let cfg = self.cfg.clone();
        py.detach(|| {
            texts
                .into_iter()
                .filter(|t| crate::quality::evaluate_quality(t, &cfg).is_passed)
                .collect()
        })
    }
}

#[pyclass(name = "MinHashDeduplicator")]
pub struct PyMinHashDeduplicator {
    inner: Mutex<crate::minhash::MinHashDeduplicator>,
}

#[pymethods]
impl PyMinHashDeduplicator {
    #[new]
    #[pyo3(signature = (threshold = 0.85, k_shingle = 5, num_perm = 64))]
    pub fn new(threshold: f64, k_shingle: usize, num_perm: usize) -> Self {
        Self {
            inner: Mutex::new(crate::minhash::MinHashDeduplicator::new(
                threshold, k_shingle, num_perm,
            )),
        }
    }

    pub fn is_unique(&self, text: &str) -> bool {
        self.inner.lock().is_unique(text)
    }

    pub fn similarity(&self, text_a: &str, text_b: &str) -> f64 {
        let dedup = self.inner.lock();
        let sig_a = dedup.compute_signature(text_a);
        let sig_b = dedup.compute_signature(text_b);
        sig_a.jaccard_similarity(&sig_b)
    }

    pub fn count(&self) -> usize {
        self.inner.lock().count()
    }

    pub fn clear(&self) {
        self.inner.lock().clear();
    }
}

#[pyclass(name = "StringTensor")]
#[derive(Clone)]
pub struct PyStringTensor {
    inner: crate::string_tensor::StringTensor,
}

#[pymethods]
impl PyStringTensor {
    #[staticmethod]
    pub fn from_list(strings: Vec<String>) -> Self {
        Self {
            inner: crate::string_tensor::StringTensor::from_strings(strings),
        }
    }

    #[staticmethod]
    pub fn from_gpu_buffers(offsets: Vec<u32>, payload: &[u8]) -> Self {
        Self {
            inner: crate::string_tensor::StringTensor::from_gpu_buffers(&offsets, payload),
        }
    }

    pub fn to_gpu_buffers<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>) {
        let (offsets, payload) = self.inner.to_gpu_buffers();
        let mut offset_bytes = Vec::with_capacity(offsets.len() * 4);
        for &off in &offsets {
            offset_bytes.extend_from_slice(&off.to_le_bytes());
        }
        (PyBytes::new(py, &offset_bytes), PyBytes::new(py, &payload))
    }

    pub fn to_offsets(&self) -> Vec<u32> {
        self.inner.to_gpu_buffers().0
    }

    pub fn to_payload<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let (_, payload) = self.inner.to_gpu_buffers();
        PyBytes::new(py, &payload)
    }

    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    pub fn __getitem__(&self, index: usize) -> PyResult<String> {
        self.inner
            .get(index)
            .map(|s| s.to_owned())
            .ok_or_else(|| PyValueError::new_err("StringTensor index out of range"))
    }

    pub fn get(&self, index: usize) -> Option<String> {
        self.inner.get(index).map(|s| s.to_owned())
    }

    pub fn to_list(&self) -> Vec<String> {
        self.inner.to_vec()
    }

    pub fn lengths(&self) -> Vec<u32> {
        self.inner.lengths()
    }

    pub fn is_inlined_mask(&self) -> Vec<bool> {
        self.inner.is_inlined_mask()
    }

    pub fn total_bytes(&self) -> usize {
        self.inner.views.iter().map(|v| v.len as usize).sum()
    }

    pub fn filter_by_length(&self, min_len: u32, max_len: u32) -> Self {
        Self {
            inner: self.inner.filter_by_length(min_len, max_len),
        }
    }

    pub fn filter_by_prefix(&self, prefix: &str) -> Self {
        Self {
            inner: self.inner.filter_by_prefix(prefix),
        }
    }

    pub fn hash_all(&self, py: Python<'_>) -> Vec<u64> {
        py.detach(|| self.inner.hash_all_parallel())
    }

    pub fn __repr__(&self) -> String {
        format!(
            "StringTensor(count={}, inlined={})",
            self.inner.len(),
            self.inner.views.iter().filter(|v| v.is_inlined()).count()
        )
    }
}

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

pub fn register_dataloader_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    parent_module.add_class::<PyStreamCoordinate>()?;
    parent_module.add_class::<PyStreamDeduplicator>()?;
    parent_module.add_class::<PyStateTracker>()?;
    parent_module.add_class::<PyQualityFilter>()?;
    parent_module.add_class::<PyMinHashDeduplicator>()?;
    parent_module.add_class::<PyStringTensor>()?;
    parent_module.add_function(wrap_pyfunction!(dataloader_cpu_isa, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(suggest_closest_matches, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(serialize_table_to_bytes, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(clean_single_record, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(clean_batch_records, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(
        postprocess_and_dedup_records,
        parent_module
    )?)?;
    parent_module.add_function(wrap_pyfunction!(
        fused_pipeline_process_batch,
        parent_module
    )?)?;
    Ok(())
}
