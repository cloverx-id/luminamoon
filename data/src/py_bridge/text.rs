use parking_lot::Mutex;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

use crate::quality::QualityConfig;

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
                .into_par_iter()
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

    pub fn is_unique(&self, py: Python<'_>, text: &str) -> bool {
        py.detach(|| self.inner.lock().is_unique(text))
    }

    pub fn similarity(&self, py: Python<'_>, text_a: &str, text_b: &str) -> f64 {
        py.detach(|| {
            let dedup = self.inner.lock();
            let sig_a = dedup.compute_signature(text_a);
            let sig_b = dedup.compute_signature(text_b);
            sig_a.jaccard_similarity(&sig_b)
        })
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
    pub(super) inner: crate::string_tensor::StringTensor,
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
        self.inner.to_offsets()
    }

    pub fn to_payload<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let payload = self.inner.to_payload();
        PyBytes::new(py, &payload)
    }

    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    pub fn __getitem__(&self, index: usize) -> PyResult<String> {
        self.inner
            .get(index)
            .map(|s| s.to_owned())
            .ok_or_else(|| PyIndexError::new_err("StringTensor index out of range"))
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
