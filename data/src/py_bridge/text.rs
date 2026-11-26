use arrow_array::Array;
use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi, to_ffi};
use parking_lot::Mutex;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyCapsule, PyTuple};
use rayon::prelude::*;

use crate::minhash::MinHashSignature;
use crate::quality::QualityConfig;

use super::to_py_err;

#[pyclass(frozen, name = "QualityFilter")]
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

    pub fn is_high_quality(&self, py: Python<'_>, text: &str) -> bool {
        py.detach(|| crate::quality::evaluate_quality(text, &self.cfg).is_passed)
    }

    pub fn metrics(&self, py: Python<'_>, text: &str) -> (usize, usize, f64, f64, f64, bool) {
        let m = py.detach(|| crate::quality::compute_quality_metrics(text, &self.cfg));
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

#[pyclass(frozen, name = "MinHashDeduplicator")]
pub struct PyMinHashDeduplicator {
    k_shingle: usize,
    num_perm: usize,
    inner: Mutex<crate::minhash::MinHashDeduplicator>,
}

#[pymethods]
impl PyMinHashDeduplicator {
    #[new]
    #[pyo3(signature = (threshold = 0.85, k_shingle = 5, num_perm = 64))]
    pub fn new(threshold: f64, k_shingle: usize, num_perm: usize) -> Self {
        let k = k_shingle.max(1);
        let p = num_perm.max(8);
        Self {
            k_shingle: k,
            num_perm: p,
            inner: Mutex::new(crate::minhash::MinHashDeduplicator::new(threshold, k, p)),
        }
    }

    pub fn is_unique(&self, py: Python<'_>, text: &str) -> bool {
        py.detach(|| self.inner.lock().is_unique(text))
    }

    pub fn filter_batch(&self, py: Python<'_>, texts: Vec<String>) -> Vec<String> {
        py.detach(|| self.inner.lock().filter_batch(&texts))
    }

    pub fn similarity(&self, py: Python<'_>, text_a: &str, text_b: &str) -> f64 {
        let k = self.k_shingle;
        let p = self.num_perm;
        py.detach(|| {
            let sig_a = MinHashSignature::compute_char_shingles(text_a, k, p);
            let sig_b = MinHashSignature::compute_char_shingles(text_b, k, p);
            sig_a.jaccard_similarity(&sig_b)
        })
    }

    pub fn num_bands(&self, py: Python<'_>) -> usize {
        py.detach(|| self.inner.lock().num_bands())
    }

    pub fn rows_per_band(&self, py: Python<'_>) -> usize {
        py.detach(|| self.inner.lock().rows_per_band())
    }

    pub fn count(&self, py: Python<'_>) -> usize {
        py.detach(|| self.inner.lock().count())
    }

    pub fn clear(&self, py: Python<'_>) {
        py.detach(|| self.inner.lock().clear());
    }
}

#[pyclass(frozen, name = "StringTensor")]
#[derive(Clone)]
pub struct PyStringTensor {
    pub(super) inner: crate::string_tensor::StringTensor,
}

#[pymethods]
impl PyStringTensor {
    #[staticmethod]
    pub fn from_list(py: Python<'_>, strings: Vec<String>) -> PyResult<Self> {
        let inner = py
            .detach(|| crate::string_tensor::StringTensor::try_from_strings(strings))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn from_gpu_buffers(py: Python<'_>, offsets: Vec<u32>, payload: &[u8]) -> PyResult<Self> {
        let inner = py
            .detach(|| crate::string_tensor::StringTensor::try_from_gpu_buffers(&offsets, payload))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Zero-copy import from any Python object implementing the Arrow PyCapsule Interface (`__arrow_c_array__`).
    #[staticmethod]
    pub fn from_arrow(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if !obj.hasattr("__arrow_c_array__")? {
            return Err(PyValueError::new_err(
                "Object does not implement the Arrow PyCapsule interface (__arrow_c_array__)",
            ));
        }
        let pair = obj.call_method0("__arrow_c_array__")?;
        let tuple = pair.cast::<PyTuple>()?;
        if tuple.len() != 2 {
            return Err(PyValueError::new_err(
                "__arrow_c_array__() must return a 2-tuple (schema_capsule, array_capsule)",
            ));
        }
        let schema_cap = tuple.get_item(0)?.cast_into::<PyCapsule>()?;
        let array_cap = tuple.get_item(1)?.cast_into::<PyCapsule>()?;

        let schema_ptr = schema_cap
            .pointer_checked(Some(c"arrow_schema"))?
            .as_ptr()
            .cast::<FFI_ArrowSchema>();
        let array_ptr = array_cap
            .pointer_checked(Some(c"arrow_array"))?
            .as_ptr()
            .cast::<FFI_ArrowArray>();

        // Transfer ownership of the C ArrowArray struct per Arrow C Data Interface spec.
        let array_data = unsafe {
            if (*array_ptr).is_empty() {
                return Err(PyValueError::new_err(
                    "Arrow array capsule has already been consumed or is empty",
                ));
            }
            let owned_array = std::ptr::replace(array_ptr, FFI_ArrowArray::empty());
            let schema_ref = &*schema_ptr;
            from_ffi(owned_array, schema_ref)
                .map_err(|e| PyValueError::new_err(format!("Arrow FFI import failed: {e}")))?
        };
        let arr = arrow_array::make_array(array_data);
        let inner = py
            .detach(|| crate::string_tensor::StringTensor::from_arrow_array(arr.as_ref()))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Exports the Arrow `Utf8View` schema capsule per Arrow PyCapsule Interface.
    pub fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        let ffi_schema = FFI_ArrowSchema::try_from(&arrow_schema::DataType::Utf8View)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        PyCapsule::new_with_value(py, ffi_schema, c"arrow_schema")
    }

    /// Exports `(schema_capsule, array_capsule)` as Arrow `Utf8View` per Arrow PyCapsule Interface.
    #[pyo3(signature = (requested_schema = None))]
    pub fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
        let _ = requested_schema;
        let sv = py
            .detach(|| self.inner.to_arrow_string_view())
            .map_err(to_py_err)?;
        let (ffi_array, ffi_schema) = to_ffi(&sv.to_data())
            .map_err(|e| PyValueError::new_err(format!("Arrow FFI export failed: {e}")))?;
        let schema_cap = PyCapsule::new_with_value(py, ffi_schema, c"arrow_schema")?;
        let array_cap = PyCapsule::new_with_value(py, ffi_array, c"arrow_array")?;
        Ok((schema_cap, array_cap))
    }

    pub fn to_gpu_buffers<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let (offset_bytes, payload) = py
            .detach(|| {
                let (offsets, payload) = self.inner.try_to_gpu_buffers()?;
                let mut off_bytes = Vec::with_capacity(offsets.len() * 4);
                for &off in &offsets {
                    off_bytes.extend_from_slice(&off.to_le_bytes());
                }
                Ok::<_, crate::error::DataLoaderError>((off_bytes, payload))
            })
            .map_err(to_py_err)?;
        Ok((PyBytes::new(py, &offset_bytes), PyBytes::new(py, &payload)))
    }

    pub fn to_offsets(&self, py: Python<'_>) -> PyResult<Vec<u32>> {
        py.detach(|| self.inner.try_to_offsets()).map_err(to_py_err)
    }

    pub fn to_payload<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let payload = py.detach(|| self.inner.to_payload());
        PyBytes::new(py, &payload)
    }

    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    pub fn __getitem__(&self, index: isize) -> PyResult<String> {
        let len = self.inner.len() as isize;
        let resolved = if index < 0 { len + index } else { index };
        if resolved < 0 || resolved >= len {
            return Err(PyIndexError::new_err("StringTensor index out of range"));
        }
        self.inner
            .get(resolved as usize)
            .map(|s| s.to_owned())
            .ok_or_else(|| PyIndexError::new_err("StringTensor index out of range"))
    }

    pub fn get(&self, index: isize) -> Option<String> {
        let len = self.inner.len() as isize;
        let resolved = if index < 0 { len + index } else { index };
        if resolved < 0 || resolved >= len {
            return None;
        }
        self.inner.get(resolved as usize).map(|s| s.to_owned())
    }

    pub fn to_list(&self, py: Python<'_>) -> Vec<String> {
        py.detach(|| self.inner.to_vec())
    }

    pub fn lengths(&self, py: Python<'_>) -> Vec<u32> {
        py.detach(|| self.inner.lengths())
    }

    pub fn is_inlined_mask(&self, py: Python<'_>) -> Vec<bool> {
        py.detach(|| self.inner.is_inlined_mask())
    }

    pub fn total_bytes(&self) -> usize {
        self.inner.views.iter().map(|v| v.len() as usize).sum()
    }

    #[pyo3(signature = (min_len, max_len=None))]
    pub fn filter_by_length(&self, py: Python<'_>, min_len: u32, max_len: Option<u32>) -> Self {
        Self {
            inner: py.detach(|| self.inner.filter_by_length_opt(min_len, max_len)),
        }
    }

    pub fn filter_by_prefix(&self, py: Python<'_>, prefix: &str) -> Self {
        Self {
            inner: py.detach(|| self.inner.filter_by_prefix(prefix)),
        }
    }

    pub fn hash_all(&self, py: Python<'_>) -> Vec<u64> {
        py.detach(|| self.inner.hash_all_parallel())
    }

    pub fn __repr__(&self) -> String {
        let inlined = self.inner.views.iter().filter(|v| v.is_inlined()).count();
        format!(
            "StringTensor(count={}, inlined={}, heap={})",
            self.inner.len(),
            inlined,
            self.inner.len() - inlined
        )
    }
}
