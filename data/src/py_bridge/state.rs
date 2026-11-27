use std::path::{Path, PathBuf};

use parking_lot::Mutex;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::coord::StreamCoordinate;
use crate::dedup::TwoStageDeduplicator;
use crate::metadata::DualMetadataManager;

use super::to_py_err;
use super::wal::{WalRead, append_wal_hashes, read_wal_file, truncate_wal_file, write_full_wal};

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

impl From<StreamCoordinate> for PyStreamCoordinate {
    fn from(c: StreamCoordinate) -> Self {
        Self {
            shard_id: c.shard_id,
            byte_offset: c.byte_offset,
            row_group_idx: c.row_group_idx,
            page_idx: c.page_idx,
            row_offset_in_page: c.row_offset_in_page,
            global_row_idx: c.global_row_idx,
        }
    }
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
        Ok(StreamCoordinate::from_bytes(arr).into())
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

pub(super) struct DeduplicatorState {
    pub dedup: TwoStageDeduplicator,
    pub wal_target: Option<PathBuf>,
    pub pending_wal_hashes: Vec<u128>,
}

#[pyclass(frozen, name = "StreamDeduplicator")]
pub struct PyStreamDeduplicator {
    pub(super) inner: Mutex<DeduplicatorState>,
}

#[pymethods]
impl PyStreamDeduplicator {
    #[new]
    #[pyo3(signature = (initial_hashes = None, max_capacity = None))]
    pub fn new(initial_hashes: Option<Vec<u128>>, max_capacity: Option<usize>) -> Self {
        let mut dedup = TwoStageDeduplicator::with_capacity(max_capacity);
        if let Some(hashes) = initial_hashes {
            dedup.import_committed_hashes(hashes);
        }
        Self {
            inner: Mutex::new(DeduplicatorState {
                dedup,
                wal_target: None,
                pending_wal_hashes: Vec::new(),
            }),
        }
    }

    #[staticmethod]
    pub fn compute_hash(payload: &[u8]) -> u64 {
        TwoStageDeduplicator::compute_hash_64(payload)
    }

    #[staticmethod]
    pub fn compute_hash_128(payload: &[u8]) -> u128 {
        TwoStageDeduplicator::compute_hash(payload)
    }

    pub fn is_seen(&self, py: Python<'_>, payload: &[u8]) -> bool {
        let hash = TwoStageDeduplicator::compute_hash(payload);
        py.detach(|| self.inner.lock().dedup.is_hash_seen(hash))
    }

    #[pyo3(signature = (payload, batch_id = 1))]
    pub fn try_add(&self, py: Python<'_>, payload: &[u8], batch_id: u64) -> bool {
        let hash = TwoStageDeduplicator::compute_hash(payload);
        py.detach(|| {
            self.inner
                .lock()
                .dedup
                .try_stage_hash(batch_id, hash)
                .is_some()
        })
    }

    #[pyo3(signature = (records, batch_id = 1))]
    pub fn filter_batch(&self, py: Python<'_>, records: Vec<String>, batch_id: u64) -> Vec<String> {
        py.detach(|| {
            let mut state = self.inner.lock();
            let mut out = Vec::with_capacity(records.len());
            for rec in records {
                if state.dedup.try_stage(batch_id, rec.as_bytes()).is_some() {
                    out.push(rec);
                }
            }
            out
        })
    }

    #[pyo3(signature = (batch_id = 1))]
    pub fn commit(&self, py: Python<'_>, batch_id: u64) -> usize {
        py.detach(|| {
            let mut state = self.inner.lock();
            let newly = state.dedup.commit_batch(batch_id);
            let n = newly.len();
            state.pending_wal_hashes.extend(newly);
            n
        })
    }

    #[pyo3(signature = (batch_id = 1))]
    pub fn commit_up_to(&self, py: Python<'_>, batch_id: u64) -> usize {
        py.detach(|| {
            let mut state = self.inner.lock();
            let newly = state.dedup.commit_up_to(batch_id);
            let n = newly.len();
            state.pending_wal_hashes.extend(newly);
            n
        })
    }

    #[pyo3(signature = (batch_id = 1))]
    pub fn rollback(&self, py: Python<'_>, batch_id: u64) -> usize {
        py.detach(|| self.inner.lock().dedup.rollback_batch(batch_id))
    }

    pub fn clear(&self, py: Python<'_>) {
        py.detach(|| self.inner.lock().dedup.clear_staged());
    }

    pub fn clear_all(&self, py: Python<'_>) {
        py.detach(|| {
            let mut state = self.inner.lock();
            state.dedup.clear_all();
            state.pending_wal_hashes.clear();
            state.wal_target = None;
        });
    }

    pub fn count(&self, py: Python<'_>) -> usize {
        py.detach(|| self.inner.lock().dedup.committed_len())
    }

    pub fn save_wal(&self, py: Python<'_>, path: &str) -> PyResult<usize> {
        let target = PathBuf::from(path);
        py.detach(|| -> std::io::Result<usize> {
            let mut state = self.inner.lock();
            if let Some(parent) = target.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent)?;
            }

            let same_target = state.wal_target.as_ref() == Some(&target) && target.exists();
            if same_target {
                append_wal_hashes(&target, &state.pending_wal_hashes)?;
                state.pending_wal_hashes.clear();
                return Ok(state.dedup.committed_len());
            }

            let hashes = state.dedup.export_committed_hashes();
            write_full_wal(&target, &hashes)?;
            state.pending_wal_hashes.clear();
            state.wal_target = Some(target);
            Ok(hashes.len())
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    pub fn load_wal(&self, py: Python<'_>, path: &str) -> PyResult<usize> {
        let target = PathBuf::from(path);
        let outcome = py
            .detach(|| read_wal_file(&target, path))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        match outcome {
            WalRead::Missing => Ok(0),
            WalRead::Corrupted(msg) => Err(PyValueError::new_err(msg)),
            WalRead::Loaded(hashes) => {
                let count = hashes.len();
                py.detach(|| {
                    let mut state = self.inner.lock();
                    state.dedup.import_committed_hashes(hashes);
                    state.pending_wal_hashes.clear();
                    state.wal_target = Some(target);
                });
                Ok(count)
            }
        }
    }

    pub fn export_hashes(&self, py: Python<'_>) -> Vec<u128> {
        py.detach(|| self.inner.lock().dedup.export_committed_hashes())
    }

    /// Truncates an on-disk `LMWL` WAL to at most `max_count` committed 16-byte hashes
    /// so crash recovery can discard hashes flushed before `.state.json` was committed.
    pub fn truncate_wal(&self, py: Python<'_>, path: &str, max_count: usize) -> PyResult<usize> {
        let target = PathBuf::from(path);
        py.detach(|| truncate_wal_file(&target, max_count))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

#[pyclass(frozen, name = "StateTracker")]
pub struct PyStateTracker {
    inner: Mutex<DualMetadataManager>,
}

#[pymethods]
impl PyStateTracker {
    #[new]
    pub fn new(py: Python<'_>, meta_path: &str) -> PyResult<Self> {
        let mgr = py
            .detach(|| DualMetadataManager::open(Path::new(meta_path)))
            .map_err(to_py_err)?;
        Ok(Self {
            inner: Mutex::new(mgr),
        })
    }

    pub fn read(&self, py: Python<'_>) -> PyStreamCoordinate {
        py.detach(|| self.inner.lock().committed_coordinate().into())
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
        py: Python<'_>,
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
        py.detach(|| self.inner.lock().stage_batch_in_ram(batch_id, coord));
    }

    #[pyo3(signature = (batch_id = 1))]
    pub fn commit(&self, py: Python<'_>, batch_id: u64) -> PyResult<PyStreamCoordinate> {
        py.detach(|| self.inner.lock().commit_gpu_batch(batch_id))
            .map(Into::into)
            .map_err(to_py_err)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn restore_snapshot(
        &self,
        py: Python<'_>,
        shard_id: u32,
        byte_offset: u64,
        row_group_idx: u32,
        page_idx: u32,
        row_offset_in_page: u32,
        global_row_idx: u64,
        batch_id: u64,
        consumed_tokens: u64,
    ) -> PyResult<PyStreamCoordinate> {
        let coord = StreamCoordinate {
            shard_id,
            byte_offset,
            row_group_idx,
            page_idx,
            row_offset_in_page,
            global_row_idx,
        };
        py.detach(|| {
            self.inner
                .lock()
                .restore_snapshot(batch_id, coord, consumed_tokens)
        })
        .map_err(to_py_err)?;
        Ok(coord.into())
    }

    #[pyo3(signature = (batch_id = 1))]
    pub fn commit_up_to(&self, py: Python<'_>, batch_id: u64) -> PyResult<PyStreamCoordinate> {
        py.detach(|| self.inner.lock().commit_up_to(batch_id))
            .map(Into::into)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (batch_id = 1))]
    pub fn rollback(&self, py: Python<'_>, batch_id: u64) -> PyResult<PyStreamCoordinate> {
        py.detach(|| self.inner.lock().rollback_staged_batch(batch_id))
            .map(Into::into)
            .map_err(to_py_err)
    }

    pub fn rollback_all(&self, py: Python<'_>) -> PyStreamCoordinate {
        py.detach(|| self.inner.lock().rollback_ram_to_committed().into())
    }

    pub fn committed_batch_id(&self, py: Python<'_>) -> u64 {
        py.detach(|| self.inner.lock().committed_batch_id())
    }

    pub fn consumed_tokens(&self, py: Python<'_>) -> u64 {
        py.detach(|| self.inner.lock().consumed_tokens())
    }

    pub fn set_consumed_tokens(&self, py: Python<'_>, tokens: u64) {
        py.detach(|| self.inner.lock().set_consumed_tokens(tokens));
    }
}
