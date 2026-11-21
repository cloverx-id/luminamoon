use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use parking_lot::Mutex;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::coord::StreamCoordinate;
use crate::dedup::TwoStageDeduplicator;
use crate::metadata::DualMetadataManager;

use super::to_py_err;

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
    pub(super) inner: Mutex<TwoStageDeduplicator>,
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

    #[pyo3(signature = (payload, batch_id = 1))]
    pub fn try_add(&self, payload: &[u8], batch_id: u64) -> bool {
        let mut dedup = self.inner.lock();
        dedup.try_stage(batch_id, payload).is_some()
    }

    #[pyo3(signature = (records, batch_id = 1))]
    pub fn filter_batch(&self, py: Python<'_>, records: Vec<String>, batch_id: u64) -> Vec<String> {
        py.detach(|| {
            let mut dedup = self.inner.lock();
            let mut out = Vec::with_capacity(records.len());
            for rec in records {
                if dedup.try_stage(batch_id, rec.as_bytes()).is_some() {
                    out.push(rec);
                }
            }
            out
        })
    }

    #[pyo3(signature = (batch_id = 1))]
    pub fn commit(&self, batch_id: u64) -> usize {
        let mut dedup = self.inner.lock();
        dedup.commit_batch(batch_id).len()
    }

    #[pyo3(signature = (batch_id = 1))]
    pub fn rollback(&self, batch_id: u64) -> usize {
        let mut dedup = self.inner.lock();
        dedup.rollback_batch(batch_id)
    }

    pub fn clear(&self) {
        let mut dedup = self.inner.lock();
        dedup.clear_staged();
    }

    pub fn count(&self) -> usize {
        self.inner.lock().committed_len()
    }

    pub fn save_wal(&self, py: Python<'_>, path: &str) -> PyResult<usize> {
        py.detach(|| {
            let hashes = self.inner.lock().export_committed_hashes();
            let mut buf = Vec::with_capacity(hashes.len() * 8);
            for h in &hashes {
                buf.extend_from_slice(&h.to_le_bytes());
            }
            let target = Path::new(path);
            if let Some(parent) = target.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent)?;
            }
            let mut tmp_os = target.as_os_str().to_owned();
            tmp_os.push(format!(".tmp.{}", std::process::id()));
            let tmp_path = std::path::PathBuf::from(tmp_os);
            {
                let mut file = File::create(&tmp_path)?;
                file.write_all(&buf)?;
                file.sync_data()?;
            }
            std::fs::rename(&tmp_path, target)?;
            Ok::<usize, std::io::Error>(hashes.len())
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    pub fn load_wal(&self, py: Python<'_>, path: &str) -> PyResult<usize> {
        let buf = py
            .detach(|| -> std::io::Result<Option<Vec<u8>>> {
                let mut file = match File::open(path) {
                    Ok(f) => f,
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                    Err(e) => return Err(e),
                };
                let mut buf = Vec::new();
                file.read_to_end(&mut buf)?;
                Ok(Some(buf))
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let Some(buf) = buf else {
            return Ok(0);
        };
        if buf.len() % 8 != 0 {
            return Err(PyValueError::new_err(format!(
                "Corrupted WAL file '{}': byte length {} is not a multiple of 8",
                path,
                buf.len()
            )));
        }
        let count = buf.len() / 8;
        py.detach(|| {
            let mut dedup = self.inner.lock();
            let (chunks, _) = buf.as_chunks::<8>();
            dedup.import_committed_hashes(chunks.iter().map(|c| u64::from_le_bytes(*c)));
        });
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
    pub fn new(py: Python<'_>, meta_path: &str) -> PyResult<Self> {
        let mgr = py
            .detach(|| DualMetadataManager::open(meta_path))
            .map_err(to_py_err)?;
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
    pub fn commit(&self, py: Python<'_>, batch_id: u64) -> PyResult<PyStreamCoordinate> {
        let c = py
            .detach(|| self.inner.lock().commit_gpu_batch(batch_id))
            .map_err(to_py_err)?;
        Ok(PyStreamCoordinate {
            shard_id: c.shard_id,
            byte_offset: c.byte_offset,
            row_group_idx: c.row_group_idx,
            page_idx: c.page_idx,
            row_offset_in_page: c.row_offset_in_page,
            global_row_idx: c.global_row_idx,
        })
    }

    #[pyo3(signature = (batch_id = 1))]
    pub fn rollback(&self, batch_id: u64) -> PyResult<PyStreamCoordinate> {
        let c = self
            .inner
            .lock()
            .rollback_staged_batch(batch_id)
            .map_err(to_py_err)?;
        Ok(PyStreamCoordinate {
            shard_id: c.shard_id,
            byte_offset: c.byte_offset,
            row_group_idx: c.row_group_idx,
            page_idx: c.page_idx,
            row_offset_in_page: c.row_offset_in_page,
            global_row_idx: c.global_row_idx,
        })
    }

    pub fn rollback_all(&self) -> PyStreamCoordinate {
        let c = self.inner.lock().rollback_ram_to_committed();
        PyStreamCoordinate {
            shard_id: c.shard_id,
            byte_offset: c.byte_offset,
            row_group_idx: c.row_group_idx,
            page_idx: c.page_idx,
            row_offset_in_page: c.row_offset_in_page,
            global_row_idx: c.global_row_idx,
        }
    }

    pub fn committed_batch_id(&self) -> u64 {
        self.inner.lock().committed_batch_id()
    }
}
