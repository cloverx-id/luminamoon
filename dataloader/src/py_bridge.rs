use std::path::PathBuf;
use std::sync::Arc;

use hashbrown::HashMap;
use parking_lot::Mutex;
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyByteArray, PyBytes, PyDict, PyFloat, PyInt, PyList, PySequence, PyString, PyTuple,
};
use tokio::runtime::{Builder, Runtime};

use crate::coord::StreamCoordinate;
use crate::dedup::TwoStageDeduplicator;
use crate::error::{DataLoaderError, Result};
use crate::hf_hub::{HfCredentials, HfHubClient};
use crate::multimodal::{
    LocalMultimodalParquetSource, MultimodalBatch, MultimodalValue, RemoteMultimodalParquetSource,
    classify_string_or_json,
};
use crate::postprocess::PostProcessConfig;
use crate::preram::{PreRamConfig, PreRamDataLoader, ShardSourceKind};
use crate::source_local::FetchedBatch;

fn to_py_err(err: DataLoaderError) -> PyErr {
    match err {
        DataLoaderError::InvalidConfig(msg) | DataLoaderError::ColumnNotFound(msg) => {
            PyValueError::new_err(msg)
        }
        DataLoaderError::InvalidCoordinate => PyValueError::new_err(err.to_string()),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

pub fn parse_shard_specs(
    sources: &[String],
    parquet_column: &str,
    auth_token: Option<&str>,
    remote_chunk_bytes: u64,
) -> Result<Vec<ShardSourceKind>> {
    if sources.is_empty() {
        return Err(DataLoaderError::InvalidConfig(
            "sources list cannot be empty".to_owned(),
        ));
    }

    let mut out = Vec::with_capacity(sources.len());
    for raw in sources {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(DataLoaderError::InvalidConfig(
                "source entry cannot be empty".to_owned(),
            ));
        }

        let is_remote = trimmed.starts_with("hf://")
            || trimmed.starts_with("https://")
            || trimmed.starts_with("http://");

        let clean_path = trimmed.split('?').next().unwrap_or(trimmed);
        let is_parquet = clean_path.ends_with(".parquet") || clean_path.ends_with(".pq");

        let token_owned = auth_token.map(str::to_owned);

        let kind = match (is_remote, is_parquet) {
            (true, true) => ShardSourceKind::RemoteParquet {
                uri: trimmed.to_owned(),
                column: parquet_column.to_owned(),
                auth_token: token_owned,
            },
            (true, false) => ShardSourceKind::RemoteLine {
                uri: trimmed.to_owned(),
                auth_token: token_owned,
                chunk_bytes: remote_chunk_bytes.max(64),
            },
            (false, true) => ShardSourceKind::LocalParquet {
                path: PathBuf::from(trimmed),
                column: parquet_column.to_owned(),
            },
            (false, false) => ShardSourceKind::LocalLine {
                path: PathBuf::from(trimmed),
            },
        };
        out.push(kind);
    }

    Ok(out)
}

#[pyclass(frozen, name = "StreamCoordinate")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl From<PyStreamCoordinate> for StreamCoordinate {
    fn from(c: PyStreamCoordinate) -> Self {
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
        global_row_idx = 0
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

    pub fn to_bytes(&self) -> Vec<u8> {
        StreamCoordinate::from(*self).to_bytes().to_vec()
    }

    #[staticmethod]
    pub fn from_bytes(buf: &[u8]) -> PyResult<Self> {
        let arr: &[u8; StreamCoordinate::BYTE_LEN] = buf
            .try_into()
            .map_err(|_| to_py_err(DataLoaderError::InvalidCoordinate))?;
        Ok(Self::from(StreamCoordinate::from_bytes(arr)))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamCoordinate(shard_id={}, byte_offset={}, row_group_idx={}, page_idx={}, row_offset_in_page={}, global_row_idx={})",
            self.shard_id,
            self.byte_offset,
            self.row_group_idx,
            self.page_idx,
            self.row_offset_in_page,
            self.global_row_idx
        )
    }
}

#[pyclass(frozen, name = "DataBatch")]
#[derive(Debug, Clone)]
pub struct PyDataBatch {
    #[pyo3(get)]
    pub batch_id: u64,
    #[pyo3(get)]
    pub records: Vec<String>,
    #[pyo3(get)]
    pub next_coord: PyStreamCoordinate,
    #[pyo3(get)]
    pub is_eof: bool,
}

impl From<FetchedBatch> for PyDataBatch {
    fn from(b: FetchedBatch) -> Self {
        Self {
            batch_id: b.batch_id,
            records: b.records,
            next_coord: PyStreamCoordinate::from(b.next_coord),
            is_eof: b.is_eof,
        }
    }
}

#[pymethods]
impl PyDataBatch {
    fn __len__(&self) -> usize {
        self.records.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "DataBatch(batch_id={}, len={}, is_eof={})",
            self.batch_id,
            self.records.len(),
            self.is_eof
        )
    }
}

#[pyclass(frozen, subclass, name = "DataStreamLoader")]
pub struct PyDataStreamLoader {
    inner: Mutex<Option<Arc<PreRamDataLoader>>>,
}

impl PyDataStreamLoader {
    fn get_active(&self) -> PyResult<Arc<PreRamDataLoader>> {
        self.inner
            .lock()
            .as_ref()
            .cloned()
            .ok_or_else(|| PyRuntimeError::new_err("DataStreamLoader is already closed"))
    }
}

#[pymethods]
impl PyDataStreamLoader {
    #[new]
    #[pyo3(signature = (
        sources,
        checkpoint_path,
        batch_size = 64,
        preram_batches = 100,
        refill_trigger = 10,
        deduplicate = true,
        auto_refill = true,
        parquet_column = "text",
        json_extract_key = None,
        trim_whitespace = true,
        normalize_newlines = true,
        min_byte_len = 1,
        max_byte_len = 0,
        truncate_over_max = false,
        auth_token = None,
        remote_chunk_bytes = 262_144,
        download_workers = 4
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        py: Python<'_>,
        sources: Vec<String>,
        checkpoint_path: String,
        batch_size: usize,
        preram_batches: usize,
        refill_trigger: usize,
        deduplicate: bool,
        auto_refill: bool,
        parquet_column: &str,
        json_extract_key: Option<String>,
        trim_whitespace: bool,
        normalize_newlines: bool,
        min_byte_len: usize,
        max_byte_len: usize,
        truncate_over_max: bool,
        auth_token: Option<String>,
        remote_chunk_bytes: u64,
        download_workers: usize,
    ) -> PyResult<Self> {
        let shards = parse_shard_specs(
            &sources,
            parquet_column,
            auth_token.as_deref(),
            remote_chunk_bytes,
        )
        .map_err(to_py_err)?;

        let cfg = PreRamConfig {
            capacity_batches: preram_batches,
            refill_trigger,
            batch_size,
            deduplicate,
            auto_refill,
            download_workers: download_workers.max(1),
            postprocess: PostProcessConfig {
                trim_whitespace,
                normalize_newlines,
                min_byte_len,
                max_byte_len,
                truncate_over_max,
                json_extract_key,
            },
        };

        let loader = py
            .detach(|| PreRamDataLoader::open(checkpoint_path, shards, cfg))
            .map_err(to_py_err)?;

        Ok(Self {
            inner: Mutex::new(Some(Arc::new(loader))),
        })
    }

    pub fn next_batch(&self, py: Python<'_>) -> PyResult<Option<PyDataBatch>> {
        let loader = self.get_active()?;
        let res = py.detach(|| loader.next_batch()).map_err(to_py_err)?;
        Ok(res.map(PyDataBatch::from))
    }

    pub fn commit_gpu_batch(&self, py: Python<'_>, batch_id: u64) -> PyResult<PyStreamCoordinate> {
        let loader = self.get_active()?;
        let coord = py
            .detach(|| loader.commit_gpu_batch(batch_id))
            .map_err(to_py_err)?;
        Ok(PyStreamCoordinate::from(coord))
    }

    pub fn mark_in_gpu(&self, py: Python<'_>, batch_id: u64) -> PyResult<PyStreamCoordinate> {
        self.commit_gpu_batch(py, batch_id)
    }

    pub fn rollback_staged_batch(
        &self,
        py: Python<'_>,
        batch_id: u64,
    ) -> PyResult<PyStreamCoordinate> {
        let loader = self.get_active()?;
        let coord = py
            .detach(|| loader.rollback_staged_batch(batch_id))
            .map_err(to_py_err)?;
        Ok(PyStreamCoordinate::from(coord))
    }

    pub fn wait_for_refill(&self, py: Python<'_>) -> PyResult<usize> {
        let loader = self.get_active()?;
        py.detach(|| loader.wait_for_refill()).map_err(to_py_err)
    }

    pub fn trigger_refill(&self) -> PyResult<()> {
        let loader = self.get_active()?;
        loader.trigger_refill();
        Ok(())
    }

    pub fn buffered_batches(&self) -> PyResult<usize> {
        let loader = self.get_active()?;
        Ok(loader.buffered_batches())
    }

    pub fn download_workers(&self) -> PyResult<usize> {
        let loader = self.get_active()?;
        Ok(loader.download_workers())
    }

    pub fn committed_batch_id(&self) -> PyResult<u64> {
        let loader = self.get_active()?;
        Ok(loader.committed_batch_id())
    }

    pub fn committed_coordinate(&self) -> PyResult<PyStreamCoordinate> {
        let loader = self.get_active()?;
        Ok(PyStreamCoordinate::from(loader.committed_coordinate()))
    }

    pub fn in_flight_coordinate(&self) -> PyResult<PyStreamCoordinate> {
        let loader = self.get_active()?;
        Ok(PyStreamCoordinate::from(loader.in_flight_coordinate()))
    }

    pub fn close(&self, py: Python<'_>) {
        let taken = self.inner.lock().take();
        if let Some(loader) = taken {
            py.detach(|| drop(loader));
        }
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        self.close(py);
        false
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<PyDataBatch> {
        match self.next_batch(py)? {
            Some(batch) => Ok(batch),
            None => Err(PyStopIteration::new_err(())),
        }
    }
}

#[pyclass(frozen, name = "MultimodalBatch")]
#[derive(Debug, Clone)]
pub struct PyMultimodalBatch {
    #[pyo3(get)]
    pub batch_id: u64,
    #[pyo3(get)]
    pub num_rows: usize,
    #[pyo3(get)]
    pub next_coord: PyStreamCoordinate,
    #[pyo3(get)]
    pub is_eof: bool,
    column_order: Vec<String>,
    columns: HashMap<String, Vec<MultimodalValue>>,
}

impl From<MultimodalBatch> for PyMultimodalBatch {
    fn from(b: MultimodalBatch) -> Self {
        Self {
            batch_id: b.batch_id,
            num_rows: b.num_rows,
            next_coord: PyStreamCoordinate::from(b.next_coord),
            is_eof: b.is_eof,
            column_order: b.column_order,
            columns: b.columns,
        }
    }
}

#[pymethods]
impl PyMultimodalBatch {
    pub fn column_names(&self) -> Vec<String> {
        self.column_order.clone()
    }

    pub fn column(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let vals = self
            .columns
            .get(name)
            .ok_or_else(|| PyValueError::new_err(format!("Column not found: {name}")))?;
        let list = PyList::empty(py);
        for v in vals {
            match v {
                MultimodalValue::Null => list.append(py.None())?,
                MultimodalValue::Text(s) | MultimodalValue::Json(s) => {
                    list.append(PyString::new(py, s))?;
                }
                MultimodalValue::Bytes(b) => {
                    list.append(PyBytes::new(py, b))?;
                }
                MultimodalValue::Int(i) => {
                    list.append(*i)?;
                }
                MultimodalValue::Float(f) => {
                    list.append(*f)?;
                }
                MultimodalValue::Bool(b) => {
                    list.append(*b)?;
                }
            }
        }
        Ok(list.into_any().unbind())
    }

    pub fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for k in &self.column_order {
            dict.set_item(k, self.column(py, k)?)?;
        }
        Ok(dict.into_any().unbind())
    }

    fn __len__(&self) -> usize {
        self.num_rows
    }

    fn __repr__(&self) -> String {
        format!(
            "MultimodalBatch(batch_id={}, num_rows={}, columns={}, is_eof={})",
            self.batch_id,
            self.num_rows,
            self.columns.len(),
            self.is_eof
        )
    }
}

fn py_any_to_multimodal_value(py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<MultimodalValue> {
    if item.is_none() {
        return Ok(MultimodalValue::Null);
    }
    if item.is_instance_of::<PyBool>() {
        return Ok(MultimodalValue::Bool(item.extract::<bool>()?));
    }
    if item.is_instance_of::<PyInt>() {
        return Ok(MultimodalValue::Int(item.extract::<i64>()?));
    }
    if item.is_instance_of::<PyFloat>() {
        return Ok(MultimodalValue::Float(item.extract::<f64>()?));
    }
    if item.is_instance_of::<PyBytes>() || item.is_instance_of::<PyByteArray>() {
        return Ok(MultimodalValue::Bytes(item.extract::<Vec<u8>>()?));
    }
    if item.is_instance_of::<PyDict>() || item.is_instance_of::<PyList>() {
        let json_mod = py.import("json")?;
        let dumped = json_mod
            .call_method1("dumps", (item,))?
            .extract::<String>()?;
        return Ok(MultimodalValue::Json(dumped));
    }
    if item.is_instance_of::<PyString>() {
        let s = item.extract::<&str>()?;
        return Ok(classify_string_or_json(s));
    }
    let s = item.str()?.to_str()?.to_owned();
    Ok(MultimodalValue::Text(s))
}

fn extract_py_columns_table(
    py: Python<'_>,
    columns: &Bound<'_, PyAny>,
) -> PyResult<Vec<(String, Vec<MultimodalValue>)>> {
    if let Ok(dict) = columns.cast::<PyDict>() {
        let mut out = Vec::with_capacity(dict.len());
        for (k, v) in dict.iter() {
            let name = k.extract::<String>()?;
            let seq = v.cast::<PySequence>()?;
            let len = seq.len()?;
            let mut col_vals = Vec::with_capacity(len);
            for idx in 0..len {
                let item = seq.get_item(idx)?;
                col_vals.push(py_any_to_multimodal_value(py, &item)?);
            }
            out.push((name, col_vals));
        }
        return Ok(out);
    }

    let outer = columns.cast::<PySequence>()?;
    let outer_len = outer.len()?;
    if outer_len > 0 && outer.get_item(0)?.is_instance_of::<PyDict>() {
        let mut col_names: Vec<String> = Vec::new();
        let mut col_index: HashMap<String, usize> = HashMap::new();
        let mut cols_data: Vec<Vec<MultimodalValue>> = Vec::new();

        for r_idx in 0..outer_len {
            let row_obj = outer.get_item(r_idx)?;
            let row_dict = row_obj.cast::<PyDict>().map_err(|_| {
                PyValueError::new_err("All rows in row-oriented table must be dicts")
            })?;

            for (k, _) in row_dict.iter() {
                let key_str = k.extract::<String>()?;
                if !col_index.contains_key(&key_str) {
                    let new_idx = col_names.len();
                    col_names.push(key_str.clone());
                    col_index.insert(key_str, new_idx);
                    cols_data.push(vec![MultimodalValue::Null; r_idx]);
                }
            }

            for (c_idx, col_name) in col_names.iter().enumerate() {
                let val = match row_dict.get_item(col_name)? {
                    Some(item) => py_any_to_multimodal_value(py, &item)?,
                    None => MultimodalValue::Null,
                };
                cols_data[c_idx].push(val);
            }
        }

        return Ok(col_names.into_iter().zip(cols_data).collect());
    }

    let mut out = Vec::with_capacity(outer_len);
    for idx in 0..outer_len {
        let pair = outer.get_item(idx)?;
        let tup = pair.cast::<PyTuple>()?;
        if tup.len() != 2 {
            return Err(PyValueError::new_err(
                "Each column entry must be a (column_name, values) pair",
            ));
        }
        let name = tup.get_item(0)?.extract::<String>()?;
        let vals_obj = tup.get_item(1)?;
        let seq = vals_obj.cast::<PySequence>()?;
        let len = seq.len()?;
        let mut col_vals = Vec::with_capacity(len);
        for r_idx in 0..len {
            let item = seq.get_item(r_idx)?;
            col_vals.push(py_any_to_multimodal_value(py, &item)?);
        }
        out.push((name, col_vals));
    }
    Ok(out)
}

#[derive(Clone)]
enum MultimodalSourceBackend {
    Local(Arc<LocalMultimodalParquetSource>),
    Remote {
        source: Arc<RemoteMultimodalParquetSource>,
        rt: Arc<Runtime>,
    },
}

struct MultimodalLoaderState {
    cursor: StreamCoordinate,
    next_batch_id: u64,
    dedup: Option<TwoStageDeduplicator>,
    batch_size: usize,
    is_eof: bool,
}

#[pyclass(frozen, subclass, name = "MultimodalParquetLoader")]
pub struct PyMultimodalParquetLoader {
    backend: MultimodalSourceBackend,
    state: Mutex<MultimodalLoaderState>,
}

#[pymethods]
impl PyMultimodalParquetLoader {
    #[new]
    #[pyo3(signature = (path, columns = None, batch_size = 64, deduplicate = true, shard_id = 0, auth_token = None, download_workers = 4))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        py: Python<'_>,
        path: String,
        columns: Option<Vec<String>>,
        batch_size: usize,
        deduplicate: bool,
        shard_id: u32,
        auth_token: Option<String>,
        download_workers: usize,
    ) -> PyResult<Self> {
        let cols = columns.unwrap_or_default();
        let trimmed = path.trim().to_owned();
        let is_remote = trimmed.starts_with("hf://")
            || trimmed.starts_with("https://")
            || trimmed.starts_with("http://");
        let workers = if download_workers == 0 { 4 } else { download_workers };

        let backend = if is_remote {
            let (src, rt) = py
                .detach(|| -> Result<(RemoteMultimodalParquetSource, Runtime)> {
                    let rt = Builder::new_multi_thread()
                        .worker_threads(workers.max(1))
                        .thread_name("luminamoon-mm-worker")
                        .enable_all()
                        .build()?;
                    let src = rt
                        .block_on(RemoteMultimodalParquetSource::connect(
                            shard_id, &trimmed, cols, auth_token,
                        ))?
                        .with_download_workers(workers);
                    Ok((src, rt))
                })
                .map_err(to_py_err)?;
            MultimodalSourceBackend::Remote {
                source: Arc::new(src),
                rt: Arc::new(rt),
            }
        } else {
            let src = py
                .detach(|| LocalMultimodalParquetSource::open(shard_id, trimmed, cols))
                .map_err(to_py_err)?;
            MultimodalSourceBackend::Local(Arc::new(src))
        };

        let init_coord = StreamCoordinate {
            shard_id,
            ..StreamCoordinate::default()
        };
        Ok(Self {
            backend,
            state: Mutex::new(MultimodalLoaderState {
                cursor: init_coord,
                next_batch_id: 1,
                dedup: if deduplicate {
                    Some(TwoStageDeduplicator::new())
                } else {
                    None
                },
                batch_size: batch_size.max(1),
                is_eof: false,
            }),
        })
    }

    pub fn next_batch(&self, py: Python<'_>) -> PyResult<Option<PyMultimodalBatch>> {
        let backend = self.backend.clone();
        let res = py.detach(|| -> Result<Option<MultimodalBatch>> {
            let mut st = self.state.lock();
            if st.is_eof {
                return Ok(None);
            }
            let batch_id = st.next_batch_id;
            let cursor = st.cursor;
            let batch_size = st.batch_size;
            let batch = match &backend {
                MultimodalSourceBackend::Local(src) => {
                    src.read_multimodal_batch(batch_id, cursor, batch_size, st.dedup.as_mut())?
                }
                MultimodalSourceBackend::Remote { source, rt } => rt.block_on(
                    source.read_multimodal_batch(batch_id, cursor, batch_size, st.dedup.as_mut()),
                )?,
            };
            st.cursor = batch.next_coord;
            st.next_batch_id += 1;
            if batch.is_eof {
                st.is_eof = true;
            }
            if batch.num_rows == 0 && batch.is_eof {
                return Ok(None);
            }
            Ok(Some(batch))
        });
        Ok(res.map_err(to_py_err)?.map(PyMultimodalBatch::from))
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<PyMultimodalBatch> {
        match self.next_batch(py)? {
            Some(b) => Ok(b),
            None => Err(PyStopIteration::new_err(())),
        }
    }
}

#[pyclass(frozen, subclass, name = "HfHubUploader")]
pub struct PyHfHubUploader {
    client: HfHubClient,
    rt: Runtime,
}

#[pymethods]
impl PyHfHubUploader {
    #[new]
    #[pyo3(signature = (token = None, username = None, org = None, endpoint = None))]
    pub fn new(
        token: Option<String>,
        username: Option<String>,
        org: Option<String>,
        endpoint: Option<String>,
    ) -> PyResult<Self> {
        let creds = HfCredentials::new(token, username, org, endpoint);
        let client = HfHubClient::new(creds).map_err(to_py_err)?;
        let rt = Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| to_py_err(DataLoaderError::Io(e)))?;
        Ok(Self { client, rt })
    }

    #[pyo3(signature = (repo, repo_type = "dataset", private = true, exist_ok = true))]
    pub fn create_repo(
        &self,
        py: Python<'_>,
        repo: &str,
        repo_type: &str,
        private: bool,
        exist_ok: bool,
    ) -> PyResult<String> {
        py.detach(|| {
            self.rt
                .block_on(self.client.create_repo(repo, repo_type, private, exist_ok))
        })
        .map_err(to_py_err)
    }

    #[pyo3(signature = (repo, local_path, path_in_repo, revision = "main", commit_message = "Upload file via LuminaMoon"))]
    pub fn upload_file(
        &self,
        py: Python<'_>,
        repo: &str,
        local_path: &str,
        path_in_repo: &str,
        revision: &str,
        commit_message: &str,
    ) -> PyResult<String> {
        py.detach(|| {
            self.rt.block_on(self.client.upload_file(
                repo,
                local_path,
                path_in_repo,
                revision,
                commit_message,
            ))
        })
        .map_err(to_py_err)
    }

    #[pyo3(signature = (repo, path_in_repo, data, revision = "main", commit_message = "Upload bytes via LuminaMoon"))]
    pub fn upload_bytes(
        &self,
        py: Python<'_>,
        repo: &str,
        path_in_repo: &str,
        data: &[u8],
        revision: &str,
        commit_message: &str,
    ) -> PyResult<String> {
        py.detach(|| {
            self.rt.block_on(self.client.upload_bytes(
                repo,
                path_in_repo,
                data,
                revision,
                commit_message,
            ))
        })
        .map_err(to_py_err)
    }

    #[pyo3(signature = (repo, path_in_repo, columns, revision = "main", commit_message = "Upload dataset shard", hf_image_struct = true))]
    #[allow(clippy::too_many_arguments)]
    pub fn upload_table(
        &self,
        py: Python<'_>,
        repo: &str,
        path_in_repo: &str,
        columns: &Bound<'_, PyAny>,
        revision: &str,
        commit_message: &str,
        hf_image_struct: bool,
    ) -> PyResult<String> {
        let rust_cols = extract_py_columns_table(py, columns)?;
        py.detach(|| {
            self.rt.block_on(self.client.upload_table_auto_format(
                repo,
                path_in_repo,
                &rust_cols,
                revision,
                commit_message,
                hf_image_struct,
            ))
        })
        .map_err(to_py_err)
    }

    #[pyo3(signature = (repo, path_in_repo, columns, revision = "main", commit_message = "Upload JSONL shard"))]
    pub fn upload_jsonl(
        &self,
        py: Python<'_>,
        repo: &str,
        path_in_repo: &str,
        columns: &Bound<'_, PyAny>,
        revision: &str,
        commit_message: &str,
    ) -> PyResult<String> {
        let rust_cols = extract_py_columns_table(py, columns)?;
        py.detach(|| {
            self.rt.block_on(self.client.upload_multimodal_jsonl(
                repo,
                path_in_repo,
                &rust_cols,
                revision,
                commit_message,
            ))
        })
        .map_err(to_py_err)
    }
}

#[pyfunction]
pub fn dataloader_cpu_isa() -> &'static str {
    crate::simd_cpu::detect_cpu_isa().as_str()
}

#[pyfunction]
#[pyo3(signature = (columns, path_in_repo="part.parquet", hf_image_struct=true))]
pub fn serialize_table_to_bytes(
    py: Python<'_>,
    columns: &Bound<'_, PyAny>,
    path_in_repo: &str,
    hf_image_struct: bool,
) -> PyResult<Vec<u8>> {
    let rust_cols = extract_py_columns_table(py, columns)?;
    let lower_path = path_in_repo.trim().to_ascii_lowercase();
    py.detach(|| {
        if lower_path.ends_with(".jsonl") || lower_path.ends_with(".ndjson") {
            crate::multimodal::serialize_columns_to_jsonl_bytes(&rust_cols)
        } else if lower_path.ends_with(".json") {
            crate::multimodal::serialize_columns_to_json_bytes(&rust_cols)
        } else {
            crate::multimodal::serialize_columns_to_parquet_bytes_with_options(
                &rust_cols,
                hf_image_struct,
            )
        }
    })
    .map_err(to_py_err)
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

pub fn register_dataloader_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    parent_module.add_class::<PyStreamCoordinate>()?;
    parent_module.add_class::<PyDataBatch>()?;
    parent_module.add_class::<PyDataStreamLoader>()?;
    parent_module.add_class::<PyMultimodalBatch>()?;
    parent_module.add_class::<PyMultimodalParquetLoader>()?;
    parent_module.add_class::<PyHfHubUploader>()?;
    parent_module.add_function(wrap_pyfunction!(dataloader_cpu_isa, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(serialize_table_to_bytes, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(postprocess_and_dedup_records, parent_module)?)?;
    Ok(())
}
