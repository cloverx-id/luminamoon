use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use parking_lot::{Condvar, Mutex};
use tokio::runtime::{Builder, Runtime};

use crate::coord::StreamCoordinate;
use crate::dedup::TwoStageDeduplicator;
use crate::error::{DataLoaderError, Result};
use crate::metadata::DualMetadataManager;
use crate::postprocess::PostProcessConfig;
use crate::source_hf::{RemoteLineSource, RemoteParquetSource};
use crate::source_local::{FetchedBatch, LocalMmapLineSource, LocalParquetSource};

#[derive(Debug, Clone)]
pub enum ShardSourceKind {
    LocalLine {
        path: PathBuf,
    },
    LocalParquet {
        path: PathBuf,
        column: String,
    },
    RemoteLine {
        uri: String,
        auth_token: Option<String>,
        chunk_bytes: u64,
    },
    RemoteParquet {
        uri: String,
        column: String,
        auth_token: Option<String>,
    },
}

#[derive(Debug, Clone)]
pub struct PreRamConfig {
    pub capacity_batches: usize,
    pub refill_trigger: usize,
    pub batch_size: usize,
    pub deduplicate: bool,
    pub auto_refill: bool,
    pub download_workers: usize,
    pub postprocess: PostProcessConfig,
}

impl Default for PreRamConfig {
    fn default() -> Self {
        Self {
            capacity_batches: 100,
            refill_trigger: 10,
            batch_size: 64,
            deduplicate: true,
            auto_refill: true,
            download_workers: std::thread::available_parallelism()
                .map(|n| n.get().clamp(2, 8))
                .unwrap_or(4),
            postprocess: PostProcessConfig::default(),
        }
    }
}

enum ActiveShard {
    LocalLine(LocalMmapLineSource),
    LocalParquet(LocalParquetSource),
    RemoteLine(RemoteLineSource),
    RemoteParquet(RemoteParquetSource),
}

impl ActiveShard {
    fn open(
        rt: &Runtime,
        shard_id: u32,
        spec: &ShardSourceKind,
        download_workers: usize,
    ) -> Result<Self> {
        match spec {
            ShardSourceKind::LocalLine { path } => {
                Ok(Self::LocalLine(LocalMmapLineSource::open(shard_id, path)?))
            }
            ShardSourceKind::LocalParquet { path, column } => Ok(Self::LocalParquet(
                LocalParquetSource::open(shard_id, path, column.clone())?,
            )),
            ShardSourceKind::RemoteLine {
                uri,
                auth_token,
                chunk_bytes,
            } => {
                let src = rt
                    .block_on(RemoteLineSource::connect(
                        shard_id,
                        uri,
                        auth_token.clone(),
                        *chunk_bytes,
                    ))?
                    .with_download_workers(download_workers);
                Ok(Self::RemoteLine(src))
            }
            ShardSourceKind::RemoteParquet {
                uri,
                column,
                auth_token,
            } => {
                let src = rt
                    .block_on(RemoteParquetSource::connect(
                        shard_id,
                        uri,
                        column.clone(),
                        auth_token.clone(),
                    ))?
                    .with_download_workers(download_workers);
                Ok(Self::RemoteParquet(src))
            }
        }
    }

    fn read_batch(
        &self,
        rt: &Runtime,
        batch_id: u64,
        start_coord: StreamCoordinate,
        batch_size: usize,
        post_cfg: &PostProcessConfig,
        dedup: Option<&mut TwoStageDeduplicator>,
    ) -> Result<FetchedBatch> {
        match self {
            Self::LocalLine(src) => {
                src.read_batch(batch_id, start_coord, batch_size, post_cfg, dedup)
            }
            Self::LocalParquet(src) => {
                src.read_batch(batch_id, start_coord, batch_size, post_cfg, dedup)
            }
            Self::RemoteLine(src) => {
                rt.block_on(src.read_batch(batch_id, start_coord, batch_size, post_cfg, dedup))
            }
            Self::RemoteParquet(src) => {
                rt.block_on(src.read_batch(batch_id, start_coord, batch_size, post_cfg, dedup))
            }
        }
    }
}

struct QueueState {
    queue: VecDeque<FetchedBatch>,
    next_batch_id: u64,
    refilling: bool,
    dataset_eof: bool,
    worker_idle: bool,
    worker_error: Option<String>,
    shutdown: bool,
}

struct SyncPrimitives {
    state: Mutex<QueueState>,
    meta: Mutex<DualMetadataManager>,
    worker_cv: Condvar,
    client_cv: Condvar,
}

pub struct PreRamDataLoader {
    config: PreRamConfig,
    sync: Arc<SyncPrimitives>,
    worker_handle: Option<JoinHandle<()>>,
}

impl PreRamDataLoader {
    pub fn open(
        meta_path: impl AsRef<Path>,
        shards: Vec<ShardSourceKind>,
        mut config: PreRamConfig,
    ) -> Result<Self> {
        if shards.is_empty() {
            return Err(DataLoaderError::InvalidConfig(
                "At least one shard source must be provided".to_owned(),
            ));
        }
        config.capacity_batches = config.capacity_batches.max(1);
        config.refill_trigger = config.refill_trigger.clamp(1, config.capacity_batches);
        config.batch_size = config.batch_size.max(1);
        config.download_workers = config.download_workers.max(1);

        let meta = DualMetadataManager::open(meta_path)?;
        let initial_batch_id = meta.committed_batch_id() + 1;
        let start_shard = meta.committed_coordinate().shard_id as usize;
        let dataset_eof = start_shard >= shards.len();

        let queue_state = QueueState {
            queue: VecDeque::with_capacity(config.capacity_batches),
            next_batch_id: initial_batch_id,
            refilling: !dataset_eof,
            dataset_eof,
            worker_idle: false,
            worker_error: None,
            shutdown: false,
        };

        let sync = Arc::new(SyncPrimitives {
            state: Mutex::new(queue_state),
            meta: Mutex::new(meta),
            worker_cv: Condvar::new(),
            client_cv: Condvar::new(),
        });

        let worker_sync = sync.clone();
        let worker_cfg = config.clone();
        let handle = thread::Builder::new()
            .name("luminamoon-preram-worker".to_owned())
            .spawn(move || {
                Self::worker_loop(worker_sync, shards, worker_cfg);
            })?;

        Ok(Self {
            config,
            sync,
            worker_handle: Some(handle),
        })
    }

    fn worker_loop(
        sync: Arc<SyncPrimitives>,
        shards: Vec<ShardSourceKind>,
        config: PreRamConfig,
    ) {
        let rt = match Builder::new_multi_thread()
            .worker_threads(config.download_workers.max(1))
            .thread_name("luminamoon-dl-worker")
            .enable_all()
            .build()
        {
            Ok(r) => r,
            Err(e) => {
                let mut guard = sync.state.lock();
                guard.worker_error = Some(e.to_string());
                guard.worker_idle = true;
                sync.client_cv.notify_all();
                return;
            }
        };

        let mut active_shard: Option<(u32, ActiveShard)> = None;

        loop {
            let batch_id = {
                let mut guard = sync.state.lock();
                while !guard.shutdown
                    && (guard.dataset_eof
                        || !guard.refilling
                        || guard.queue.len() >= config.capacity_batches)
                {
                    if guard.queue.len() >= config.capacity_batches {
                        guard.refilling = false;
                    }
                    guard.worker_idle = true;
                    sync.client_cv.notify_all();
                    sync.worker_cv.wait(&mut guard);
                }

                if guard.shutdown {
                    return;
                }

                guard.worker_idle = false;
                guard.next_batch_id
            };

            let mut coord = sync.meta.lock().in_flight_coordinate();
            let mut accumulated_records = Vec::with_capacity(config.batch_size);
            let mut batch_eof = false;

            while accumulated_records.len() < config.batch_size {
                if sync.state.lock().shutdown {
                    return;
                }

                if (coord.shard_id as usize) >= shards.len() {
                    batch_eof = true;
                    break;
                }

                let need_open = active_shard
                    .as_ref()
                    .is_none_or(|(id, _)| *id != coord.shard_id);

                if need_open {
                    match ActiveShard::open(
                        &rt,
                        coord.shard_id,
                        &shards[coord.shard_id as usize],
                        config.download_workers,
                    ) {
                        Ok(opened) => {
                            active_shard = Some((coord.shard_id, opened));
                        }
                        Err(e) => {
                            let mut guard = sync.state.lock();
                            guard.worker_error = Some(e.to_string());
                            guard.refilling = false;
                            guard.worker_idle = true;
                            sync.client_cv.notify_all();
                            return;
                        }
                    }
                }

                let remaining_needed = config.batch_size - accumulated_records.len();
                let reader = &active_shard.as_ref().unwrap().1;
                let fetch_res = {
                    let mut meta_guard = sync.meta.lock();
                    let dedup_opt = if config.deduplicate {
                        Some(&mut meta_guard.deduplicator)
                    } else {
                        None
                    };
                    reader.read_batch(
                        &rt,
                        batch_id,
                        coord,
                        remaining_needed,
                        &config.postprocess,
                        dedup_opt,
                    )
                };

                match fetch_res {
                    Ok(mut fetched) => {
                        if fetched.is_eof && (coord.shard_id as usize + 1) < shards.len() {
                            fetched.next_coord = StreamCoordinate {
                                shard_id: coord.shard_id + 1,
                                byte_offset: 0,
                                row_group_idx: 0,
                                page_idx: 0,
                                row_offset_in_page: 0,
                                global_row_idx: fetched.next_coord.global_row_idx,
                            };
                            fetched.is_eof = false;
                        }

                        coord = fetched.next_coord;
                        accumulated_records.append(&mut fetched.records);
                        sync.meta.lock().set_in_flight_coordinate(coord);

                        if fetched.is_eof {
                            batch_eof = true;
                            break;
                        }
                    }
                    Err(e) => {
                        let mut guard = sync.state.lock();
                        guard.worker_error = Some(e.to_string());
                        guard.refilling = false;
                        guard.worker_idle = true;
                        sync.client_cv.notify_all();
                        return;
                    }
                }
            }

            if accumulated_records.is_empty() {
                let mut guard = sync.state.lock();
                guard.dataset_eof = true;
                guard.refilling = false;
                sync.client_cv.notify_all();
            } else {
                sync.meta.lock().stage_batch_in_ram(batch_id, coord);
                let fetched_batch = FetchedBatch {
                    batch_id,
                    records: accumulated_records,
                    next_coord: coord,
                    is_eof: batch_eof,
                };
                let mut guard = sync.state.lock();
                guard.next_batch_id += 1;
                guard.queue.push_back(fetched_batch);
                if batch_eof {
                    guard.dataset_eof = true;
                    guard.refilling = false;
                } else if guard.queue.len() >= config.capacity_batches {
                    guard.refilling = false;
                }
                sync.client_cv.notify_all();
            }
        }
    }

    pub fn wait_for_refill(&self) -> Result<usize> {
        let mut guard = self.sync.state.lock();
        while !guard.worker_idle && guard.worker_error.is_none() {
            self.sync.client_cv.wait(&mut guard);
        }
        if let Some(err) = &guard.worker_error {
            return Err(DataLoaderError::InvalidConfig(err.clone()));
        }
        Ok(guard.queue.len())
    }

    pub fn trigger_refill(&self) {
        let mut guard = self.sync.state.lock();
        if !guard.dataset_eof && guard.queue.len() < self.config.capacity_batches {
            guard.refilling = true;
            guard.worker_idle = false;
            self.sync.worker_cv.notify_one();
        }
    }

    pub fn next_batch(&self) -> Result<Option<FetchedBatch>> {
        let mut guard = self.sync.state.lock();
        loop {
            if let Some(err) = &guard.worker_error {
                return Err(DataLoaderError::InvalidConfig(err.clone()));
            }

            if let Some(batch) = guard.queue.pop_front() {
                let low_watermark = self
                    .config
                    .capacity_batches
                    .saturating_sub(self.config.refill_trigger);

                if self.config.auto_refill
                    && !guard.dataset_eof
                    && (guard.queue.len() <= low_watermark || guard.queue.is_empty())
                {
                    guard.refilling = true;
                    guard.worker_idle = false;
                    self.sync.worker_cv.notify_one();
                }
                return Ok(Some(batch));
            }

            if guard.dataset_eof {
                return Ok(None);
            }

            guard.refilling = true;
            guard.worker_idle = false;
            self.sync.worker_cv.notify_one();
            self.sync.client_cv.wait(&mut guard);
        }
    }

    pub fn commit_gpu_batch(&self, batch_id: u64) -> Result<StreamCoordinate> {
        self.sync.meta.lock().commit_gpu_batch(batch_id)
    }

    pub fn rollback_staged_batch(&self, batch_id: u64) -> Result<StreamCoordinate> {
        self.sync.meta.lock().rollback_staged_batch(batch_id)
    }

    pub fn buffered_batches(&self) -> usize {
        self.sync.state.lock().queue.len()
    }

    pub fn committed_coordinate(&self) -> StreamCoordinate {
        self.sync.meta.lock().committed_coordinate()
    }

    pub fn in_flight_coordinate(&self) -> StreamCoordinate {
        self.sync.meta.lock().in_flight_coordinate()
    }

    pub fn committed_batch_id(&self) -> u64 {
        self.sync.meta.lock().committed_batch_id()
    }

    pub fn download_workers(&self) -> usize {
        self.config.download_workers
    }
}

impl Drop for PreRamDataLoader {
    fn drop(&mut self) {
        {
            let mut guard = self.sync.state.lock();
            guard.shutdown = true;
            self.sync.worker_cv.notify_all();
        }
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }
}
