use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crc32fast::Hasher;
use hashbrown::HashMap;

use crate::constants::{
    DEDUP_FILE_SUFFIX, META_SLOT_MAGIC, META_SLOT_SIZE, META_SLOT_VERSION, TOTAL_META_SIZE,
};
use crate::coord::StreamCoordinate;
use crate::dedup::TwoStageDeduplicator;
use crate::error::{DataLoaderError, Result};

const SLOT_MAGIC: [u8; 4] = META_SLOT_MAGIC;
const SLOT_VERSION: u32 = META_SLOT_VERSION;
const SLOT_SIZE: usize = META_SLOT_SIZE;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct DiskSlot {
    epoch: u64,
    committed_batch_id: u64,
    coord: StreamCoordinate,
    dedup_count: u64,
}

impl DiskSlot {
    fn encode(&self) -> [u8; SLOT_SIZE] {
        let mut buf = [0u8; SLOT_SIZE];
        buf[0..4].copy_from_slice(&SLOT_MAGIC);
        buf[4..8].copy_from_slice(&SLOT_VERSION.to_le_bytes());
        buf[8..16].copy_from_slice(&self.epoch.to_le_bytes());
        buf[16..24].copy_from_slice(&self.committed_batch_id.to_le_bytes());
        buf[24..56].copy_from_slice(&self.coord.to_bytes());
        buf[56..60].copy_from_slice(&(self.dedup_count as u32).to_le_bytes());

        let mut hasher = Hasher::new();
        hasher.update(&buf[0..60]);
        let crc = hasher.finalize();
        buf[60..64].copy_from_slice(&crc.to_le_bytes());
        buf
    }

    fn decode(buf: &[u8; SLOT_SIZE]) -> Option<Self> {
        if buf[0..4] != SLOT_MAGIC {
            return None;
        }
        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        if version != SLOT_VERSION {
            return None;
        }
        let expected_crc = u32::from_le_bytes(buf[60..64].try_into().unwrap());
        let mut hasher = Hasher::new();
        hasher.update(&buf[0..60]);
        if hasher.finalize() != expected_crc {
            return None;
        }
        let coord_bytes: &[u8; StreamCoordinate::BYTE_LEN] = buf[24..56].try_into().unwrap();
        Some(Self {
            epoch: u64::from_le_bytes(buf[8..16].try_into().unwrap()),
            committed_batch_id: u64::from_le_bytes(buf[16..24].try_into().unwrap()),
            coord: StreamCoordinate::from_bytes(coord_bytes),
            dedup_count: u32::from_le_bytes(buf[56..60].try_into().unwrap()) as u64,
        })
    }
}

pub struct DualMetadataManager {
    meta_file: File,
    dedup_file: File,
    epoch: u64,
    committed_batch_id: u64,
    committed_coord: StreamCoordinate,
    in_flight_coord: StreamCoordinate,
    staged_coords: HashMap<u64, StreamCoordinate>,
    pub deduplicator: TwoStageDeduplicator,
}

impl DualMetadataManager {
    pub fn open(meta_path: impl AsRef<Path>) -> Result<Self> {
        let meta_path = meta_path.as_ref();
        if let Some(parent) = meta_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }

        let dedup_path = Self::dedup_path_for(meta_path);

        let mut meta_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(meta_path)?;

        let mut dedup_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&dedup_path)?;

        let file_len = meta_file.metadata()?.len();
        let active_slot = if file_len < TOTAL_META_SIZE {
            meta_file.set_len(TOTAL_META_SIZE)?;
            let initial = DiskSlot::default();
            let encoded = initial.encode();
            meta_file.seek(SeekFrom::Start(0))?;
            meta_file.write_all(&encoded)?;
            meta_file.write_all(&encoded)?;
            meta_file.sync_data()?;
            initial
        } else {
            let mut raw = [0u8; SLOT_SIZE * 2];
            meta_file.seek(SeekFrom::Start(0))?;
            meta_file.read_exact(&mut raw)?;

            let slot_a = DiskSlot::decode(raw[0..SLOT_SIZE].try_into().unwrap());
            let slot_b = DiskSlot::decode(raw[SLOT_SIZE..SLOT_SIZE * 2].try_into().unwrap());

            match (slot_a, slot_b) {
                (Some(a), Some(b)) => {
                    if a.epoch >= b.epoch {
                        a
                    } else {
                        b
                    }
                }
                (Some(a), None) => a,
                (None, Some(b)) => b,
                (None, None) => return Err(DataLoaderError::CorruptedMetadata),
            }
        };

        let valid_dedup_bytes = active_slot.dedup_count * 8;
        if dedup_file.metadata()?.len() < valid_dedup_bytes {
            return Err(DataLoaderError::CorruptedMetadata);
        }
        dedup_file.set_len(valid_dedup_bytes)?;
        dedup_file.seek(SeekFrom::Start(0))?;

        let mut raw_hashes = vec![0u8; valid_dedup_bytes as usize];
        if valid_dedup_bytes > 0 {
            dedup_file.read_exact(&mut raw_hashes)?;
        }

        let restored_hashes = raw_hashes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&c| u64::from_le_bytes(c));

        let deduplicator = TwoStageDeduplicator::with_committed(restored_hashes);

        Ok(Self {
            meta_file,
            dedup_file,
            epoch: active_slot.epoch,
            committed_batch_id: active_slot.committed_batch_id,
            committed_coord: active_slot.coord,
            in_flight_coord: active_slot.coord,
            staged_coords: HashMap::new(),
            deduplicator,
        })
    }

    fn dedup_path_for(meta_path: &Path) -> PathBuf {
        let mut p = meta_path.as_os_str().to_owned();
        p.push(DEDUP_FILE_SUFFIX);
        PathBuf::from(p)
    }

    #[inline]
    pub fn committed_coordinate(&self) -> StreamCoordinate {
        self.committed_coord
    }

    #[inline]
    pub fn in_flight_coordinate(&self) -> StreamCoordinate {
        self.in_flight_coord
    }

    #[inline]
    pub fn committed_batch_id(&self) -> u64 {
        self.committed_batch_id
    }

    #[inline]
    pub fn set_in_flight_coordinate(&mut self, next_coord: StreamCoordinate) {
        self.in_flight_coord = next_coord;
    }

    pub fn stage_batch_in_ram(&mut self, batch_id: u64, next_coord: StreamCoordinate) {
        self.in_flight_coord = next_coord;
        self.staged_coords.insert(batch_id, next_coord);
    }

    pub fn commit_gpu_batch(&mut self, batch_id: u64) -> Result<StreamCoordinate> {
        let Some(next_coord) = self.staged_coords.remove(&batch_id) else {
            return Err(DataLoaderError::UnknownBatchId(batch_id));
        };

        let prev_dedup_count = self.deduplicator.committed_len() as u64;
        let new_hashes = self.deduplicator.commit_batch(batch_id);
        if !new_hashes.is_empty() {
            let mut hash_buf = Vec::with_capacity(new_hashes.len() * 8);
            for h in new_hashes {
                hash_buf.extend_from_slice(&h.to_le_bytes());
            }
            let write_offset = prev_dedup_count * 8;
            self.dedup_file.seek(SeekFrom::Start(write_offset))?;
            self.dedup_file.write_all(&hash_buf)?;
            self.dedup_file
                .set_len(write_offset + hash_buf.len() as u64)?;
            self.dedup_file.sync_data()?;
        }

        self.epoch += 1;
        self.committed_batch_id = batch_id;
        self.committed_coord = next_coord;

        let slot = DiskSlot {
            epoch: self.epoch,
            committed_batch_id: self.committed_batch_id,
            coord: self.committed_coord,
            dedup_count: self.deduplicator.committed_len() as u64,
        };

        let slot_offset = if self.epoch % 2 == 1 {
            0
        } else {
            SLOT_SIZE as u64
        };

        let encoded = slot.encode();
        self.meta_file.seek(SeekFrom::Start(slot_offset))?;
        self.meta_file.write_all(&encoded)?;
        self.meta_file.sync_data()?;

        Ok(self.committed_coord)
    }

    pub fn rollback_staged_batch(&mut self, batch_id: u64) -> Result<StreamCoordinate> {
        if self.staged_coords.remove(&batch_id).is_none() {
            return Err(DataLoaderError::UnknownBatchId(batch_id));
        }
        self.deduplicator.rollback_batch(batch_id);
        self.in_flight_coord = self
            .staged_coords
            .iter()
            .max_by_key(|&(&id, _)| id)
            .map(|(_, &coord)| coord)
            .unwrap_or(self.committed_coord);
        Ok(self.in_flight_coord)
    }

    pub fn rollback_ram_to_committed(&mut self) -> StreamCoordinate {
        self.staged_coords.clear();
        self.deduplicator.clear_staged();
        self.in_flight_coord = self.committed_coord;
        self.committed_coord
    }
}
