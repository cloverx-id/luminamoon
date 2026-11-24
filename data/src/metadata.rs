use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crc32fast::Hasher;

use crate::constants::{
    DEDUP_FILE_SUFFIX, DEDUP_HASH_BYTE_LEN, META_SLOT_MAGIC, META_SLOT_SIZE, META_SLOT_VERSION,
    TOTAL_META_SIZE, WAL_HEADER_SIZE, WAL_MAGIC, WAL_VERSION,
};
use crate::coord::StreamCoordinate;
use crate::dedup::TwoStageDeduplicator;
use crate::error::{DataLoaderError, Result};

const SLOT_MAGIC: [u8; 4] = META_SLOT_MAGIC;
const SLOT_VERSION: u32 = META_SLOT_VERSION;
const SLOT_SIZE: usize = META_SLOT_SIZE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DiskSlot {
    epoch: u64,
    committed_batch_id: u64,
    coord: StreamCoordinate,
    dedup_count: u64,
    consumed_tokens: u64,
    hash_width_bytes: u32,
}

impl Default for DiskSlot {
    fn default() -> Self {
        Self {
            epoch: 0,
            committed_batch_id: 0,
            coord: StreamCoordinate::default(),
            dedup_count: 0,
            consumed_tokens: 0,
            hash_width_bytes: DEDUP_HASH_BYTE_LEN as u32,
        }
    }
}

impl DiskSlot {
    fn encode(&self) -> [u8; SLOT_SIZE] {
        let mut buf = [0u8; SLOT_SIZE];
        buf[0..4].copy_from_slice(&SLOT_MAGIC);
        buf[4..8].copy_from_slice(&SLOT_VERSION.to_le_bytes());
        buf[8..16].copy_from_slice(&self.epoch.to_le_bytes());
        buf[16..24].copy_from_slice(&self.committed_batch_id.to_le_bytes());
        buf[24..56].copy_from_slice(&self.coord.to_bytes());
        buf[56..64].copy_from_slice(&self.dedup_count.to_le_bytes());
        buf[64..72].copy_from_slice(&self.consumed_tokens.to_le_bytes());
        buf[72..76].copy_from_slice(&self.hash_width_bytes.to_le_bytes());

        let mut hasher = Hasher::new();
        hasher.update(&buf[0..124]);
        let crc = hasher.finalize();
        buf[124..128].copy_from_slice(&crc.to_le_bytes());
        buf
    }

    fn decode(buf: &[u8; SLOT_SIZE]) -> Option<Self> {
        if buf[0..4] != SLOT_MAGIC {
            return None;
        }
        let version = u32::from_le_bytes(buf[4..8].try_into().ok()?);
        if version != SLOT_VERSION {
            return None;
        }
        let expected_crc = u32::from_le_bytes(buf[124..128].try_into().ok()?);
        let mut hasher = Hasher::new();
        hasher.update(&buf[0..124]);
        if hasher.finalize() != expected_crc {
            return None;
        }
        let coord_bytes: &[u8; StreamCoordinate::BYTE_LEN] = buf[24..56].try_into().ok()?;
        let hash_width_bytes = u32::from_le_bytes(buf[72..76].try_into().ok()?);
        if hash_width_bytes != DEDUP_HASH_BYTE_LEN as u32 {
            return None;
        }
        Some(Self {
            epoch: u64::from_le_bytes(buf[8..16].try_into().ok()?),
            committed_batch_id: u64::from_le_bytes(buf[16..24].try_into().ok()?),
            coord: StreamCoordinate::from_bytes(coord_bytes),
            dedup_count: u64::from_le_bytes(buf[56..64].try_into().ok()?),
            consumed_tokens: u64::from_le_bytes(buf[64..72].try_into().ok()?),
            hash_width_bytes,
        })
    }
}

fn wal_header_bytes() -> [u8; WAL_HEADER_SIZE] {
    let mut hdr = [0u8; WAL_HEADER_SIZE];
    hdr[0..4].copy_from_slice(&WAL_MAGIC);
    hdr[4..8].copy_from_slice(&WAL_VERSION.to_le_bytes());
    hdr
}

fn ensure_wal_header(dedup_file: &mut File) -> Result<u64> {
    let len = dedup_file.metadata()?.len();
    if len == 0 {
        dedup_file.seek(SeekFrom::Start(0))?;
        dedup_file.write_all(&wal_header_bytes())?;
        dedup_file.sync_data()?;
        return Ok(WAL_HEADER_SIZE as u64);
    }
    if len < WAL_HEADER_SIZE as u64 {
        return Err(DataLoaderError::CorruptedMetadata);
    }
    let mut hdr = [0u8; WAL_HEADER_SIZE];
    dedup_file.seek(SeekFrom::Start(0))?;
    dedup_file.read_exact(&mut hdr)?;
    if hdr[0..4] != WAL_MAGIC || u32::from_le_bytes(hdr[4..8].try_into().unwrap()) != WAL_VERSION {
        return Err(DataLoaderError::CorruptedMetadata);
    }
    Ok(len)
}

pub struct DualMetadataManager {
    meta_file: File,
    dedup_file: File,
    epoch: u64,
    committed_batch_id: u64,
    committed_coord: StreamCoordinate,
    consumed_tokens: u64,
    in_flight_coord: StreamCoordinate,
    staged_coords: BTreeMap<u64, StreamCoordinate>,
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
        let wal_len = ensure_wal_header(&mut dedup_file)?;

        let candidates: Vec<DiskSlot> = if file_len == 0 {
            meta_file.set_len(TOTAL_META_SIZE)?;
            let initial = DiskSlot::default();
            let encoded = initial.encode();
            meta_file.seek(SeekFrom::Start(0))?;
            meta_file.write_all(&encoded)?;
            meta_file.write_all(&encoded)?;
            meta_file.sync_data()?;
            vec![initial]
        } else if file_len < TOTAL_META_SIZE {
            // Never silently wipe a partially truncated metadata file.
            if file_len >= SLOT_SIZE as u64 {
                let mut raw = [0u8; SLOT_SIZE];
                meta_file.seek(SeekFrom::Start(0))?;
                meta_file.read_exact(&mut raw)?;
                match DiskSlot::decode(&raw) {
                    Some(slot_a) => vec![slot_a],
                    None => return Err(DataLoaderError::CorruptedMetadata),
                }
            } else {
                return Err(DataLoaderError::CorruptedMetadata);
            }
        } else {
            let mut raw = [0u8; SLOT_SIZE * 2];
            meta_file.seek(SeekFrom::Start(0))?;
            meta_file.read_exact(&mut raw)?;

            let slot_a = DiskSlot::decode(raw[0..SLOT_SIZE].try_into().unwrap());
            let slot_b = DiskSlot::decode(raw[SLOT_SIZE..SLOT_SIZE * 2].try_into().unwrap());

            let mut list = Vec::with_capacity(2);
            if let Some(a) = slot_a {
                list.push(a);
            }
            if let Some(b) = slot_b {
                list.push(b);
            }
            list.sort_by_key(|x| std::cmp::Reverse(x.epoch));
            if list.is_empty() {
                return Err(DataLoaderError::CorruptedMetadata);
            }
            list
        };

        // Pick highest-epoch valid slot whose WAL payload is intact (falls back to older slot on crash).
        let active_slot = candidates
            .into_iter()
            .find(|slot| {
                slot.dedup_count
                    .checked_mul(DEDUP_HASH_BYTE_LEN as u64)
                    .and_then(|n| n.checked_add(WAL_HEADER_SIZE as u64))
                    .is_some_and(|needed| wal_len >= needed)
            })
            .ok_or(DataLoaderError::CorruptedMetadata)?;

        let valid_wal_bytes =
            (WAL_HEADER_SIZE as u64) + active_slot.dedup_count * (DEDUP_HASH_BYTE_LEN as u64);
        if wal_len > valid_wal_bytes {
            dedup_file.set_len(valid_wal_bytes)?;
            dedup_file.sync_data()?;
        }

        // Stream WAL in 64 KiB chunks to avoid 2x RAM spikes.
        dedup_file.seek(SeekFrom::Start(WAL_HEADER_SIZE as u64))?;
        let mut reader = BufReader::with_capacity(64 * 1024, &mut dedup_file);
        let mut chunk = [0u8; 64 * 1024];
        let mut remaining = (active_slot.dedup_count * (DEDUP_HASH_BYTE_LEN as u64)) as usize;
        let mut restored = Vec::with_capacity(active_slot.dedup_count.min(1_000_000) as usize);

        while remaining > 0 {
            let to_read = remaining.min(chunk.len());
            reader.read_exact(&mut chunk[..to_read])?;
            for entry in chunk[..to_read].as_chunks::<DEDUP_HASH_BYTE_LEN>().0 {
                restored.push(u128::from_le_bytes(*entry));
            }
            remaining -= to_read;
        }
        drop(reader);

        let deduplicator = TwoStageDeduplicator::with_committed(restored);

        Ok(Self {
            meta_file,
            dedup_file,
            epoch: active_slot.epoch,
            committed_batch_id: active_slot.committed_batch_id,
            committed_coord: active_slot.coord,
            consumed_tokens: active_slot.consumed_tokens,
            in_flight_coord: active_slot.coord,
            staged_coords: BTreeMap::new(),
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
    pub fn consumed_tokens(&self) -> u64 {
        self.consumed_tokens
    }

    #[inline]
    pub fn set_consumed_tokens(&mut self, tokens: u64) {
        self.consumed_tokens = tokens;
    }

    #[inline]
    pub fn set_in_flight_coordinate(&mut self, next_coord: StreamCoordinate) {
        self.in_flight_coord = next_coord;
    }

    pub fn stage_batch_in_ram(&mut self, batch_id: u64, next_coord: StreamCoordinate) {
        self.in_flight_coord = next_coord;
        self.staged_coords.insert(batch_id, next_coord);
    }

    fn persist_commit(
        &mut self,
        next_batch_id: u64,
        next_coord: StreamCoordinate,
        pending_hashes: &[u128],
    ) -> Result<()> {
        let prev_dedup_count = self.deduplicator.committed_len() as u64;
        let next_epoch = self.epoch.checked_add(1).ok_or_else(|| {
            DataLoaderError::InvalidConfig("Checkpoint epoch overflow".to_owned())
        })?;

        // Write-before-mutate: flush WAL + metadata slot to disk before mutating RAM state.
        if !pending_hashes.is_empty() {
            let mut hash_buf = Vec::with_capacity(pending_hashes.len() * DEDUP_HASH_BYTE_LEN);
            for &h in pending_hashes {
                hash_buf.extend_from_slice(&h.to_le_bytes());
            }
            let write_offset =
                (WAL_HEADER_SIZE as u64) + prev_dedup_count * (DEDUP_HASH_BYTE_LEN as u64);
            self.dedup_file.seek(SeekFrom::Start(write_offset))?;
            self.dedup_file.write_all(&hash_buf)?;
            self.dedup_file
                .set_len(write_offset + hash_buf.len() as u64)?;
            self.dedup_file.sync_data()?;
        }

        let new_dedup_count = prev_dedup_count + pending_hashes.len() as u64;
        let effective_batch_id = next_batch_id.max(self.committed_batch_id);
        let effective_coord = if next_batch_id >= self.committed_batch_id {
            next_coord
        } else {
            self.committed_coord
        };

        let slot = DiskSlot {
            epoch: next_epoch,
            committed_batch_id: effective_batch_id,
            coord: effective_coord,
            dedup_count: new_dedup_count,
            consumed_tokens: self.consumed_tokens,
            hash_width_bytes: DEDUP_HASH_BYTE_LEN as u32,
        };

        let slot_offset = if next_epoch % 2 == 1 {
            0
        } else {
            SLOT_SIZE as u64
        };
        let encoded = slot.encode();
        self.meta_file.seek(SeekFrom::Start(slot_offset))?;
        self.meta_file.write_all(&encoded)?;
        self.meta_file.sync_data()?;

        self.epoch = next_epoch;
        self.committed_batch_id = effective_batch_id;
        self.committed_coord = effective_coord;
        Ok(())
    }

    pub fn commit_gpu_batch(&mut self, batch_id: u64) -> Result<StreamCoordinate> {
        let Some(&next_coord) = self.staged_coords.get(&batch_id) else {
            return Err(DataLoaderError::UnknownBatchId(batch_id));
        };
        let pending_hashes = self.deduplicator.peek_batch(batch_id);
        self.persist_commit(batch_id, next_coord, &pending_hashes)?;
        self.staged_coords.remove(&batch_id);
        self.deduplicator.commit_batch(batch_id);
        Ok(self.committed_coord)
    }

    /// Commits all staged batches with ID `<= batch_id` atomically and idempotently.
    pub fn commit_up_to(&mut self, batch_id: u64) -> Result<StreamCoordinate> {
        let keys: Vec<u64> = self
            .staged_coords
            .range(..=batch_id)
            .map(|(&k, _)| k)
            .collect();
        if keys.is_empty() {
            if batch_id <= self.committed_batch_id {
                return Ok(self.committed_coord);
            }
            return Err(DataLoaderError::UnknownBatchId(batch_id));
        }
        let latest_bid = *keys.last().unwrap();
        let latest_coord = self.staged_coords[&latest_bid];
        let pending_hashes = self.deduplicator.peek_up_to(latest_bid);
        self.persist_commit(latest_bid, latest_coord, &pending_hashes)?;
        for k in keys {
            self.staged_coords.remove(&k);
        }
        self.deduplicator.commit_up_to(latest_bid);
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
            .next_back()
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
