use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::Path;

use crate::constants::{DEDUP_HASH_BYTE_LEN, WAL_HEADER_SIZE, WAL_MAGIC, WAL_VERSION};

pub(super) enum WalRead {
    Missing,
    Corrupted(String),
    Loaded(Vec<u128>),
}

pub(super) fn make_wal_header() -> [u8; WAL_HEADER_SIZE] {
    let mut hdr = [0u8; WAL_HEADER_SIZE];
    hdr[0..4].copy_from_slice(&WAL_MAGIC);
    hdr[4..8].copy_from_slice(&WAL_VERSION.to_le_bytes());
    hdr
}

pub(super) fn append_wal_hashes(target: &Path, hashes: &[u128]) -> std::io::Result<()> {
    if hashes.is_empty() {
        return Ok(());
    }
    let mut buf = Vec::with_capacity(hashes.len() * DEDUP_HASH_BYTE_LEN);
    for &h in hashes {
        buf.extend_from_slice(&h.to_le_bytes());
    }
    let mut file = OpenOptions::new().append(true).open(target)?;
    file.write_all(&buf)?;
    file.sync_data()
}

pub(super) fn write_full_wal(target: &Path, hashes: &[u128]) -> std::io::Result<()> {
    let mut buf = Vec::with_capacity(WAL_HEADER_SIZE + hashes.len() * DEDUP_HASH_BYTE_LEN);
    buf.extend_from_slice(&make_wal_header());
    for &h in hashes {
        buf.extend_from_slice(&h.to_le_bytes());
    }
    let tmp_path = target.with_extension("dedup.tmp");
    {
        let mut file = File::create(&tmp_path)?;
        file.write_all(&buf)?;
        file.sync_data()?;
    }
    std::fs::rename(&tmp_path, target)
}

pub(super) fn read_wal_file(target: &Path, display_path: &str) -> std::io::Result<WalRead> {
    let mut file = match File::open(target) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(WalRead::Missing),
        Err(e) => return Err(e),
    };
    let len = file.metadata()?.len() as usize;
    if len < WAL_HEADER_SIZE {
        return Ok(WalRead::Corrupted(format!(
            "Corrupted WAL file '{display_path}': length {len} < header size {WAL_HEADER_SIZE}"
        )));
    }
    let mut hdr = [0u8; WAL_HEADER_SIZE];
    file.read_exact(&mut hdr)?;
    if &hdr[0..4] == b"LMWP" {
        return Ok(WalRead::Corrupted(format!(
            "WAL '{display_path}' uses Python 'LMWP', incompatible with Rust 'LMWL'"
        )));
    }
    if hdr[0..4] != WAL_MAGIC || u32::from_le_bytes(hdr[4..8].try_into().unwrap()) != WAL_VERSION {
        return Ok(WalRead::Corrupted(format!(
            "Corrupted WAL '{display_path}': invalid LMWL v{WAL_VERSION} header"
        )));
    }
    let payload_len = len - WAL_HEADER_SIZE;
    let valid_payload_len = (payload_len / DEDUP_HASH_BYTE_LEN) * DEDUP_HASH_BYTE_LEN;
    if valid_payload_len < payload_len {
        // Recover from a torn append by truncating incomplete trailing bytes.
        let valid_total_len = (WAL_HEADER_SIZE + valid_payload_len) as u64;
        if let Ok(trunc_file) = OpenOptions::new().write(true).open(target) {
            let _ = trunc_file.set_len(valid_total_len);
            let _ = trunc_file.sync_data();
        }
    }
    let count = valid_payload_len / DEDUP_HASH_BYTE_LEN;
    let mut hashes = Vec::with_capacity(count.min(1_000_000));
    let mut reader = BufReader::with_capacity(64 * 1024, &mut file);
    let mut chunk = [0u8; 64 * 1024];
    let mut rem = valid_payload_len;
    while rem > 0 {
        let n = rem.min(chunk.len());
        reader.read_exact(&mut chunk[..n])?;
        for c in chunk[..n].as_chunks::<DEDUP_HASH_BYTE_LEN>().0 {
            hashes.push(u128::from_le_bytes(*c));
        }
        rem -= n;
    }
    Ok(WalRead::Loaded(hashes))
}

pub(super) fn truncate_wal_file(target: &Path, max_count: usize) -> std::io::Result<usize> {
    let file = match OpenOptions::new().read(true).write(true).open(target) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e),
    };
    let len = file.metadata()?.len() as usize;
    if len < WAL_HEADER_SIZE {
        return Ok(0);
    }
    let current_count = (len - WAL_HEADER_SIZE) / DEDUP_HASH_BYTE_LEN;
    let keep_count = current_count.min(max_count);
    let target_len = WAL_HEADER_SIZE + keep_count * DEDUP_HASH_BYTE_LEN;
    if target_len < len {
        file.set_len(target_len as u64)?;
        file.sync_data()?;
    }
    Ok(keep_count)
}
