use std::hash::{BuildHasherDefault, Hasher};

use hashbrown::{HashMap, HashSet};
use xxhash_rust::xxh3::xxh3_64;

#[derive(Default)]
pub struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.0 = xxh3_64(bytes);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }
}

pub type IdentityBuildHasher = BuildHasherDefault<IdentityHasher>;

#[derive(Debug, Default)]
pub struct TwoStageDeduplicator {
    committed_hashes: HashSet<u64, IdentityBuildHasher>,
    staged_lookup: HashSet<u64, IdentityBuildHasher>,
    staged_batches: HashMap<u64, Vec<u64>>,
}

impl TwoStageDeduplicator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_committed(hashes: impl IntoIterator<Item = u64>) -> Self {
        let committed_hashes: HashSet<u64, IdentityBuildHasher> = hashes.into_iter().collect();
        Self {
            committed_hashes,
            staged_lookup: HashSet::default(),
            staged_batches: HashMap::new(),
        }
    }

    #[inline]
    pub fn compute_hash(payload: &[u8]) -> u64 {
        xxh3_64(payload)
    }

    #[inline]
    pub fn try_stage(&mut self, batch_id: u64, payload: &[u8]) -> Option<u64> {
        let hash = Self::compute_hash(payload);
        if self.committed_hashes.contains(&hash) || self.staged_lookup.contains(&hash) {
            return None;
        }
        self.staged_lookup.insert(hash);
        self.staged_batches.entry(batch_id).or_default().push(hash);
        Some(hash)
    }

    pub fn commit_batch(&mut self, batch_id: u64) -> Vec<u64> {
        let Some(hashes) = self.staged_batches.remove(&batch_id) else {
            return Vec::new();
        };
        for &h in &hashes {
            self.staged_lookup.remove(&h);
            self.committed_hashes.insert(h);
        }
        hashes
    }

    pub fn rollback_batch(&mut self, batch_id: u64) -> usize {
        let Some(hashes) = self.staged_batches.remove(&batch_id) else {
            return 0;
        };
        for &h in &hashes {
            self.staged_lookup.remove(&h);
        }
        hashes.len()
    }

    pub fn clear_staged(&mut self) {
        self.staged_lookup.clear();
        self.staged_batches.clear();
    }

    pub fn committed_len(&self) -> usize {
        self.committed_hashes.len()
    }

    pub fn staged_len(&self) -> usize {
        self.staged_lookup.len()
    }
}
