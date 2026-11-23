use std::collections::BTreeMap;
use std::hash::{BuildHasherDefault, Hasher};

use hashbrown::HashSet;
use xxhash_rust::xxh3::{xxh3_64, xxh3_128};

#[derive(Default)]
pub struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let h = xxh3_64(bytes);
        self.0 = self.0.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ h;
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = self.0.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ i;
    }

    #[inline]
    fn write_u128(&mut self, i: u128) {
        let lo = i as u64;
        let hi = (i >> 64) as u64;
        self.0 = self.0.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ lo ^ hi.rotate_left(31);
    }
}

pub type IdentityBuildHasher = BuildHasherDefault<IdentityHasher>;

#[derive(Debug, Default)]
pub struct TwoStageDeduplicator {
    committed_hashes: HashSet<u128, IdentityBuildHasher>,
    staged_lookup: HashSet<u128, IdentityBuildHasher>,
    staged_batches: BTreeMap<u64, Vec<u128>>,
    max_capacity: Option<usize>,
}

impl TwoStageDeduplicator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(max_capacity: Option<usize>) -> Self {
        Self {
            committed_hashes: HashSet::default(),
            staged_lookup: HashSet::default(),
            staged_batches: BTreeMap::new(),
            max_capacity,
        }
    }

    pub fn with_committed(hashes: impl IntoIterator<Item = u128>) -> Self {
        let committed_hashes: HashSet<u128, IdentityBuildHasher> = hashes.into_iter().collect();
        Self {
            committed_hashes,
            staged_lookup: HashSet::default(),
            staged_batches: BTreeMap::new(),
            max_capacity: None,
        }
    }

    #[inline]
    pub fn compute_hash_64(payload: &[u8]) -> u64 {
        xxh3_64(payload)
    }

    #[inline]
    pub fn compute_hash(payload: &[u8]) -> u128 {
        xxh3_128(payload)
    }

    #[inline]
    pub fn is_seen(&self, payload: &[u8]) -> bool {
        let hash = Self::compute_hash(payload);
        self.committed_hashes.contains(&hash) || self.staged_lookup.contains(&hash)
    }

    #[inline]
    pub fn is_hash_seen(&self, hash: u128) -> bool {
        self.committed_hashes.contains(&hash) || self.staged_lookup.contains(&hash)
    }

    #[inline]
    pub fn try_stage_hash(&mut self, batch_id: u64, hash: u128) -> Option<u128> {
        if self.committed_hashes.contains(&hash) || self.staged_lookup.contains(&hash) {
            return None;
        }
        if let Some(cap) = self.max_capacity
            && self.committed_hashes.len() + self.staged_lookup.len() >= cap
        {
            return Some(hash);
        }
        self.staged_lookup.insert(hash);
        self.staged_batches.entry(batch_id).or_default().push(hash);
        Some(hash)
    }

    #[inline]
    pub fn try_stage(&mut self, batch_id: u64, payload: &[u8]) -> Option<u128> {
        let hash = Self::compute_hash(payload);
        self.try_stage_hash(batch_id, hash)
    }

    /// Preview staged hashes for a single batch without mutating state (for atomic commit).
    pub fn peek_batch(&self, batch_id: u64) -> Vec<u128> {
        self.staged_batches
            .get(&batch_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Preview staged hashes for all batches `<= batch_id` in ascending batch order.
    pub fn peek_up_to(&self, batch_id: u64) -> Vec<u128> {
        let mut out = Vec::new();
        for (&bid, hashes) in self.staged_batches.range(..=batch_id) {
            let _ = bid;
            out.extend_from_slice(hashes);
        }
        out
    }

    pub fn commit_batch(&mut self, batch_id: u64) -> Vec<u128> {
        let Some(hashes) = self.staged_batches.remove(&batch_id) else {
            return Vec::new();
        };
        let mut newly_committed = Vec::with_capacity(hashes.len());
        for h in hashes {
            self.staged_lookup.remove(&h);
            if self.committed_hashes.insert(h) {
                newly_committed.push(h);
            }
        }
        newly_committed
    }

    /// Commits all staged batches with ID `<= batch_id` in ascending order.
    pub fn commit_up_to(&mut self, batch_id: u64) -> Vec<u128> {
        let keys: Vec<u64> = self.staged_batches.range(..=batch_id).map(|(&k, _)| k).collect();
        let mut newly_committed = Vec::new();
        for k in keys {
            newly_committed.extend(self.commit_batch(k));
        }
        newly_committed
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

    pub fn clear_all(&mut self) {
        self.committed_hashes.clear();
        self.staged_lookup.clear();
        self.staged_batches.clear();
    }

    pub fn committed_len(&self) -> usize {
        self.committed_hashes.len()
    }

    pub fn staged_len(&self) -> usize {
        self.staged_lookup.len()
    }

    pub fn export_committed_hashes(&self) -> Vec<u128> {
        self.committed_hashes.iter().copied().collect()
    }

    pub fn import_committed_hashes(&mut self, hashes: impl IntoIterator<Item = u128>) {
        let mut imported_any = false;
        for h in hashes {
            self.committed_hashes.insert(h);
            if self.staged_lookup.remove(&h) {
                imported_any = true;
            }
        }
        if imported_any {
            for batch in self.staged_batches.values_mut() {
                batch.retain(|h| !self.committed_hashes.contains(h));
            }
            self.staged_batches.retain(|_, v| !v.is_empty());
        }
    }
}
