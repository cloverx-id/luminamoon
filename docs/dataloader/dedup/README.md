# Two-Stage XXH3-64 Deduplication & Identity SwissTable

[`data/src/dedup.rs`](https://github.com/cloverx-id/luminamoon/blob/main/data/src/dedup.rs) provides a two-stage streaming deduplicator that separates prefetched records from records committed by training steps.

---

## 1. Two-Stage State Machine (`TwoStageDeduplicator`)

In asynchronous pipelines, multiple batches can be staged in memory ahead of the active GPU step. If a single set marked hashes as seen upon prefetch and the process stopped before training consumed them, resuming would either skip unread samples or lose deduplication consistency.

`TwoStageDeduplicator` avoids this by partitioning tracked hashes into two disjoint sets:

$$\mathcal{H}_{\text{committed}} \cap \mathcal{H}_{\text{staged}} = \emptyset$$

```mermaid
flowchart LR
    Record["Candidate Record"] --> Hash["Compute XXH3-64 (IdentityHasher)"]
    Hash --> Drop["Duplicate Digest: Drop Record"]
    Hash --> Stage["Unique Digest: Insert into staged_lookup"]
    Stage --> Commit["commit_batch: Promote to committed_hashes & WAL"]
    Stage --> Evict["rollback_batch: Evict from staged_lookup"]
```

### Core Operations

1. **`try_stage(batch_id, payload) -> Option<u64>`**:
   - Computes $h = \text{XXH3}_{64}(\text{payload})$.
   - Rejects the record (`None`) if $h \in \text{committed}_{\text{hashes}} \cup \text{staged}_{\text{lookup}}$.
   - Otherwise inserts $h$ into `staged_lookup` and appends $h$ to `staged_batches[batch_id]`.
2. **`commit_batch(batch_id) -> Vec<u64>`**:
   - Atomically promotes all hashes in `staged_batches[batch_id]` from `staged_lookup` into `committed_hashes` and returns the vector of promoted hashes to be flushed to the `.meta.dedup` write-ahead log.
3. **`rollback_batch(batch_id)` & `clear_staged()`**:
   - Evicts uncommitted batch hashes from `staged_lookup` without touching `committed_hashes`, restoring exact state parity with the last committed GPU step.

---

## 2. Zero-Rehash SwissTable (`IdentityBuildHasher`)

Because keys stored in `committed_hashes` and `staged_lookup` are already uniformly distributed 64-bit `xxh3_64` digests, hashing them again with `hashbrown`'s default `foldhash` would waste CPU cycles.

`dedup.rs` defines an `IdentityHasher` that passes the digest directly to the table probe:

```rust
#[derive(Default)]
pub struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }
}

pub type IdentityBuildHasher = BuildHasherDefault<IdentityHasher>;
```

---

## 3. MinHash LSH Near-Duplicate Shingling (`src/minhash.rs`)

For fuzzy near-duplicate rejection alongside exact 64-bit `xxh3_64` deduplication, `MinHashDeduplicator` computes $K$-character UTF-8 shingles hashed across $P$ deterministic permuted `xxh3_64_with_seed` lanes and estimates pairwise Jaccard similarity against indexed document signatures.
