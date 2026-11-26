# Stream Coordinates & Ping-Pong Checkpoint Recovery

[`data/src/coord.rs`](https://github.com/cloverx-id/luminamoon/blob/main/data/src/coord.rs) and [`data/src/metadata.rs`](https://github.com/cloverx-id/luminamoon/blob/main/data/src/metadata.rs) define stream coordinate tracking and atomic dual-slot checkpointing, enabling dataset resume without re-reading or re-hashing earlier shards.

---

## 1. 32-Byte `StreamCoordinate` Binary Layout (`src/coord.rs`)

Positions across multi-shard streams are addressed by a fixed 32-byte little-endian struct:

| Byte Offset | Field | Rust Type | Width | Description |
| :--- | :--- | :--- | :--- | :--- |
| `0..4` | `shard_id` | `u32` | 4 B | Active index in the expanded shard URI list. |
| `4..12` | `byte_offset` | `u64` | 8 B | Exact byte offset for `.jsonl` / `.txt` `mmap` or HTTP 206 Range streams. |
| `12..16` | `row_group_idx` | `u32` | 4 B | Active Parquet Row Group index. |
| `16..20` | `page_idx` | `u32` | 4 B | Reserved page index within the Row Group. |
| `20..24` | `row_offset_in_page` | `u32` | 4 B | Exact row offset within the active Parquet Row Group (`RowSelector::skip`). |
| `24..32` | `global_row_idx` | `u64` | 8 B | Cumulative number of raw records scanned across all shards. |

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StreamCoordinate {
    pub shard_id: u32,
    pub byte_offset: u64,
    pub row_group_idx: u32,
    pub page_idx: u32,
    pub row_offset_in_page: u32,
    pub global_row_idx: u64,
}
```

---

## 2. 256-Byte `LMDL v2` Ping-Pong Checkpoint File (`src/metadata.rs`)

To prevent checkpoint corruption from sudden power loss or process interruption, `DualMetadataManager` maintains a fixed 256-byte `.meta` file divided into two 128-byte slots (`Slot A` at `0..128` and `Slot B` at `128..256`):

```mermaid
flowchart LR
    Commit["commit_gpu_batch(batch_id)"] --> WAL["Append Hashes to LMWL v2 .dedup WAL"]
    WAL --> Epoch["Increment Epoch Counter"]
    Epoch --> SlotA["Odd Epoch: Slot A (Bytes 0..128 + CRC32)"]
    Epoch --> SlotB["Even Epoch: Slot B (Bytes 128..256 + CRC32)"]
    SlotA --> Recover["Verify CRC32 & Select max(epoch)"]
    SlotB --> Recover
```

### 128-Byte `DiskSlot` Wire Format (`LMDL v2`)

| Byte Range | Field | Type | Description |
| :--- | :--- | :--- | :--- |
| `0..4` | `SLOT_MAGIC` | `[u8; 4]` | Constant magic header `*b"LMDL"`. |
| `4..8` | `SLOT_VERSION` | `u32` | Format version (`2`). |
| `8..16` | `epoch` | `u64` | Monotonically increasing commit sequence number. |
| `16..24` | `committed_batch_id` | `u64` | Last batch ID confirmed consumed by the training step. |
| `24..56` | `coord` | `[u8; 32]` | Encoded `StreamCoordinate` immediately following the committed batch. |
| `56..64` | `dedup_count` | `u64` | Total number of committed 16-byte `xxh3_128` hashes valid in `.meta.dedup`. |
| `64..72` | `consumed_tokens` | `u64` | Cumulative number of training tokens committed. |
| `72..76` | `hash_width_bytes` | `u32` | Hash byte width (`16` for `XXH3-128`). |
| `76..124` | `_reserved` | `[u8; 48]` | Zero-filled reserved bytes for future alignment. |
| `124..128` | `crc32` | `u32` | Hardware-accelerated IEEE CRC32 (`crc32fast`) over bytes `0..124`. |

---

## 3. Recovery & Truncation Invariants

When initializing `DualMetadataManager`:

1. Both `Slot A` and `Slot B` are decoded and verified against `SLOT_MAGIC`, `SLOT_VERSION`, and their 32-bit IEEE CRC32 checksum.
2. If both slots are valid, the slot with the higher `epoch` is selected:

   $\displaystyle \text{Slot} _ {\text{active}} = \arg\max _ {S \in \{A, B\}} S.\text{epoch}$

3. If one slot was torn during a system crash, its CRC32 check fails and the manager falls back to the intact partner slot.
4. The companion `.meta.dedup` write-ahead log (8-byte `LMWL` v2 header + 16-byte `u128` entries) is truncated to `8 + dedup_count * 16` bytes (`dedup_file.set_len(valid_dedup_bytes)`), discarding any uncommitted partial hash appends before reconstructing `TwoStageDeduplicator::with_committed`.
