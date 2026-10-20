use std::sync::Arc;

use rayon::prelude::*;
use xxhash_rust::xxh3::xxh3_64;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StringView16 {
    pub len: u32,
    pub prefix_or_inline: [u8; 4],
    pub suffix_or_offset: [u8; 8],
}

impl StringView16 {
    pub const INLINE_CAPACITY: usize = 12;

    #[inline]
    pub fn new(s: &str, buffer_id: u32, offset: u32) -> Self {
        let len = s.len() as u32;
        let bytes = s.as_bytes();
        let mut prefix_or_inline = [0u8; 4];
        let mut suffix_or_offset = [0u8; 8];

        if (len as usize) <= Self::INLINE_CAPACITY {
            let n = bytes.len();
            let first_part = n.min(4);
            prefix_or_inline[..first_part].copy_from_slice(&bytes[..first_part]);
            if n > 4 {
                suffix_or_offset[..n - 4].copy_from_slice(&bytes[4..n]);
            }
        } else {
            prefix_or_inline.copy_from_slice(&bytes[..4]);
            suffix_or_offset[0..4].copy_from_slice(&offset.to_le_bytes());
            suffix_or_offset[4..8].copy_from_slice(&buffer_id.to_le_bytes());
        }

        Self {
            len,
            prefix_or_inline,
            suffix_or_offset,
        }
    }

    #[inline]
    pub fn is_inlined(&self) -> bool {
        (self.len as usize) <= Self::INLINE_CAPACITY
    }

    #[inline]
    pub fn as_str<'a>(&'a self, pool: &'a [u8]) -> &'a str {
        if self.is_inlined() {
            let len = self.len as usize;
            // The 12 inline bytes immediately follow the 4-byte len in memory
            unsafe {
                let ptr = (self as *const Self as *const u8).add(4);
                let slice = std::slice::from_raw_parts(ptr, len);
                std::str::from_utf8_unchecked(slice)
            }
        } else {
            let offset =
                u32::from_le_bytes(self.suffix_or_offset[0..4].try_into().unwrap()) as usize;
            let len = self.len as usize;
            if let Some(slice) = pool.get(offset..offset + len) {
                unsafe { std::str::from_utf8_unchecked(slice) }
            } else {
                ""
            }
        }
    }

    #[inline]
    pub fn fast_eq<'a>(&'a self, other: &'a Self, pool: &'a [u8]) -> bool {
        if self.len != other.len {
            return false;
        }
        if self.prefix_or_inline != other.prefix_or_inline {
            return false;
        }
        if self.is_inlined() {
            return self.suffix_or_offset == other.suffix_or_offset;
        }
        self.as_str(pool) == other.as_str(pool)
    }

    #[inline]
    pub fn compute_hash<'a>(&'a self, pool: &'a [u8]) -> u64 {
        let s = self.as_str(pool);
        xxh3_64(s.as_bytes())
    }
}

#[derive(Clone, Debug, Default)]
pub struct StringTensor {
    pub views: Vec<StringView16>,
    pub pool: Arc<Vec<u8>>,
}

impl StringTensor {
    pub fn new() -> Self {
        Self {
            views: Vec::new(),
            pool: Arc::new(Vec::new()),
        }
    }

    pub fn from_strings<I, S>(strings: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let items: Vec<S> = strings.into_iter().collect();
        let mut total_ref_bytes = 0usize;
        for s in &items {
            let str_ref = s.as_ref();
            if str_ref.len() > StringView16::INLINE_CAPACITY {
                total_ref_bytes += str_ref.len();
            }
        }

        let mut pool = Vec::with_capacity(total_ref_bytes);
        let mut views = Vec::with_capacity(items.len());

        for s in items {
            let str_ref = s.as_ref();
            if str_ref.len() <= StringView16::INLINE_CAPACITY {
                views.push(StringView16::new(str_ref, 0, 0));
            } else {
                let offset = pool.len() as u32;
                pool.extend_from_slice(str_ref.as_bytes());
                views.push(StringView16::new(str_ref, 0, offset));
            }
        }

        Self {
            views,
            pool: Arc::new(pool),
        }
    }

    pub fn from_gpu_buffers(offsets: &[u32], payload: &[u8]) -> Self {
        if offsets.is_empty() {
            return Self::new();
        }

        let num_strings = offsets.len().saturating_sub(1);
        let mut views = Vec::with_capacity(num_strings);
        let pool = Arc::new(payload.to_vec());

        for i in 0..num_strings {
            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            if start <= end && end <= payload.len() {
                if let Ok(s) = std::str::from_utf8(&payload[start..end]) {
                    views.push(StringView16::new(s, 0, start as u32));
                } else {
                    views.push(StringView16::new("", 0, 0));
                }
            } else {
                views.push(StringView16::new("", 0, 0));
            }
        }

        Self { views, pool }
    }

    pub fn to_gpu_buffers(&self) -> (Vec<u32>, Vec<u8>) {
        let num_strings = self.views.len();
        let mut offsets = Vec::with_capacity(num_strings + 1);
        let mut total_chars = 0usize;

        for v in &self.views {
            total_chars += v.len as usize;
        }

        let mut payload = Vec::with_capacity(total_chars);
        let mut current_offset = 0u32;
        offsets.push(current_offset);

        for v in &self.views {
            let s = v.as_str(&self.pool);
            payload.extend_from_slice(s.as_bytes());
            current_offset += v.len;
            offsets.push(current_offset);
        }

        (offsets, payload)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.views.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<&str> {
        self.views.get(index).map(|v| v.as_str(&self.pool))
    }

    pub fn to_vec(&self) -> Vec<String> {
        self.views
            .iter()
            .map(|v| v.as_str(&self.pool).to_owned())
            .collect()
    }

    pub fn lengths(&self) -> Vec<u32> {
        self.views.iter().map(|v| v.len).collect()
    }

    pub fn is_inlined_mask(&self) -> Vec<bool> {
        self.views.iter().map(|v| v.is_inlined()).collect()
    }

    pub fn hash_all_parallel(&self) -> Vec<u64> {
        let pool = &self.pool;
        self.views
            .par_iter()
            .map(|v| v.compute_hash(pool))
            .collect()
    }

    pub fn filter_by_length(&self, min_len: u32, max_len: u32) -> Self {
        let mut new_views = Vec::new();
        for v in &self.views {
            let l = v.len;
            if l >= min_len && (max_len == 0 || l <= max_len) {
                new_views.push(*v);
            }
        }
        Self {
            views: new_views,
            pool: self.pool.clone(),
        }
    }

    pub fn filter_by_prefix(&self, prefix: &str) -> Self {
        let prefix_bytes = prefix.as_bytes();
        let pool = &self.pool;
        let mut new_views = Vec::new();

        for v in &self.views {
            let l = v.len as usize;
            if l < prefix_bytes.len() {
                continue;
            }
            // Fast prefix check
            if prefix_bytes.len() <= 4 {
                if v.prefix_or_inline[..prefix_bytes.len()] == prefix_bytes[..] {
                    new_views.push(*v);
                }
            } else if v.prefix_or_inline[..4] == prefix_bytes[..4] {
                let s = v.as_str(pool);
                if s.as_bytes().starts_with(prefix_bytes) {
                    new_views.push(*v);
                }
            }
        }

        Self {
            views: new_views,
            pool: self.pool.clone(),
        }
    }
}
