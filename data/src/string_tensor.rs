use std::sync::Arc;

use arrow_array::array::Array;
use arrow_array::StringViewArray;
use arrow_buffer::{Buffer, ScalarBuffer};
use rayon::prelude::*;
use xxhash_rust::xxh3::xxh3_64;

use crate::error::{DataLoaderError, Result};

const MAX_BUFFER_BYTES: usize = i32::MAX as usize;

/// 16-byte string view binary-compatible with Apache Arrow `Utf8View` (`ByteView`):
/// - `len <= 12`: `[len: u32][inline: [u8; 12]]`
/// - `len > 12`:  `[len: u32][prefix: [u8; 4]][buffer_index: u32][offset: u32]`
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StringView16 {
    len: u32,
    prefix_or_inline: [u8; 4],
    buffer_id_or_inline: [u8; 4],
    offset_or_inline: [u8; 4],
}

impl StringView16 {
    pub const INLINE_CAPACITY: usize = 12;

    #[inline]
    pub fn try_new(s: &str, buffer_id: u32, offset: u32) -> Result<Self> {
        let len = u32::try_from(s.len()).map_err(|_| {
            DataLoaderError::InvalidConfig("String length exceeds u32::MAX".to_owned())
        })?;
        let bytes = s.as_bytes();
        let mut prefix_or_inline = [0u8; 4];
        let mut buffer_id_or_inline = [0u8; 4];
        let mut offset_or_inline = [0u8; 4];

        if (len as usize) <= Self::INLINE_CAPACITY {
            let n = bytes.len();
            let p1 = n.min(4);
            prefix_or_inline[..p1].copy_from_slice(&bytes[..p1]);
            if n > 4 {
                let p2 = (n - 4).min(4);
                buffer_id_or_inline[..p2].copy_from_slice(&bytes[4..4 + p2]);
            }
            if n > 8 {
                let p3 = n - 8;
                offset_or_inline[..p3].copy_from_slice(&bytes[8..8 + p3]);
            }
        } else {
            prefix_or_inline.copy_from_slice(&bytes[..4]);
            buffer_id_or_inline.copy_from_slice(&buffer_id.to_le_bytes());
            offset_or_inline.copy_from_slice(&offset.to_le_bytes());
        }

        Ok(Self {
            len,
            prefix_or_inline,
            buffer_id_or_inline,
            offset_or_inline,
        })
    }

    #[inline]
    pub fn new(s: &str, buffer_id: u32, offset: u32) -> Self {
        Self::try_new(s, buffer_id, offset).expect("StringView16 string length exceeds u32::MAX")
    }

    #[inline]
    pub fn len(&self) -> u32 {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn is_inlined(&self) -> bool {
        (self.len as usize) <= Self::INLINE_CAPACITY
    }

    #[inline]
    pub fn to_arrow_u128(&self) -> u128 {
        let mut raw = [0u8; 16];
        raw[0..4].copy_from_slice(&self.len.to_le_bytes());
        raw[4..8].copy_from_slice(&self.prefix_or_inline);
        raw[8..12].copy_from_slice(&self.buffer_id_or_inline);
        raw[12..16].copy_from_slice(&self.offset_or_inline);
        u128::from_le_bytes(raw)
    }

    #[inline]
    pub fn as_str<'a>(&'a self, buffers: &'a [Arc<Vec<u8>>]) -> &'a str {
        if self.is_inlined() {
            let len = (self.len as usize).min(Self::INLINE_CAPACITY);
            // Layout is #[repr(C)] with 12 contiguous inline UTF-8 bytes immediately after `len`.
            unsafe {
                let ptr = (self as *const Self as *const u8).add(4);
                let slice = std::slice::from_raw_parts(ptr, len);
                std::str::from_utf8_unchecked(slice)
            }
        } else {
            let buf_idx = u32::from_le_bytes(self.buffer_id_or_inline) as usize;
            let offset = u32::from_le_bytes(self.offset_or_inline) as usize;
            let len = self.len as usize;
            if let Some(pool) = buffers.get(buf_idx)
                && let Some(end) = offset.checked_add(len)
                && let Some(slice) = pool.get(offset..end)
            {
                // All strings inserted into `buffers` are already validated UTF-8.
                return unsafe { std::str::from_utf8_unchecked(slice) };
            }
            ""
        }
    }

    #[inline]
    pub fn fast_eq<'a>(&'a self, other: &'a Self, buffers: &'a [Arc<Vec<u8>>]) -> bool {
        if self.len != other.len || self.prefix_or_inline != other.prefix_or_inline {
            return false;
        }
        if self.is_inlined() {
            return self.buffer_id_or_inline == other.buffer_id_or_inline
                && self.offset_or_inline == other.offset_or_inline;
        }
        self.as_str(buffers) == other.as_str(buffers)
    }

    #[inline]
    pub fn compute_hash<'a>(&'a self, buffers: &'a [Arc<Vec<u8>>]) -> u64 {
        xxh3_64(self.as_str(buffers).as_bytes())
    }
}

#[derive(Clone, Debug, Default)]
pub struct StringTensor {
    pub views: Vec<StringView16>,
    pub buffers: Vec<Arc<Vec<u8>>>,
}

impl StringTensor {
    pub fn new() -> Self {
        Self {
            views: Vec::new(),
            buffers: Vec::new(),
        }
    }

    pub fn try_from_strings<I, S>(strings: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let iter = strings.into_iter();
        let (lower, _) = iter.size_hint();
        let mut views = Vec::with_capacity(lower);
        let mut buffers: Vec<Arc<Vec<u8>>> = Vec::new();
        let mut cur_buf: Vec<u8> = Vec::new();

        for s in iter {
            let str_ref = s.as_ref();
            let bytes = str_ref.as_bytes();
            if bytes.len() <= StringView16::INLINE_CAPACITY {
                views.push(StringView16::try_new(str_ref, 0, 0)?);
            } else {
                if bytes.len() > MAX_BUFFER_BYTES {
                    return Err(DataLoaderError::InvalidConfig(
                        "Single string exceeds Arrow 2 GiB buffer limit".to_owned(),
                    ));
                }
                if cur_buf
                    .len()
                    .checked_add(bytes.len())
                    .is_none_or(|end| end > MAX_BUFFER_BYTES)
                {
                    buffers.push(Arc::new(std::mem::take(&mut cur_buf)));
                }
                let buf_id = u32::try_from(buffers.len()).map_err(|_| {
                    DataLoaderError::InvalidConfig("StringTensor buffer count overflow".to_owned())
                })?;
                let offset = cur_buf.len() as u32;
                cur_buf.extend_from_slice(bytes);
                views.push(StringView16::try_new(str_ref, buf_id, offset)?);
            }
        }
        if !cur_buf.is_empty() {
            buffers.push(Arc::new(cur_buf));
        }
        Ok(Self { views, buffers })
    }

    pub fn from_strings<I, S>(strings: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::try_from_strings(strings).expect("StringTensor exceeded memory/offset limits")
    }

    pub fn try_from_gpu_buffers(offsets: &[u32], payload: &[u8]) -> Result<Self> {
        if offsets.is_empty() {
            return Ok(Self::new());
        }
        let num_strings = offsets.len().saturating_sub(1);
        let mut views = Vec::with_capacity(num_strings);
        let mut has_non_inlined = false;

        for i in 0..num_strings {
            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            if start > end || end > payload.len() {
                return Err(DataLoaderError::InvalidConfig(format!(
                    "Invalid GPU string offset slice [{start}..{end}] for payload length {}",
                    payload.len()
                )));
            }
            let s = std::str::from_utf8(&payload[start..end]).map_err(|e| {
                DataLoaderError::InvalidConfig(format!("Invalid UTF-8 in GPU string payload: {e}"))
            })?;
            if s.len() > StringView16::INLINE_CAPACITY {
                has_non_inlined = true;
            }
            views.push(StringView16::try_new(s, 0, start as u32)?);
        }

        let buffers = if has_non_inlined {
            vec![Arc::new(payload.to_vec())]
        } else {
            Vec::new()
        };
        Ok(Self { views, buffers })
    }

    pub fn from_gpu_buffers(offsets: &[u32], payload: &[u8]) -> Self {
        Self::try_from_gpu_buffers(offsets, payload).unwrap_or_else(|_| Self::new())
    }

    pub fn try_to_gpu_buffers(&self) -> Result<(Vec<u32>, Vec<u8>)> {
        let total_bytes: usize = self.views.iter().map(|v| v.len as usize).sum();
        if total_bytes > u32::MAX as usize {
            return Err(DataLoaderError::InvalidConfig(format!(
                "StringTensor total payload ({total_bytes} bytes) exceeds 32-bit GPU offset limit"
            )));
        }

        let mut offsets = Vec::with_capacity(self.views.len() + 1);
        let mut payload = Vec::with_capacity(total_bytes);
        let mut current_offset = 0u32;
        offsets.push(current_offset);

        for v in &self.views {
            let s = v.as_str(&self.buffers);
            payload.extend_from_slice(s.as_bytes());
            current_offset += v.len;
            offsets.push(current_offset);
        }
        Ok((offsets, payload))
    }

    pub fn to_gpu_buffers(&self) -> (Vec<u32>, Vec<u8>) {
        self.try_to_gpu_buffers()
            .expect("StringTensor exceeds 32-bit GPU offset limit")
    }

    pub fn try_to_offsets(&self) -> Result<Vec<u32>> {
        let mut offsets = Vec::with_capacity(self.views.len() + 1);
        let mut current_offset = 0u32;
        offsets.push(current_offset);
        for v in &self.views {
            current_offset = current_offset.checked_add(v.len).ok_or_else(|| {
                DataLoaderError::InvalidConfig("StringTensor offsets exceed u32::MAX".to_owned())
            })?;
            offsets.push(current_offset);
        }
        Ok(offsets)
    }

    pub fn to_offsets(&self) -> Vec<u32> {
        self.try_to_offsets()
            .expect("StringTensor offsets exceed u32::MAX")
    }

    pub fn to_payload(&self) -> Vec<u8> {
        let total_chars: usize = self.views.iter().map(|v| v.len as usize).sum();
        let mut payload = Vec::with_capacity(total_chars);
        for v in &self.views {
            payload.extend_from_slice(v.as_str(&self.buffers).as_bytes());
        }
        payload
    }

    /// Builds a spec-compliant Apache Arrow `StringViewArray` directly from `views` and `buffers`.
    pub fn to_arrow_string_view(&self) -> Result<StringViewArray> {
        let raw_views: Vec<u128> = self.views.iter().map(|v| v.to_arrow_u128()).collect();
        let scalar_views = ScalarBuffer::from(raw_views);
        let arrow_buffers: Vec<Buffer> = self
            .buffers
            .iter()
            .map(|b| Buffer::from_slice_ref(b.as_slice()))
            .collect();
        StringViewArray::try_new(scalar_views, arrow_buffers, None).map_err(DataLoaderError::Arrow)
    }

    /// Imports from an Arrow `StringViewArray`, `StringArray`, or `LargeStringArray`.
    pub fn from_arrow_array(array: &dyn Array) -> Result<Self> {
        if let Some(sv) = array.as_any().downcast_ref::<StringViewArray>() {
            let mut items = Vec::with_capacity(sv.len());
            for i in 0..sv.len() {
                items.push(if sv.is_null(i) { "" } else { sv.value(i) });
            }
            return Self::try_from_strings(items);
        }
        if let Some(sa) = array.as_any().downcast_ref::<arrow_array::StringArray>() {
            let mut items = Vec::with_capacity(sa.len());
            for i in 0..sa.len() {
                items.push(if sa.is_null(i) { "" } else { sa.value(i) });
            }
            return Self::try_from_strings(items);
        }
        if let Some(lsa) = array.as_any().downcast_ref::<arrow_array::LargeStringArray>() {
            let mut items = Vec::with_capacity(lsa.len());
            for i in 0..lsa.len() {
                items.push(if lsa.is_null(i) { "" } else { lsa.value(i) });
            }
            return Self::try_from_strings(items);
        }
        Err(DataLoaderError::InvalidConfig(format!(
            "Unsupported Arrow array data type for StringTensor: {:?}",
            array.data_type()
        )))
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
        self.views.get(index).map(|v| v.as_str(&self.buffers))
    }

    pub fn to_vec(&self) -> Vec<String> {
        self.views
            .iter()
            .map(|v| v.as_str(&self.buffers).to_owned())
            .collect()
    }

    pub fn lengths(&self) -> Vec<u32> {
        self.views.iter().map(|v| v.len).collect()
    }

    pub fn is_inlined_mask(&self) -> Vec<bool> {
        self.views.iter().map(|v| v.is_inlined()).collect()
    }

    pub fn hash_all_parallel(&self) -> Vec<u64> {
        let buffers = &self.buffers;
        self.views
            .par_iter()
            .map(|v| v.compute_hash(buffers))
            .collect()
    }

    pub fn filter_by_length(&self, min_len: u32, max_len: u32) -> Self {
        let max_opt = if max_len == 0 { None } else { Some(max_len) };
        self.filter_by_length_opt(min_len, max_opt)
    }

    pub fn filter_by_length_opt(&self, min_len: u32, max_len: Option<u32>) -> Self {
        let mut new_views = Vec::new();
        for v in &self.views {
            let l = v.len;
            if l >= min_len && max_len.is_none_or(|mx| l <= mx) {
                new_views.push(*v);
            }
        }
        Self {
            views: new_views,
            buffers: self.buffers.clone(),
        }
    }

    pub fn filter_by_prefix(&self, prefix: &str) -> Self {
        let prefix_bytes = prefix.as_bytes();
        let buffers = &self.buffers;
        let mut new_views = Vec::new();

        for v in &self.views {
            let l = v.len as usize;
            if l < prefix_bytes.len() {
                continue;
            }
            if prefix_bytes.len() <= 4 {
                if v.prefix_or_inline[..prefix_bytes.len()] == prefix_bytes[..] {
                    new_views.push(*v);
                }
            } else if v.prefix_or_inline[..4] == prefix_bytes[..4]
                && v.as_str(buffers).as_bytes().starts_with(prefix_bytes)
            {
                new_views.push(*v);
            }
        }

        Self {
            views: new_views,
            buffers: self.buffers.clone(),
        }
    }
}
