use crate::constants::STREAM_COORDINATE_BYTE_LEN;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StreamCoordinate {
    pub shard_id: u32,
    pub byte_offset: u64,
    pub row_group_idx: u32,
    pub page_idx: u32,
    pub row_offset_in_page: u32,
    pub global_row_idx: u64,
}

impl StreamCoordinate {
    pub const BYTE_LEN: usize = STREAM_COORDINATE_BYTE_LEN;

    #[inline]
    pub fn to_bytes(&self) -> [u8; Self::BYTE_LEN] {
        let mut buf = [0u8; Self::BYTE_LEN];
        buf[0..4].copy_from_slice(&self.shard_id.to_le_bytes());
        buf[4..12].copy_from_slice(&self.byte_offset.to_le_bytes());
        buf[12..16].copy_from_slice(&self.row_group_idx.to_le_bytes());
        buf[16..20].copy_from_slice(&self.page_idx.to_le_bytes());
        buf[20..24].copy_from_slice(&self.row_offset_in_page.to_le_bytes());
        buf[24..32].copy_from_slice(&self.global_row_idx.to_le_bytes());
        buf
    }

    #[inline]
    pub fn from_bytes(buf: &[u8; Self::BYTE_LEN]) -> Self {
        Self {
            shard_id: u32::from_le_bytes(buf[0..4].try_into().unwrap()),
            byte_offset: u64::from_le_bytes(buf[4..12].try_into().unwrap()),
            row_group_idx: u32::from_le_bytes(buf[12..16].try_into().unwrap()),
            page_idx: u32::from_le_bytes(buf[16..20].try_into().unwrap()),
            row_offset_in_page: u32::from_le_bytes(buf[20..24].try_into().unwrap()),
            global_row_idx: u64::from_le_bytes(buf[24..32].try_into().unwrap()),
        }
    }
}
