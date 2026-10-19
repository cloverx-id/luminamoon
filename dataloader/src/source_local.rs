use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{Array, LargeStringArray, RecordBatch, StringArray, StringViewArray};
use memmap2::Mmap;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder, RowSelection,
    RowSelector,
};
use parquet::file::metadata::ParquetMetaData;

use crate::coord::StreamCoordinate;
use crate::dedup::TwoStageDeduplicator;
use crate::error::{DataLoaderError, Result};
use crate::postprocess::{
    PostProcessConfig, process_batch_parallel_indexed, process_single_record,
};

#[derive(Debug, Clone)]
pub struct FetchedBatch {
    pub batch_id: u64,
    pub records: Vec<String>,
    pub next_coord: StreamCoordinate,
    pub is_eof: bool,
}

pub struct LocalMmapLineSource {
    shard_id: u32,
    mmap: Mmap,
}

impl LocalMmapLineSource {
    pub fn open(shard_id: u32, path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { shard_id, mmap })
    }

    pub fn file_len(&self) -> u64 {
        self.mmap.len() as u64
    }

    pub fn read_batch(
        &self,
        batch_id: u64,
        start_coord: StreamCoordinate,
        batch_size: usize,
        post_cfg: &PostProcessConfig,
        mut dedup: Option<&mut TwoStageDeduplicator>,
    ) -> Result<FetchedBatch> {
        let total_len = self.mmap.len();
        let mut cursor = (start_coord.byte_offset as usize).min(total_len);
        let mut global_row = start_coord.global_row_idx;
        let mut candidates = Vec::with_capacity(batch_size * 2);

        while cursor < total_len && candidates.len() < batch_size {
            let slice = &self.mmap[cursor..];
            let rel_end = crate::simd_cpu::find_newline_fast(slice).unwrap_or(slice.len());
            let line_bytes = &slice[..rel_end];
            let step = if cursor + rel_end < total_len {
                rel_end + 1
            } else {
                rel_end
            };
            cursor += step;
            global_row += 1;

            if let Ok(raw_str) = std::str::from_utf8(line_bytes)
                && let Some(cleaned) = process_single_record(raw_str, post_cfg)
            {
                if let Some(d) = dedup.as_deref_mut() {
                    if d.try_stage(batch_id, cleaned.as_bytes()).is_some() {
                        candidates.push(cleaned);
                    }
                } else {
                    candidates.push(cleaned);
                }
            }
        }

        let next_coord = StreamCoordinate {
            shard_id: self.shard_id,
            byte_offset: cursor as u64,
            row_group_idx: 0,
            page_idx: 0,
            row_offset_in_page: 0,
            global_row_idx: global_row,
        };

        Ok(FetchedBatch {
            batch_id,
            records: candidates,
            next_coord,
            is_eof: cursor >= total_len,
        })
    }
}

pub struct LocalParquetSource {
    shard_id: u32,
    file: File,
    arrow_metadata: ArrowReaderMetadata,
    metadata: Arc<ParquetMetaData>,
    column_name: String,
    column_root_idx: usize,
    row_group_counts: Vec<usize>,
}

impl LocalParquetSource {
    pub fn open(shard_id: u32, path: impl AsRef<Path>, column_name: impl Into<String>) -> Result<Self> {
        let file = File::open(path)?;
        let arrow_metadata = ArrowReaderMetadata::load(&file, ArrowReaderOptions::new())?;
        let metadata = arrow_metadata.metadata().clone();
        let col_name = column_name.into();
        let schema_descr = metadata.file_metadata().schema_descr();

        let column_root_idx = schema_descr
            .root_schema()
            .get_fields()
            .iter()
            .position(|f| f.name() == col_name)
            .ok_or_else(|| DataLoaderError::ColumnNotFound(col_name.clone()))?;

        let row_group_counts = metadata
            .row_groups()
            .iter()
            .map(|rg| rg.num_rows() as usize)
            .collect();

        Ok(Self {
            shard_id,
            file,
            arrow_metadata,
            metadata,
            column_name: col_name,
            column_root_idx,
            row_group_counts,
        })
    }

    pub fn read_batch(
        &self,
        batch_id: u64,
        start_coord: StreamCoordinate,
        batch_size: usize,
        post_cfg: &PostProcessConfig,
        mut dedup: Option<&mut TwoStageDeduplicator>,
    ) -> Result<FetchedBatch> {
        let mut rg_idx = start_coord.row_group_idx as usize;
        let mut row_in_rg = start_coord.row_offset_in_page as usize;
        let mut global_row = start_coord.global_row_idx;
        let mut collected = Vec::with_capacity(batch_size);

        while rg_idx < self.row_group_counts.len() && collected.len() < batch_size {
            let rg_total = self.row_group_counts[rg_idx];
            if row_in_rg >= rg_total {
                rg_idx += 1;
                row_in_rg = 0;
                continue;
            }

            let remaining_needed = batch_size - collected.len();
            let take_rows = (rg_total - row_in_rg).min(remaining_needed * 2).max(remaining_needed);
            let actual_take = take_rows.min(rg_total - row_in_rg);

            let mut selectors = Vec::with_capacity(2);
            if row_in_rg > 0 {
                selectors.push(RowSelector::skip(row_in_rg));
            }
            selectors.push(RowSelector::select(actual_take));

            let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(
                self.file.try_clone()?,
                self.arrow_metadata.clone(),
            );
            let mask = ProjectionMask::roots(
                self.metadata.file_metadata().schema_descr(),
                [self.column_root_idx],
            );

            let reader = builder
                .with_row_groups(vec![rg_idx])
                .with_projection(mask)
                .with_row_selection(RowSelection::from(selectors))
                .with_batch_size(actual_take.max(1))
                .build()?;

            let mut raw_strings = Vec::with_capacity(actual_take);
            for batch_res in reader {
                let batch = batch_res?;
                extract_strings_from_batch(&batch, &self.column_name, &mut raw_strings)?;
            }

            let processed = process_batch_parallel_indexed(&raw_strings, post_cfg);
            let mut consumed_in_slice = 0usize;

            for maybe_cleaned in processed {
                consumed_in_slice += 1;
                if let Some(cleaned) = maybe_cleaned {
                    let accepted = if let Some(d) = dedup.as_deref_mut() {
                        d.try_stage(batch_id, cleaned.as_bytes()).is_some()
                    } else {
                        true
                    };
                    if accepted {
                        collected.push(cleaned);
                        if collected.len() >= batch_size {
                            break;
                        }
                    }
                }
            }

            row_in_rg += consumed_in_slice;
            global_row += consumed_in_slice as u64;

            if row_in_rg >= rg_total {
                rg_idx += 1;
                row_in_rg = 0;
            }
        }

        let is_eof = rg_idx >= self.row_group_counts.len();
        let next_coord = StreamCoordinate {
            shard_id: self.shard_id,
            byte_offset: 0,
            row_group_idx: rg_idx as u32,
            page_idx: 0,
            row_offset_in_page: row_in_rg as u32,
            global_row_idx: global_row,
        };

        Ok(FetchedBatch {
            batch_id,
            records: collected,
            next_coord,
            is_eof,
        })
    }
}

pub fn extract_strings_from_batch(
    batch: &RecordBatch,
    col_name: &str,
    out: &mut Vec<String>,
) -> Result<()> {
    let col = batch
        .column_by_name(col_name)
        .ok_or_else(|| DataLoaderError::ColumnNotFound(col_name.to_owned()))?;

    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        for i in 0..arr.len() {
            if !arr.is_null(i) {
                out.push(arr.value(i).to_owned());
            } else {
                out.push(String::new());
            }
        }
        return Ok(());
    }

    if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
        for i in 0..arr.len() {
            if !arr.is_null(i) {
                out.push(arr.value(i).to_owned());
            } else {
                out.push(String::new());
            }
        }
        return Ok(());
    }

    if let Some(arr) = col.as_any().downcast_ref::<StringViewArray>() {
        for i in 0..arr.len() {
            if !arr.is_null(i) {
                out.push(arr.value(i).to_owned());
            } else {
                out.push(String::new());
            }
        }
        return Ok(());
    }

    Err(DataLoaderError::InvalidConfig(format!(
        "Column '{col_name}' is not Utf8, LargeUtf8, or Utf8View"
    )))
}
