use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, LargeStringArray,
    RecordBatch, StringArray, StringViewArray,
};
use memmap2::Mmap;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder, RowSelection,
    RowSelector,
};
use parquet::file::metadata::ParquetMetaData;

use crate::constants::{
    MAGIC_ARROW_IPC, MAGIC_GIT_LFS_V1, MAGIC_GZIP, MAGIC_LZ4_FRAME, MAGIC_PARQUET, MAGIC_ZSTD,
};
use crate::coord::StreamCoordinate;
use crate::dedup::TwoStageDeduplicator;
use crate::error::{
    DataLoaderError, Result, column_not_found_with_suggestions, diagnose_parquet_file_error,
    inspect_magic_bytes_hint, open_local_file_with_diagnostics,
};
use crate::postprocess::{
    PostProcessConfig, collect_json_object_keys, process_batch_parallel_indexed,
};

#[derive(Debug, Clone)]
pub struct FetchedBatch {
    pub batch_id: u64,
    pub records: Vec<String>,
    pub next_coord: StreamCoordinate,
    pub is_eof: bool,
}

pub fn decompress_bytes_if_compressed(raw: &[u8], path_hint: &str) -> Result<Option<Vec<u8>>> {
    let lower = path_hint.to_ascii_lowercase();
    let is_gzip = raw.starts_with(MAGIC_GZIP) || lower.ends_with(".gz") || lower.ends_with(".tgz");
    let is_zstd =
        raw.starts_with(MAGIC_ZSTD) || lower.ends_with(".zst") || lower.ends_with(".zstd");
    let is_lz4 = raw.starts_with(MAGIC_LZ4_FRAME) || lower.ends_with(".lz4");

    if is_gzip {
        let mut decoder = flate2::read::MultiGzDecoder::new(raw);
        let mut out = Vec::new();
        decoder.read_to_end(&mut out).map_err(|e| {
            DataLoaderError::InvalidConfig(format!(
                "Failed to decompress Gzip stream '{path_hint}': {e}"
            ))
        })?;
        return Ok(Some(out));
    }

    if is_zstd {
        let out = zstd::stream::decode_all(raw).map_err(|e| {
            DataLoaderError::InvalidConfig(format!(
                "Failed to decompress Zstandard (.zst) stream '{path_hint}': {e}"
            ))
        })?;
        return Ok(Some(out));
    }

    if is_lz4 {
        let mut decoder = lz4_flex::frame::FrameDecoder::new(raw);
        let mut out = Vec::new();
        decoder.read_to_end(&mut out).map_err(|e| {
            DataLoaderError::InvalidConfig(format!(
                "Failed to decompress LZ4 frame stream '{path_hint}': {e}"
            ))
        })?;
        return Ok(Some(out));
    }

    Ok(None)
}

enum LineStorage {
    Mmap(Mmap),
    Buffer(Vec<u8>),
}

impl LineStorage {
    #[inline]
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mmap(m) => m,
            Self::Buffer(b) => b.as_slice(),
        }
    }
}

pub struct LocalMmapLineSource {
    shard_id: u32,
    storage: LineStorage,
}

impl LocalMmapLineSource {
    pub fn open(shard_id: u32, path: impl AsRef<Path>) -> Result<Self> {
        let p = path.as_ref();
        let file = open_local_file_with_diagnostics(p)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let path_str = p.to_string_lossy();

        if mmap.starts_with(MAGIC_PARQUET) {
            return Err(DataLoaderError::InvalidConfig(format!(
                "File '{}' has Apache Parquet magic bytes ('PAR1'), but was opened as a text/JSONL line source. Rename it with a '.parquet' extension or use MultimodalParquetLoader.",
                p.display()
            )));
        }
        if mmap.starts_with(MAGIC_ARROW_IPC) {
            return Err(DataLoaderError::InvalidConfig(format!(
                "File '{}' has Apache Arrow IPC/Feather magic bytes ('ARROW1'), but was opened as a text/JSONL line source. Rename it with a '.arrow' or '.feather' extension.",
                p.display()
            )));
        }
        if let Some(lfs_hint) =
            inspect_magic_bytes_hint(&mmap[..mmap.len().min(128)], "text/JSONL", &path_str)
            && mmap.starts_with(MAGIC_GIT_LFS_V1)
        {
            return Err(DataLoaderError::InvalidConfig(lfs_hint));
        }

        if let Some(decompressed) = decompress_bytes_if_compressed(&mmap, &path_str)? {
            return Ok(Self {
                shard_id,
                storage: LineStorage::Buffer(decompressed),
            });
        }

        Ok(Self {
            shard_id,
            storage: LineStorage::Mmap(mmap),
        })
    }

    pub fn file_len(&self) -> u64 {
        self.storage.as_slice().len() as u64
    }

    pub fn read_batch(
        &self,
        batch_id: u64,
        start_coord: StreamCoordinate,
        batch_size: usize,
        post_cfg: &PostProcessConfig,
        mut dedup: Option<&mut TwoStageDeduplicator>,
    ) -> Result<FetchedBatch> {
        let data = self.storage.as_slice();
        let total_len = data.len();
        let mut cursor = (start_coord.byte_offset as usize).min(total_len);
        let mut global_row = start_coord.global_row_idx;
        let mut candidates = Vec::with_capacity(batch_size);
        let mut sample_json_line: Option<String> = None;

        while cursor < total_len && candidates.len() < batch_size {
            let chunk_target = (batch_size - candidates.len()).clamp(1, 256);
            let mut raw_lines = Vec::with_capacity(chunk_target);
            let mut line_steps = Vec::with_capacity(chunk_target);
            let mut scan_cursor = cursor;

            while scan_cursor < total_len && raw_lines.len() < chunk_target {
                let slice = &data[scan_cursor..];
                let rel_end = crate::simd_cpu::find_newline_fast(slice).unwrap_or(slice.len());
                let line_bytes = &slice[..rel_end];
                let step = if scan_cursor + rel_end < total_len {
                    rel_end + 1
                } else {
                    rel_end
                };
                scan_cursor += step;

                match std::str::from_utf8(line_bytes) {
                    Ok(raw_str) => {
                        if sample_json_line.is_none() && !raw_str.trim().is_empty() {
                            sample_json_line = Some(raw_str.to_owned());
                        }
                        raw_lines.push(raw_str.to_owned());
                        line_steps.push((step, true));
                    }
                    Err(_) => {
                        raw_lines.push(String::new());
                        line_steps.push((step, false));
                    }
                }
            }

            let processed = process_batch_parallel_indexed(&raw_lines, post_cfg);
            for (maybe_cleaned, (step, is_valid_utf8)) in processed.into_iter().zip(line_steps) {
                if candidates.len() >= batch_size {
                    break;
                }
                cursor += step;
                global_row += 1;

                if !is_valid_utf8 {
                    continue;
                }

                if let Some(cleaned) = maybe_cleaned {
                    if let Some(d) = dedup.as_deref_mut() {
                        if d.try_stage(batch_id, cleaned.as_bytes()).is_some() {
                            candidates.push(cleaned);
                        }
                    } else {
                        candidates.push(cleaned);
                    }
                }
            }
        }

        if candidates.is_empty()
            && let Some(ref req_key) = post_cfg.json_extract_key
            && let Some(ref first_line) = sample_json_line
            && let Some(avail_keys) = collect_json_object_keys(first_line)
        {
            let top_segment = req_key.split('.').next().unwrap_or(req_key);
            if !avail_keys.iter().any(|k| k == req_key || k == top_segment) {
                return Err(column_not_found_with_suggestions(req_key, &avail_keys));
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
    min_score_filter: Option<(String, f64, usize)>,
}

pub fn resolve_parquet_text_column(
    schema_descr: &parquet::schema::types::SchemaDescriptor,
    requested: &str,
) -> Result<(String, usize)> {
    let fields = schema_descr.root_schema().get_fields();
    if let Some(idx) = fields.iter().position(|f| f.name() == requested) {
        return Ok((requested.to_owned(), idx));
    }

    if requested == "text" {
        if let Some((idx, field)) = fields.iter().enumerate().find(|(_, f)| {
            f.is_primitive()
                && (f.get_basic_info().converted_type() == parquet::basic::ConvertedType::UTF8
                    || matches!(
                        f.get_basic_info().logical_type_ref(),
                        Some(parquet::basic::LogicalType::String)
                    ))
        }) {
            return Ok((field.name().to_owned(), idx));
        }

        if let Some((idx, field)) = fields.iter().enumerate().find(|(_, f)| {
            f.is_primitive() && f.get_physical_type() == parquet::basic::Type::BYTE_ARRAY
        }) {
            return Ok((field.name().to_owned(), idx));
        }
    }

    let available: Vec<&str> = fields.iter().map(|f| f.name()).collect();
    Err(column_not_found_with_suggestions(requested, &available))
}

pub fn resolve_parquet_score_column(
    schema_descr: &parquet::schema::types::SchemaDescriptor,
    requested: Option<&str>,
) -> Result<(String, usize)> {
    let fields = schema_descr.root_schema().get_fields();
    let available: Vec<&str> = fields.iter().map(|f| f.name()).collect();
    if let Some(req) = requested.filter(|s| !s.is_empty()) {
        if let Some(idx) = fields.iter().position(|f| f.name() == req) {
            return Ok((req.to_owned(), idx));
        }
        return Err(column_not_found_with_suggestions(req, &available));
    }

    if let Some((idx, field)) = fields.iter().enumerate().find(|(_, f)| {
        f.is_primitive()
            && matches!(
                f.get_physical_type(),
                parquet::basic::Type::DOUBLE | parquet::basic::Type::FLOAT
            )
    }) {
        return Ok((field.name().to_owned(), idx));
    }

    if let Some((idx, field)) = fields.iter().enumerate().find(|(_, f)| {
        f.is_primitive()
            && matches!(
                f.get_physical_type(),
                parquet::basic::Type::INT64 | parquet::basic::Type::INT32
            )
    }) {
        return Ok((field.name().to_owned(), idx));
    }

    Err(column_not_found_with_suggestions("score", &available))
}

impl LocalParquetSource {
    pub fn open(
        shard_id: u32,
        path: impl AsRef<Path>,
        column_name: impl Into<String>,
    ) -> Result<Self> {
        let p = path.as_ref();
        let file = open_local_file_with_diagnostics(p)?;
        let arrow_metadata = ArrowReaderMetadata::load(&file, ArrowReaderOptions::new())
            .map_err(|e| diagnose_parquet_file_error(p, e))?;
        let metadata = arrow_metadata.metadata().clone();
        let requested_col = column_name.into();
        let schema_descr = metadata.file_metadata().schema_descr();
        let (col_name, column_root_idx) =
            resolve_parquet_text_column(schema_descr, &requested_col)?;

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
            min_score_filter: None,
        })
    }

    pub fn with_min_score_filter(mut self, filter: Option<(Option<String>, f64)>) -> Result<Self> {
        if let Some((score_col_opt, min_score)) = filter {
            let schema_descr = self.metadata.file_metadata().schema_descr();
            let (resolved_col, score_idx) =
                resolve_parquet_score_column(schema_descr, score_col_opt.as_deref())?;
            self.min_score_filter = Some((resolved_col, min_score, score_idx));
        } else {
            self.min_score_filter = None;
        }
        Ok(self)
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
            let take_rows = (rg_total - row_in_rg)
                .min(remaining_needed * 2)
                .max(remaining_needed);
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
            let mut proj_roots = vec![self.column_root_idx];
            if let Some((_, _, score_idx)) = self.min_score_filter
                && score_idx != self.column_root_idx
            {
                proj_roots.push(score_idx);
            }
            let mask =
                ProjectionMask::roots(self.metadata.file_metadata().schema_descr(), proj_roots);

            let reader = builder
                .with_row_groups(vec![rg_idx])
                .with_projection(mask)
                .with_row_selection(RowSelection::from(selectors))
                .with_batch_size(actual_take.max(1))
                .build()?;

            let mut raw_strings = Vec::with_capacity(actual_take);
            let mut score_mask = Vec::new();
            for batch_res in reader {
                let batch = batch_res?;
                extract_strings_from_batch(&batch, &self.column_name, &mut raw_strings)?;
                if let Some((ref score_col, min_score, _)) = self.min_score_filter {
                    score_mask.extend(crate::multimodal::extract_score_mask_from_batch(
                        &batch, score_col, min_score,
                    )?);
                }
            }

            let processed = process_batch_parallel_indexed(&raw_strings, post_cfg);
            let mut consumed_in_slice = 0usize;

            for (idx, maybe_cleaned) in processed.into_iter().enumerate() {
                consumed_in_slice += 1;
                if !score_mask.is_empty() && !score_mask[idx] {
                    continue;
                }
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
    let Some(col) = batch.column_by_name(col_name) else {
        let schema = batch.schema();
        let available: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        return Err(column_not_found_with_suggestions(col_name, &available));
    };

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

    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
        for i in 0..arr.len() {
            out.push(if arr.is_null(i) {
                String::new()
            } else {
                arr.value(i).to_string()
            });
        }
        return Ok(());
    }

    if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
        for i in 0..arr.len() {
            out.push(if arr.is_null(i) {
                String::new()
            } else {
                arr.value(i).to_string()
            });
        }
        return Ok(());
    }

    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        for i in 0..arr.len() {
            out.push(if arr.is_null(i) {
                String::new()
            } else {
                arr.value(i).to_string()
            });
        }
        return Ok(());
    }

    if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
        for i in 0..arr.len() {
            out.push(if arr.is_null(i) {
                String::new()
            } else {
                arr.value(i).to_string()
            });
        }
        return Ok(());
    }

    if let Some(arr) = col.as_any().downcast_ref::<BooleanArray>() {
        for i in 0..arr.len() {
            out.push(if arr.is_null(i) {
                String::new()
            } else {
                arr.value(i).to_string()
            });
        }
        return Ok(());
    }

    Err(DataLoaderError::InvalidConfig(format!(
        "Column '{col_name}' has Arrow type {:?}, which cannot be streamed as text. Use columns=['{col_name}'] (MultimodalBatch) for binary/structured columns.",
        col.data_type()
    )))
}
