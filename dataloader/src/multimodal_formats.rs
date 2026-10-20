use std::io::Cursor;

use arrow_array::RecordBatch;
use arrow_ipc::reader::{FileReader as ArrowIpcFileReader, StreamReader as ArrowIpcStreamReader};
use arrow_schema::DataType;
use hashbrown::HashMap;

use crate::constants::{
    ARROW_IPC_MAGIC_BYTES, JSON_ARRAY_WRAPPER_KEYS, TAR_USTAR_BLOCK_SIZE,
    WEBDATASET_TEXT_EXTENSIONS,
};
use crate::coord::StreamCoordinate;
use crate::dedup::TwoStageDeduplicator;
use crate::error::{DataLoaderError, Result, column_not_found_with_suggestions};
use crate::multimodal::{
    MultimodalBatch, MultimodalValue, decode_base64, extract_multimodal_column,
    extract_score_mask_from_batch,
};
use crate::source_local::decompress_bytes_if_compressed;

pub fn classify_string_or_json(s: &str) -> MultimodalValue {
    let trimmed = s.trim();
    if ((trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']')))
        && serde_json::from_str::<serde_json::Value>(trimmed).is_ok()
    {
        MultimodalValue::Json(trimmed.to_owned())
    } else {
        MultimodalValue::Text(s.to_owned())
    }
}

pub fn json_value_to_multimodal(v: &serde_json::Value) -> MultimodalValue {
    match v {
        serde_json::Value::Null => MultimodalValue::Null,
        serde_json::Value::Bool(b) => MultimodalValue::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                MultimodalValue::Int(i)
            } else if let Some(u) = n.as_u64() {
                MultimodalValue::Int(u as i64)
            } else if let Some(f) = n.as_f64() {
                MultimodalValue::Float(f)
            } else {
                MultimodalValue::Null
            }
        }
        serde_json::Value::String(s) => classify_string_or_json(s),
        serde_json::Value::Array(arr) => {
            MultimodalValue::List(arr.iter().map(json_value_to_multimodal).collect())
        }
        serde_json::Value::Object(map) => {
            if let Some(serde_json::Value::String(b64)) = map.get("bytes")
                && map.len() <= 2
                && let Some(decoded) = decode_base64(b64)
            {
                return MultimodalValue::Bytes(decoded);
            }
            MultimodalValue::Json(v.to_string())
        }
    }
}

#[derive(Clone)]
pub struct InMemoryMultimodalTable {
    pub column_names: Vec<String>,
    pub columns: HashMap<String, Vec<MultimodalValue>>,
    pub num_rows: usize,
}

impl InMemoryMultimodalTable {
    pub fn from_record_batches(
        batches: &[RecordBatch],
        requested_cols: &[String],
        min_score_filter: Option<(Option<String>, f64)>,
    ) -> Result<Self> {
        if batches.is_empty() {
            return Ok(Self {
                column_names: requested_cols.to_vec(),
                columns: requested_cols
                    .iter()
                    .map(|c| (c.clone(), Vec::new()))
                    .collect(),
                num_rows: 0,
            });
        }
        let schema = batches[0].schema();
        let avail_names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
        let selected_cols = if requested_cols.is_empty() {
            avail_names.clone()
        } else {
            for req in requested_cols {
                if !avail_names.iter().any(|a| a == req) {
                    return Err(column_not_found_with_suggestions(req, &avail_names));
                }
            }
            requested_cols.to_vec()
        };

        let resolved_score_col = if let Some((score_col_opt, min_score)) = min_score_filter {
            let sc = if let Some(explicit) = score_col_opt.filter(|s| !s.is_empty()) {
                if !avail_names.iter().any(|a| a == &explicit) {
                    return Err(column_not_found_with_suggestions(&explicit, &avail_names));
                }
                explicit
            } else if avail_names.iter().any(|a| a == "score") {
                "score".to_owned()
            } else {
                schema
                    .fields()
                    .iter()
                    .find(|f| {
                        matches!(
                            f.data_type(),
                            DataType::Float64
                                | DataType::Float32
                                | DataType::Float16
                                | DataType::Int64
                                | DataType::Int32
                        )
                    })
                    .map(|f| f.name().clone())
                    .ok_or_else(|| column_not_found_with_suggestions("score", &avail_names))?
            };
            Some((sc, min_score))
        } else {
            None
        };

        let mut columns: HashMap<String, Vec<MultimodalValue>> = selected_cols
            .iter()
            .map(|c| (c.clone(), Vec::new()))
            .collect();
        let mut total_rows = 0usize;

        for batch in batches {
            let mask = if let Some((ref sc, min_s)) = resolved_score_col {
                Some(extract_score_mask_from_batch(batch, sc, min_s)?)
            } else {
                None
            };
            let mut extracted: Vec<Vec<MultimodalValue>> = Vec::with_capacity(selected_cols.len());
            for c_name in &selected_cols {
                extracted.push(extract_multimodal_column(batch, c_name)?);
            }
            for r in 0..batch.num_rows() {
                if let Some(ref m) = mask
                    && !m[r]
                {
                    continue;
                }
                for (c_idx, c_name) in selected_cols.iter().enumerate() {
                    if let Some(v) = columns.get_mut(c_name) {
                        v.push(extracted[c_idx][r].clone());
                    }
                }
                total_rows += 1;
            }
        }

        Ok(Self {
            column_names: selected_cols,
            columns,
            num_rows: total_rows,
        })
    }

    pub fn from_json_rows(
        rows: &[serde_json::Value],
        requested_cols: &[String],
        min_score_filter: Option<(Option<String>, f64)>,
    ) -> Result<Self> {
        let mut discovered_cols: Vec<String> = Vec::new();
        for row in rows {
            if let Some(obj) = row.as_object() {
                for k in obj.keys() {
                    if !discovered_cols.contains(k) {
                        discovered_cols.push(k.clone());
                    }
                }
            }
        }

        let selected_cols = if requested_cols.is_empty() {
            discovered_cols.clone()
        } else {
            for req in requested_cols {
                let top = req.split('.').next().unwrap_or(req);
                if !discovered_cols.is_empty()
                    && !discovered_cols.iter().any(|c| c == req || c == top)
                {
                    return Err(column_not_found_with_suggestions(req, &discovered_cols));
                }
            }
            requested_cols.to_vec()
        };

        let resolved_score = if let Some((score_col_opt, min_score)) = min_score_filter {
            let sc = score_col_opt
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "score".to_owned());
            let top = sc.split('.').next().unwrap_or(&sc);
            if !discovered_cols.is_empty() && !discovered_cols.iter().any(|c| c == &sc || c == top)
            {
                return Err(column_not_found_with_suggestions(&sc, &discovered_cols));
            }
            Some((sc, min_score))
        } else {
            None
        };

        let mut columns: HashMap<String, Vec<MultimodalValue>> = selected_cols
            .iter()
            .map(|c| (c.clone(), Vec::new()))
            .collect();
        let mut num_rows = 0usize;

        for row in rows {
            if let Some((ref sc, min_s)) = resolved_score {
                let pass = crate::postprocess::extract_json_value_by_path(row, sc)
                    .and_then(|v| v.as_f64().or_else(|| v.as_i64().map(|i| i as f64)))
                    .is_some_and(|val| val >= min_s);
                if !pass {
                    continue;
                }
            }
            for col_name in &selected_cols {
                let val = crate::postprocess::extract_json_value_by_path(row, col_name)
                    .map(json_value_to_multimodal)
                    .unwrap_or(MultimodalValue::Null);
                if let Some(vec_target) = columns.get_mut(col_name) {
                    vec_target.push(val);
                }
            }
            num_rows += 1;
        }

        Ok(Self {
            column_names: selected_cols,
            columns,
            num_rows,
        })
    }

    pub fn from_bytes_auto(
        raw_bytes: &[u8],
        path_hint: &str,
        requested_cols: &[String],
        min_score_filter: Option<(Option<String>, f64)>,
    ) -> Result<Self> {
        let decomp_opt = decompress_bytes_if_compressed(raw_bytes, path_hint)?;
        let bytes = decomp_opt.as_deref().unwrap_or(raw_bytes);
        let clean_path = path_hint
            .split('?')
            .next()
            .unwrap_or(path_hint)
            .trim_end_matches(".gz")
            .trim_end_matches(".zst")
            .trim_end_matches(".zstd")
            .trim_end_matches(".lz4")
            .to_ascii_lowercase();

        if bytes.starts_with(ARROW_IPC_MAGIC_BYTES)
            || clean_path.ends_with(".arrow")
            || clean_path.ends_with(".feather")
            || clean_path.ends_with(".ipc")
        {
            let mut batches = Vec::new();
            if bytes.starts_with(ARROW_IPC_MAGIC_BYTES) {
                let reader = ArrowIpcFileReader::try_new(Cursor::new(bytes), None)?;
                for b in reader {
                    batches.push(b?);
                }
            } else {
                let reader = ArrowIpcStreamReader::try_new(Cursor::new(bytes), None)?;
                for b in reader {
                    batches.push(b?);
                }
            }
            return Self::from_record_batches(&batches, requested_cols, min_score_filter);
        }

        if clean_path.ends_with(".tar") || path_hint.to_ascii_lowercase().ends_with(".tgz") {
            return parse_webdataset_tar_bytes(bytes, requested_cols, min_score_filter);
        }

        if clean_path.ends_with(".csv") || clean_path.ends_with(".tsv") {
            let delim = if clean_path.ends_with(".tsv") {
                b'\t'
            } else {
                b','
            };
            return parse_delimited_bytes(bytes, delim, requested_cols, min_score_filter);
        }

        let trimmed = bytes.trim_ascii_start();
        if clean_path.ends_with(".json") || trimmed.starts_with(b"[") {
            let parsed: serde_json::Value = serde_json::from_slice(trimmed).map_err(|e| {
                DataLoaderError::InvalidConfig(format!(
                    "Failed to parse JSON file '{path_hint}': {e}"
                ))
            })?;
            let rows_slice: Vec<serde_json::Value> = match parsed {
                serde_json::Value::Array(arr) => arr,
                serde_json::Value::Object(mut map) => {
                    let mut extracted_arr = None;
                    for &wrapper_key in JSON_ARRAY_WRAPPER_KEYS {
                        if let Some(serde_json::Value::Array(arr)) = map.remove(wrapper_key) {
                            extracted_arr = Some(arr);
                            break;
                        }
                    }
                    extracted_arr.unwrap_or_else(|| vec![serde_json::Value::Object(map)])
                }
                other => vec![other],
            };
            return Self::from_json_rows(&rows_slice, requested_cols, min_score_filter);
        }

        if clean_path.ends_with(".jsonl")
            || clean_path.ends_with(".ndjson")
            || trimmed.starts_with(b"{")
        {
            let text = std::str::from_utf8(bytes).map_err(|e| {
                DataLoaderError::InvalidConfig(format!(
                    "Invalid UTF-8 in JSONL source '{path_hint}': {e}"
                ))
            })?;
            let mut rows = Vec::new();
            for (line_idx, line) in text.lines().enumerate() {
                let l = line.trim();
                if l.is_empty() {
                    continue;
                }
                let val: serde_json::Value = serde_json::from_str(l).map_err(|e| {
                    DataLoaderError::InvalidConfig(format!(
                        "Malformed JSON on line {} of '{path_hint}': {e}",
                        line_idx + 1
                    ))
                })?;
                rows.push(val);
            }
            return Self::from_json_rows(&rows, requested_cols, min_score_filter);
        }

        Err(DataLoaderError::InvalidConfig(format!(
            "Unrecognized dataset shard format for '{path_hint}'. Supported formats: .parquet, .arrow/.feather, .jsonl, .json, .csv/.tsv, .tar (WebDataset), and .gz/.zst/.lz4 compressed streams."
        )))
    }

    pub fn read_batch(
        &self,
        shard_id: u32,
        batch_id: u64,
        start_coord: StreamCoordinate,
        batch_size: usize,
        mut dedup: Option<&mut TwoStageDeduplicator>,
    ) -> MultimodalBatch {
        let mut cursor = (start_coord.row_offset_in_page as usize).min(self.num_rows);
        let mut global_row = start_coord.global_row_idx;
        let mut collected: HashMap<String, Vec<MultimodalValue>> = self
            .column_names
            .iter()
            .map(|c| (c.clone(), Vec::with_capacity(batch_size)))
            .collect();
        let mut collected_rows = 0usize;
        let mut hash_buf = crate::simd_cpu::CacheAlignedBuffer::with_capacity(256);

        while cursor < self.num_rows && collected_rows < batch_size {
            let r_idx = cursor;
            cursor += 1;
            global_row += 1;

            let accepted = if let Some(d) = dedup.as_deref_mut() {
                hash_buf.data.clear();
                for c_name in &self.column_names {
                    if let Some(col_vals) = self.columns.get(c_name) {
                        col_vals[r_idx].update_hash_payload(&mut hash_buf.data);
                    }
                }
                d.try_stage(batch_id, &hash_buf.data).is_some()
            } else {
                true
            };

            if accepted {
                for c_name in &self.column_names {
                    if let (Some(src_vals), Some(dst_vals)) =
                        (self.columns.get(c_name), collected.get_mut(c_name))
                    {
                        dst_vals.push(src_vals[r_idx].clone());
                    }
                }
                collected_rows += 1;
            }
        }

        let is_eof = cursor >= self.num_rows;
        MultimodalBatch {
            batch_id,
            column_order: self.column_names.clone(),
            columns: collected,
            num_rows: collected_rows,
            next_coord: StreamCoordinate {
                shard_id,
                byte_offset: 0,
                row_group_idx: usize::from(is_eof) as u32,
                page_idx: 0,
                row_offset_in_page: cursor as u32,
                global_row_idx: global_row,
            },
            is_eof,
        }
    }
}

fn parse_csv_line(line: &str, delimiter: u8) -> Vec<String> {
    let delim_char = char::from(delimiter);
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut chars = line.chars().peekable();
    let mut in_quotes = false;

    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    current.push('"');
                    let _ = chars.next();
                } else {
                    in_quotes = false;
                }
            } else {
                current.push(ch);
            }
        } else if ch == '"' {
            in_quotes = true;
        } else if ch == delim_char {
            fields.push(std::mem::take(&mut current));
        } else {
            current.push(ch);
        }
    }
    fields.push(current);
    fields
}

fn parse_cell_value(cell: &str) -> MultimodalValue {
    let trimmed = cell.trim();
    if trimmed.is_empty() {
        return MultimodalValue::Null;
    }
    if trimmed.eq_ignore_ascii_case("true") {
        return MultimodalValue::Bool(true);
    }
    if trimmed.eq_ignore_ascii_case("false") {
        return MultimodalValue::Bool(false);
    }
    if let Ok(i) = trimmed.parse::<i64>() {
        return MultimodalValue::Int(i);
    }
    if let Ok(f) = trimmed.parse::<f64>() {
        return MultimodalValue::Float(f);
    }
    classify_string_or_json(cell)
}

fn parse_delimited_bytes(
    bytes: &[u8],
    delimiter: u8,
    requested_cols: &[String],
    min_score_filter: Option<(Option<String>, f64)>,
) -> Result<InMemoryMultimodalTable> {
    let text = std::str::from_utf8(bytes).map_err(|e| {
        DataLoaderError::InvalidConfig(format!("CSV/TSV stream is not valid UTF-8: {e}"))
    })?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let Some(header_line) = lines.next() else {
        return Ok(InMemoryMultimodalTable {
            column_names: requested_cols.to_vec(),
            columns: HashMap::new(),
            num_rows: 0,
        });
    };

    let headers: Vec<String> = parse_csv_line(header_line.trim_end_matches('\r'), delimiter)
        .into_iter()
        .map(|h| h.trim().to_owned())
        .collect();

    let selected_cols = if requested_cols.is_empty() {
        headers.clone()
    } else {
        for req in requested_cols {
            if !headers.iter().any(|h| h == req) {
                return Err(column_not_found_with_suggestions(req, &headers));
            }
        }
        requested_cols.to_vec()
    };

    let score_idx_and_min = if let Some((score_col_opt, min_score)) = min_score_filter {
        let sc = score_col_opt
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "score".to_owned());
        let idx = headers
            .iter()
            .position(|h| h == &sc)
            .ok_or_else(|| column_not_found_with_suggestions(&sc, &headers))?;
        Some((idx, min_score))
    } else {
        None
    };

    let col_positions: Vec<(String, usize)> = selected_cols
        .iter()
        .map(|c| {
            let pos = headers.iter().position(|h| h == c).unwrap();
            (c.clone(), pos)
        })
        .collect();

    let mut columns: HashMap<String, Vec<MultimodalValue>> = selected_cols
        .iter()
        .map(|c| (c.clone(), Vec::new()))
        .collect();
    let mut num_rows = 0usize;

    for raw_line in lines {
        let cells = parse_csv_line(raw_line.trim_end_matches('\r'), delimiter);
        if let Some((s_idx, min_s)) = score_idx_and_min {
            let pass = cells
                .get(s_idx)
                .and_then(|v| v.trim().parse::<f64>().ok())
                .is_some_and(|val| val >= min_s);
            if !pass {
                continue;
            }
        }
        for (c_name, pos) in &col_positions {
            let val = cells
                .get(*pos)
                .map(|s| parse_cell_value(s))
                .unwrap_or(MultimodalValue::Null);
            if let Some(target) = columns.get_mut(c_name) {
                target.push(val);
            }
        }
        num_rows += 1;
    }

    Ok(InMemoryMultimodalTable {
        column_names: selected_cols,
        columns,
        num_rows,
    })
}

fn parse_webdataset_tar_bytes(
    bytes: &[u8],
    requested_cols: &[String],
    min_score_filter: Option<(Option<String>, f64)>,
) -> Result<InMemoryMultimodalTable> {
    let mut offset = 0usize;
    let mut sample_keys_order: Vec<String> = Vec::new();
    let mut samples: HashMap<String, HashMap<String, MultimodalValue>> = HashMap::new();
    let mut discovered_cols: Vec<String> = Vec::new();

    while offset + TAR_USTAR_BLOCK_SIZE <= bytes.len() {
        let header = &bytes[offset..offset + TAR_USTAR_BLOCK_SIZE];
        offset += TAR_USTAR_BLOCK_SIZE;
        if header.iter().all(|&b| b == 0) {
            break;
        }
        let name_end = header[..100].iter().position(|&b| b == 0).unwrap_or(100);
        let raw_name = String::from_utf8_lossy(&header[..name_end])
            .trim()
            .to_owned();
        let size_str = String::from_utf8_lossy(&header[124..136])
            .trim_matches(|c: char| c == '\0' || c.is_ascii_whitespace())
            .to_owned();
        let file_size = usize::from_str_radix(&size_str, 8).unwrap_or(0);
        let typeflag = header[156];

        let content_end = (offset + file_size).min(bytes.len());
        let content = &bytes[offset..content_end];
        let padded_size = file_size.div_ceil(TAR_USTAR_BLOCK_SIZE) * TAR_USTAR_BLOCK_SIZE;
        offset = (offset + padded_size).min(bytes.len());

        if (typeflag == b'0' || typeflag == 0) && !raw_name.is_empty() && !raw_name.ends_with('/') {
            let base = raw_name.rsplit('/').next().unwrap_or(&raw_name);
            if let Some((sample_id, ext)) = base.split_once('.') {
                let col_name = ext.to_ascii_lowercase();
                if !discovered_cols.contains(&col_name) {
                    discovered_cols.push(col_name.clone());
                }
                if !samples.contains_key(sample_id) {
                    sample_keys_order.push(sample_id.to_owned());
                }
                let val = if WEBDATASET_TEXT_EXTENSIONS.contains(&col_name.as_str()) {
                    MultimodalValue::Text(String::from_utf8_lossy(content).trim().to_owned())
                } else if col_name == "json" {
                    MultimodalValue::Json(String::from_utf8_lossy(content).trim().to_owned())
                } else {
                    MultimodalValue::Bytes(content.to_vec())
                };
                samples
                    .entry(sample_id.to_owned())
                    .or_default()
                    .insert(col_name, val);
            }
        }
    }

    let selected_cols = if requested_cols.is_empty() {
        discovered_cols.clone()
    } else {
        for req in requested_cols {
            if !discovered_cols.is_empty() && !discovered_cols.iter().any(|c| c == req) {
                return Err(column_not_found_with_suggestions(req, &discovered_cols));
            }
        }
        requested_cols.to_vec()
    };

    let mut columns: HashMap<String, Vec<MultimodalValue>> = selected_cols
        .iter()
        .map(|c| (c.clone(), Vec::new()))
        .collect();
    let mut num_rows = 0usize;

    for key in &sample_keys_order {
        let Some(row_map) = samples.get(key) else {
            continue;
        };
        if let Some((ref score_col_opt, min_s)) = min_score_filter {
            let sc = score_col_opt.as_deref().unwrap_or("score");
            let pass = row_map
                .get(sc)
                .and_then(MultimodalValue::as_f64)
                .is_some_and(|v| v >= min_s);
            if !pass {
                continue;
            }
        }
        for c_name in &selected_cols {
            let v = row_map
                .get(c_name)
                .cloned()
                .unwrap_or(MultimodalValue::Null);
            if let Some(target) = columns.get_mut(c_name) {
                target.push(v);
            }
        }
        num_rows += 1;
    }

    Ok(InMemoryMultimodalTable {
        column_names: selected_cols,
        columns,
        num_rows,
    })
}
