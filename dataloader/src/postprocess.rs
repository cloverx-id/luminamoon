use rayon::prelude::*;
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct PostProcessConfig {
    pub trim_whitespace: bool,
    pub normalize_newlines: bool,
    pub min_byte_len: usize,
    pub max_byte_len: usize,
    pub truncate_over_max: bool,
    pub json_extract_key: Option<String>,
}

impl Default for PostProcessConfig {
    fn default() -> Self {
        Self {
            trim_whitespace: true,
            normalize_newlines: true,
            min_byte_len: 1,
            max_byte_len: 0,
            truncate_over_max: false,
            json_extract_key: None,
        }
    }
}

pub fn process_single_record(raw: &str, cfg: &PostProcessConfig) -> Option<String> {
    let mut text = if let Some(key) = &cfg.json_extract_key {
        let parsed: Value = serde_json::from_str(raw).ok()?;
        parsed.get(key)?.as_str()?.to_owned()
    } else {
        raw.to_owned()
    };

    if cfg.normalize_newlines && crate::simd_cpu::contains_byte_fast(text.as_bytes(), b'\r') {
        text = text.replace("\r\n", "\n").replace('\r', "\n");
    }

    if cfg.trim_whitespace {
        let trimmed = text.trim();
        if trimmed.len() != text.len() {
            text = trimmed.to_owned();
        }
    }

    if text.len() < cfg.min_byte_len {
        return None;
    }

    if cfg.max_byte_len > 0 && text.len() > cfg.max_byte_len {
        if cfg.truncate_over_max {
            let mut cut = cfg.max_byte_len;
            while cut > 0 && !text.is_char_boundary(cut) {
                cut -= 1;
            }
            text.truncate(cut);
            if text.len() < cfg.min_byte_len {
                return None;
            }
        } else {
            return None;
        }
    }

    Some(text)
}

pub fn process_batch_parallel(raw_records: &[String], cfg: &PostProcessConfig) -> Vec<String> {
    raw_records
        .par_iter()
        .filter_map(|item| process_single_record(item, cfg))
        .collect()
}

pub fn process_batch_parallel_indexed(
    raw_records: &[String],
    cfg: &PostProcessConfig,
) -> Vec<Option<String>> {
    raw_records
        .par_iter()
        .map(|item| process_single_record(item, cfg))
        .collect()
}
