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

pub fn extract_json_value_by_path<'a>(root: &'a Value, key_path: &str) -> Option<&'a Value> {
    if let Some(direct) = root.get(key_path) {
        return Some(direct);
    }
    if !key_path.contains('.') {
        return None;
    }
    let mut current = root;
    for segment in key_path.split('.') {
        if segment.is_empty() {
            continue;
        }
        match current {
            Value::Object(map) => {
                current = map.get(segment)?;
            }
            Value::Array(arr) => {
                let idx: isize = segment.parse().ok()?;
                let resolved = if idx < 0 {
                    let abs = idx.unsigned_abs();
                    if abs > arr.len() {
                        return None;
                    }
                    arr.len() - abs
                } else {
                    idx as usize
                };
                current = arr.get(resolved)?;
            }
            _ => return None,
        }
    }
    Some(current)
}

fn json_value_to_extracted_string(val: &Value) -> Option<String> {
    match val {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        Value::Bool(b) => Some(b.to_string()),
        Value::Number(n) => Some(n.to_string()),
        Value::Array(_) | Value::Object(_) => Some(val.to_string()),
    }
}

pub fn collect_json_object_keys(raw_line: &str) -> Option<Vec<String>> {
    let parsed: Value = serde_json::from_str(raw_line).ok()?;
    let obj = parsed.as_object()?;
    Some(obj.keys().cloned().collect())
}

pub fn process_single_record(raw: &str, cfg: &PostProcessConfig) -> Option<String> {
    let mut text = if let Some(key) = &cfg.json_extract_key {
        let parsed: Value = serde_json::from_str(raw).ok()?;
        let val = extract_json_value_by_path(&parsed, key)?;
        json_value_to_extracted_string(val)?
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
