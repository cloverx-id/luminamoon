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
            min_byte_len: 0,
            max_byte_len: 0,
            truncate_over_max: false,
            json_extract_key: None,
        }
    }
}

fn resolve_array_segment<'a>(arr: &'a [Value], segment: &str) -> Option<&'a Value> {
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
    arr.get(resolved)
}

pub fn extract_json_value_by_path<'a>(root: &'a Value, key_path: &str) -> Option<&'a Value> {
    if key_path.is_empty() {
        return None;
    }
    match root {
        Value::Object(map) => {
            if let Some(direct) = map.get(key_path) {
                return Some(direct);
            }
        }
        Value::Array(arr) => {
            if !key_path.contains('.') {
                return resolve_array_segment(arr, key_path);
            }
        }
        _ => return None,
    }

    if !key_path.contains('.') {
        return None;
    }

    let mut current = root;
    for segment in key_path.split('.') {
        match current {
            Value::Object(map) => {
                current = map.get(segment)?;
            }
            Value::Array(arr) => {
                current = resolve_array_segment(arr, segment)?;
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

fn normalize_newlines_single_pass(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = String::with_capacity(bytes.len());
    let mut i = 0usize;
    let mut last_copied = 0usize;

    while i < bytes.len() {
        if bytes[i] == b'\r' {
            out.push_str(&input[last_copied..i]);
            out.push('\n');
            if i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
                i += 2;
            } else {
                i += 1;
            }
            last_copied = i;
        } else {
            i += 1;
        }
    }
    out.push_str(&input[last_copied..]);
    out
}

pub fn process_single_record(raw: &str, cfg: &PostProcessConfig) -> Option<String> {
    let extracted: Option<String>;
    let mut working: &str = if let Some(key) = &cfg.json_extract_key {
        let parsed: Value = serde_json::from_str(raw).ok()?;
        let val = extract_json_value_by_path(&parsed, key)?;
        extracted = Some(json_value_to_extracted_string(val)?);
        extracted.as_deref().unwrap()
    } else {
        raw
    };

    let normalized_storage: Option<String>;
    if cfg.normalize_newlines && crate::simd_cpu::contains_byte_fast(working.as_bytes(), b'\r') {
        normalized_storage = Some(normalize_newlines_single_pass(working));
        working = normalized_storage.as_deref().unwrap();
    }

    if cfg.trim_whitespace {
        working = working.trim();
    }

    if cfg.min_byte_len > 0 && working.len() < cfg.min_byte_len {
        return None;
    }

    if cfg.max_byte_len > 0 && working.len() > cfg.max_byte_len {
        if cfg.truncate_over_max {
            let mut cut = cfg.max_byte_len;
            while cut > 0 && !working.is_char_boundary(cut) {
                cut -= 1;
            }
            working = &working[..cut];
            if cfg.trim_whitespace {
                working = working.trim();
            }
            if cfg.min_byte_len > 0 && working.len() < cfg.min_byte_len {
                return None;
            }
        } else {
            return None;
        }
    }

    Some(working.to_owned())
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
