use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

use thiserror::Error;

use crate::constants::{
    HF_SCHEME_TYPOS, HTTP_SCHEME_TYPOS, MAGIC_ARROW_IPC, MAGIC_GIT_LFS_V1,
    MAX_COLUMN_PREVIEW_COUNT, MAX_SIBLING_SCAN_ENTRIES, MAX_TYPO_SUGGESTIONS,
    URI_HF_DATASETS_PREFIX,
};

#[derive(Debug, Error)]
pub enum DataLoaderError {
    #[error("I/O failure: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP failure: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Parquet failure: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error("Arrow failure: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),

    #[error("Corrupted metadata checkpoint in both slots")]
    CorruptedMetadata,

    #[error("Unknown staged batch id: {0}")]
    UnknownBatchId(u64),

    #[error("Invalid coordinate buffer length")]
    InvalidCoordinate,

    #[error("Target column not found: {0}")]
    ColumnNotFound(String),

    #[error("Invalid source configuration: {0}")]
    InvalidConfig(String),
}

pub type Result<T> = std::result::Result<T, DataLoaderError>;

pub fn damerau_levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let len_a = a_chars.len();
    let len_b = b_chars.len();

    if len_a == 0 {
        return len_b;
    }
    if len_b == 0 {
        return len_a;
    }

    let width = len_b + 1;
    let mut matrix = vec![0usize; (len_a + 1) * width];

    for i in 0..=len_a {
        matrix[i * width] = i;
    }
    for (j, cell) in matrix.iter_mut().enumerate().take(len_b + 1) {
        *cell = j;
    }

    for i in 1..=len_a {
        for j in 1..=len_b {
            let cost = usize::from(a_chars[i - 1] != b_chars[j - 1]);
            let deletion = matrix[(i - 1) * width + j] + 1;
            let insertion = matrix[i * width + (j - 1)] + 1;
            let substitution = matrix[(i - 1) * width + (j - 1)] + cost;
            let mut best = deletion.min(insertion).min(substitution);

            if i > 1
                && j > 1
                && a_chars[i - 1] == b_chars[j - 2]
                && a_chars[i - 2] == b_chars[j - 1]
            {
                let transposition = matrix[(i - 2) * width + (j - 2)] + 1;
                best = best.min(transposition);
            }
            matrix[i * width + j] = best;
        }
    }

    matrix[len_a * width + len_b]
}

pub fn find_closest_matches<S: AsRef<str>>(
    target: &str,
    candidates: &[S],
    max_suggestions: usize,
) -> Vec<String> {
    let target_trim = target.trim();
    if target_trim.is_empty() || candidates.is_empty() || max_suggestions == 0 {
        return Vec::new();
    }
    let target_lower = target_trim.to_ascii_lowercase();

    let mut scored: Vec<(usize, usize, String)> = Vec::new();
    for cand in candidates {
        let c_str = cand.as_ref();
        if c_str.is_empty() {
            continue;
        }
        let c_lower = c_str.to_ascii_lowercase();
        if c_lower == target_lower {
            scored.push((0, 0, c_str.to_owned()));
            continue;
        }
        let dist = damerau_levenshtein(&target_lower, &c_lower);
        let max_len = target_lower.chars().count().max(c_lower.chars().count());
        let threshold = match max_len {
            0..=3 => 1,
            4..=6 => 2,
            7..=11 => 3,
            _ => (max_len * 2) / 5,
        };
        let is_sub = c_lower.contains(&target_lower) || target_lower.contains(&c_lower);
        if dist <= threshold || (target_lower.len() >= 3 && is_sub) {
            let rank_bonus = if is_sub { 0 } else { 1 };
            scored.push((dist, rank_bonus, c_str.to_owned()));
        }
    }

    scored.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    scored.dedup_by(|a, b| a.2 == b.2);
    scored
        .into_iter()
        .take(max_suggestions)
        .map(|(_, _, s)| s)
        .collect()
}

pub fn format_did_you_mean<S: AsRef<str>>(target: &str, candidates: &[S]) -> String {
    let matches = find_closest_matches(target, candidates, MAX_TYPO_SUGGESTIONS);
    if matches.is_empty() {
        return String::new();
    }
    if matches.len() == 1 {
        format!(" Did you mean '{}'?", matches[0])
    } else {
        let quoted: Vec<String> = matches.iter().map(|m| format!("'{m}'")).collect();
        format!(" Did you mean one of [{}]?", quoted.join(", "))
    }
}

pub fn column_not_found_with_suggestions<S: AsRef<str>>(
    requested: &str,
    available: &[S],
) -> DataLoaderError {
    let suggestion = format_did_you_mean(requested, available);
    let avail_preview: Vec<String> = available
        .iter()
        .take(MAX_COLUMN_PREVIEW_COUNT)
        .map(|s| format!("'{}'", s.as_ref()))
        .collect();
    let avail_suffix = if available.len() > MAX_COLUMN_PREVIEW_COUNT {
        format!(
            " (and {} more)",
            available.len().saturating_sub(avail_preview.len())
        )
    } else {
        String::new()
    };

    if avail_preview.is_empty() {
        DataLoaderError::ColumnNotFound(requested.to_owned())
    } else {
        DataLoaderError::ColumnNotFound(format!(
            "'{requested}'.{suggestion} Available columns: [{}]{avail_suffix}",
            avail_preview.join(", ")
        ))
    }
}

pub fn diagnose_uri_typo(uri: &str) -> Option<String> {
    let trimmed = uri.trim();
    let lower = trimmed.to_ascii_lowercase();

    if let Some(rest) = trimmed
        .strip_prefix("hf://dataset/")
        .or_else(|| trimmed.strip_prefix("HF://dataset/"))
    {
        return Some(format!(
            "Invalid HuggingFace URI '{trimmed}': Did you mean '{URI_HF_DATASETS_PREFIX}{rest}'? (Note plural 'datasets')"
        ));
    }

    if let Some(rest) = trimmed
        .strip_prefix("hf:/datasets/")
        .or_else(|| trimmed.strip_prefix("hf:///datasets/"))
    {
        return Some(format!(
            "Malformed HuggingFace URI '{trimmed}': Did you mean '{URI_HF_DATASETS_PREFIX}{rest}'?"
        ));
    }

    for &bad_prefix in HF_SCHEME_TYPOS {
        if lower.starts_with(bad_prefix) {
            let after = &trimmed[bad_prefix.len()..].trim_start_matches("datasets/");
            return Some(format!(
                "Unsupported URI scheme in '{trimmed}': Did you mean '{URI_HF_DATASETS_PREFIX}{after}'?"
            ));
        }
    }

    for &(typo, fixed) in HTTP_SCHEME_TYPOS {
        if lower.starts_with(typo) && !lower.starts_with(fixed) {
            let rest = trimmed[typo.len()..].trim_start_matches('/');
            return Some(format!(
                "Malformed HTTP(S) URI '{trimmed}': Did you mean '{fixed}{rest}'?"
            ));
        }
    }

    None
}

pub fn open_local_file_with_diagnostics(path: impl AsRef<Path>) -> Result<File> {
    let p = path.as_ref();
    match File::open(p) {
        Ok(f) => {
            if let Ok(meta) = f.metadata()
                && meta.is_dir()
            {
                return Err(DataLoaderError::InvalidConfig(format!(
                    "Path '{}' is a directory, not a dataset shard file. Pass specific shard files or use lm.stream() directory discovery.",
                    p.display()
                )));
            }
            Ok(f)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let path_str = p.to_string_lossy();
            if let Some(uri_hint) = diagnose_uri_typo(&path_str) {
                return Err(DataLoaderError::InvalidConfig(uri_hint));
            }

            let mut sibling_names = Vec::new();
            let parent = p
                .parent()
                .filter(|par| !par.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."));
            if let Ok(entries) = fs::read_dir(parent) {
                for entry in entries.flatten().take(MAX_SIBLING_SCAN_ENTRIES) {
                    if let Some(name) = entry.file_name().to_str() {
                        sibling_names.push(name.to_owned());
                    }
                }
            }

            let file_name = p
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(path_str.as_ref());
            let suggestion = format_did_you_mean(file_name, &sibling_names);
            Err(DataLoaderError::InvalidConfig(format!(
                "Local dataset file not found: '{}'.{}",
                p.display(),
                suggestion
            )))
        }
        Err(e) => Err(DataLoaderError::Io(e)),
    }
}

pub fn inspect_magic_bytes_hint(
    header: &[u8],
    expected_format: &str,
    source_label: &str,
) -> Option<String> {
    let trimmed = header.trim_ascii_start();
    if trimmed.starts_with(MAGIC_GIT_LFS_V1) {
        return Some(format!(
            "Source '{source_label}' is an un-fetched Git LFS text pointer, not a binary {expected_format} file. Run 'git lfs pull' or stream directly via 'hf://datasets/...'."
        ));
    }
    if trimmed.starts_with(b"<!DOCTYPE html")
        || trimmed.starts_with(b"<!doctype html")
        || trimmed.starts_with(b"<html")
    {
        return Some(format!(
            "Source '{source_label}' contains an HTML webpage instead of {expected_format} data. Verify that you are using a direct download URL ('/resolve/main/...') rather than a web viewer URL ('/blob/main/...'), and check HF_TOKEN permissions."
        ));
    }
    if expected_format.eq_ignore_ascii_case("Parquet") {
        if header.starts_with(MAGIC_ARROW_IPC) {
            return Some(format!(
                "Source '{source_label}' has Apache Arrow IPC/Feather magic bytes ('ARROW1'), not Parquet ('PAR1'). Rename the file to '.arrow'/'.feather' or use a universal loader."
            ));
        }
        if trimmed.starts_with(b"{") || trimmed.starts_with(b"[") {
            return Some(format!(
                "Source '{source_label}' starts with JSON/JSONL text ('{}'), not Parquet binary magic ('PAR1'). Did you name a JSONL/JSON file with a '.parquet' extension?",
                char::from(trimmed[0])
            ));
        }
    }
    None
}

pub fn diagnose_parquet_file_error(
    path: &Path,
    err: parquet::errors::ParquetError,
) -> DataLoaderError {
    if let Ok(mut f) = File::open(path) {
        if let Ok(meta) = f.metadata()
            && meta.len() == 0
        {
            return DataLoaderError::InvalidConfig(format!(
                "Parquet file '{}' is empty (0 bytes).",
                path.display()
            ));
        }
        let mut buf = [0u8; 128];
        if let Ok(n) = f.read(&mut buf)
            && let Some(hint) =
                inspect_magic_bytes_hint(&buf[..n], "Parquet", &path.display().to_string())
        {
            return DataLoaderError::InvalidConfig(hint);
        }
    }
    DataLoaderError::Parquet(err)
}
