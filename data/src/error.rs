use thiserror::Error;

use crate::constants::{MAX_TYPO_SUGGESTIONS, MAX_COLUMN_PREVIEW_COUNT};

#[derive(Debug, Error)]
pub enum DataLoaderError {
    #[error("I/O failure: {0}")]
    Io(#[from] std::io::Error),

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

    #[error("Invalid configuration: {0}")]
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
