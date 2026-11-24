use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct QualityConfig {
    pub min_chars: usize,
    pub max_chars: usize,
    pub min_words: usize,
    pub min_entropy: f64,
    pub max_repetition_ratio: f64,
    pub max_symbol_ratio: f64,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            min_chars: 0,
            max_chars: 0,
            min_words: 0,
            min_entropy: 0.0,
            max_repetition_ratio: 1.0,
            max_symbol_ratio: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityMetrics {
    pub char_count: usize,
    pub word_count: usize,
    pub entropy: f64,
    pub repetition_ratio: f64,
    pub symbol_ratio: f64,
    pub is_passed: bool,
}

#[inline]
fn passes_bounds(
    char_count: usize,
    word_count: usize,
    entropy: f64,
    repetition_ratio: f64,
    symbol_ratio: f64,
    cfg: &QualityConfig,
) -> bool {
    let min_ent_ok =
        cfg.min_entropy.is_nan() || cfg.min_entropy <= 0.0 || entropy >= cfg.min_entropy;
    let max_rep_ok = cfg.max_repetition_ratio.is_nan()
        || cfg.max_repetition_ratio >= 1.0
        || repetition_ratio <= cfg.max_repetition_ratio;
    let max_sym_ok = cfg.max_symbol_ratio.is_nan()
        || cfg.max_symbol_ratio >= 1.0
        || symbol_ratio <= cfg.max_symbol_ratio;

    (cfg.min_chars == 0 || char_count >= cfg.min_chars)
        && (cfg.max_chars == 0 || char_count <= cfg.max_chars)
        && (cfg.min_words == 0 || word_count >= cfg.min_words)
        && min_ent_ok
        && max_rep_ok
        && max_sym_ok
}

/// Computes `(char_count, entropy, symbol_ratio)` in a single pass over ASCII or UTF-8 text.
fn single_pass_char_stats(text: &str) -> (usize, f64, f64) {
    if text.is_empty() {
        return (0, 0.0, 0.0);
    }

    if text.is_ascii() {
        let mut freq = [0u64; 256];
        let mut visible = 0u64;
        let mut symbols = 0u64;
        for &b in text.as_bytes() {
            freq[b as usize] += 1;
            if !b.is_ascii_whitespace() {
                visible += 1;
                if !b.is_ascii_alphanumeric() {
                    symbols += 1;
                }
            }
        }
        let n = text.len() as f64;
        let mut entropy = 0.0;
        for &count in &freq {
            if count > 0 {
                let p = (count as f64) / n;
                entropy -= p * p.log2();
            }
        }
        let symbol_ratio = if visible == 0 {
            0.0
        } else {
            (symbols as f64) / (visible as f64)
        };
        return (text.len(), entropy, symbol_ratio);
    }

    let mut char_count = 0usize;
    let mut visible = 0u64;
    let mut symbols = 0u64;
    let mut freq: HashMap<char, u64> = HashMap::new();
    for c in text.chars() {
        char_count += 1;
        *freq.entry(c).or_insert(0) += 1;
        if !c.is_whitespace() {
            visible += 1;
            if !c.is_alphanumeric() {
                symbols += 1;
            }
        }
    }
    let n = char_count as f64;
    let mut entropy = 0.0;
    for &count in freq.values() {
        let p = (count as f64) / n;
        entropy -= p * p.log2();
    }
    let symbol_ratio = if visible == 0 {
        0.0
    } else {
        (symbols as f64) / (visible as f64)
    };
    (char_count, entropy, symbol_ratio)
}

pub fn calculate_shannon_entropy(text: &str) -> f64 {
    single_pass_char_stats(text).1
}

pub fn calculate_repetition_ratio(words: &[&str]) -> f64 {
    if words.len() < 4 {
        return 0.0;
    }

    let total_trigrams = words.len() - 2;
    let mut unique_trigrams = HashSet::with_capacity(total_trigrams);

    for w in words.windows(3) {
        unique_trigrams.insert((w[0], w[1], w[2]));
    }

    1.0 - ((unique_trigrams.len() as f64) / (total_trigrams as f64))
}

pub fn calculate_symbol_ratio(text: &str) -> (usize, f64) {
    let mut symbols = 0usize;
    let mut visible = 0usize;
    for c in text.chars() {
        if !c.is_whitespace() {
            visible += 1;
            if !c.is_alphanumeric() {
                symbols += 1;
            }
        }
    }
    if visible == 0 {
        return (0, 0.0);
    }
    (visible, (symbols as f64) / (visible as f64))
}

/// Computes all quality metrics without short-circuiting so callers inspect true values.
pub fn compute_quality_metrics(text: &str, cfg: &QualityConfig) -> QualityMetrics {
    let (char_count, entropy, symbol_ratio) = single_pass_char_stats(text);
    let words: Vec<&str> = text.split_whitespace().collect();
    let word_count = words.len();
    let repetition_ratio = calculate_repetition_ratio(&words);
    let is_passed = passes_bounds(
        char_count,
        word_count,
        entropy,
        repetition_ratio,
        symbol_ratio,
        cfg,
    );

    QualityMetrics {
        char_count,
        word_count,
        entropy,
        repetition_ratio,
        symbol_ratio,
        is_passed,
    }
}

/// Fast short-circuiting filter evaluation for high-throughput batch filtering.
pub fn evaluate_quality(text: &str, cfg: &QualityConfig) -> QualityMetrics {
    let char_count = if text.is_ascii() {
        text.len()
    } else {
        text.chars().count()
    };
    if (cfg.min_chars > 0 && char_count < cfg.min_chars)
        || (cfg.max_chars > 0 && char_count > cfg.max_chars)
    {
        return QualityMetrics {
            char_count,
            word_count: 0,
            entropy: 0.0,
            repetition_ratio: 0.0,
            symbol_ratio: 0.0,
            is_passed: false,
        };
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    let word_count = words.len();
    if cfg.min_words > 0 && word_count < cfg.min_words {
        return QualityMetrics {
            char_count,
            word_count,
            entropy: 0.0,
            repetition_ratio: 0.0,
            symbol_ratio: 0.0,
            is_passed: false,
        };
    }

    let (_, entropy, symbol_ratio) = single_pass_char_stats(text);
    let repetition_ratio = calculate_repetition_ratio(&words);
    let is_passed = passes_bounds(
        char_count,
        word_count,
        entropy,
        repetition_ratio,
        symbol_ratio,
        cfg,
    );

    QualityMetrics {
        char_count,
        word_count,
        entropy,
        repetition_ratio,
        symbol_ratio,
        is_passed,
    }
}

pub fn filter_quality_batch_parallel(texts: &[String], cfg: &QualityConfig) -> Vec<bool> {
    texts
        .par_iter()
        .map(|t| evaluate_quality(t, cfg).is_passed)
        .collect()
}
