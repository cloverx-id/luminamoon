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

pub fn calculate_shannon_entropy(text: &str) -> f64 {
    let char_count = text.chars().count();
    if char_count == 0 {
        return 0.0;
    }

    // Zero-allocation stack array fast-path for pure ASCII text
    if text.is_ascii() {
        let mut freq = [0u32; 256];
        for &b in text.as_bytes() {
            freq[b as usize] += 1;
        }

        let n = char_count as f64;
        let mut entropy = 0.0;
        for &count in &freq {
            if count > 0 {
                let p = (count as f64) / n;
                entropy -= p * p.log2();
            }
        }
        return entropy;
    }

    let mut freq = HashMap::new();
    for c in text.chars() {
        *freq.entry(c).or_insert(0usize) += 1;
    }

    let n = char_count as f64;
    let mut entropy = 0.0;
    for &count in freq.values() {
        let p = (count as f64) / n;
        entropy -= p * p.log2();
    }
    entropy
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

pub fn evaluate_quality(text: &str, cfg: &QualityConfig) -> QualityMetrics {
    let char_count = text.chars().count();
    if cfg.min_chars > 0 && char_count < cfg.min_chars {
        return QualityMetrics {
            char_count,
            word_count: 0,
            entropy: 0.0,
            repetition_ratio: 0.0,
            symbol_ratio: 0.0,
            is_passed: false,
        };
    }

    if cfg.max_chars > 0 && char_count > cfg.max_chars {
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

    let entropy = calculate_shannon_entropy(text);
    let repetition_ratio = calculate_repetition_ratio(&words);
    let (_, symbol_ratio) = calculate_symbol_ratio(text);

    let is_passed = (cfg.min_entropy <= 0.0 || entropy >= cfg.min_entropy)
        && (cfg.max_repetition_ratio >= 1.0 || repetition_ratio <= cfg.max_repetition_ratio)
        && (cfg.max_symbol_ratio >= 1.0 || symbol_ratio <= cfg.max_symbol_ratio);

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
