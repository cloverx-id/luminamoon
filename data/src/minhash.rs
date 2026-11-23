use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use xxhash_rust::xxh3::{xxh3_64, xxh3_128};

pub const DEFAULT_MINHASH_PERMUTATIONS: usize = 64;

#[inline]
fn mix64(mut z: u64) -> u64 {
    z ^= z >> 33;
    z = z.wrapping_mul(0xff51_afd7_ed55_8ccd);
    z ^= z >> 33;
    z = z.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    z ^= z >> 33;
    z
}

#[inline]
fn update_mins_from_shingle(bytes: &[u8], mins: &mut [u64]) {
    // Compute a single 128-bit XXH3 hash per shingle and derive all k universal permutations
    // via Kirsch-Mitzenmacher double hashing + SplitMix finalizer (20-40x faster than k full hashes).
    let h128 = xxh3_128(bytes);
    let h1 = h128 as u64;
    let h2 = ((h128 >> 64) as u64) | 1;
    let mut acc = h1;
    for min_val in mins.iter_mut() {
        let perm = mix64(acc);
        if perm < *min_val {
            *min_val = perm;
        }
        acc = acc.wrapping_add(h2);
    }
}

#[derive(Debug, Clone)]
pub struct MinHashSignature {
    pub values: Vec<u64>,
    pub is_empty_doc: bool,
}

impl MinHashSignature {
    pub fn compute_char_shingles(text: &str, k: usize, num_perm: usize) -> Self {
        let mut mins = vec![u64::MAX; num_perm];
        if text.is_empty() {
            return Self {
                values: mins,
                is_empty_doc: true,
            };
        }

        if text.is_ascii() {
            let raw = text.as_bytes();
            let effective_k = k.min(raw.len()).max(1);
            for bytes in raw.windows(effective_k) {
                update_mins_from_shingle(bytes, &mut mins);
            }
            return Self {
                values: mins,
                is_empty_doc: false,
            };
        }

        // Sliding window over char boundaries without allocating a Vec<usize> for the whole document.
        let mut start_iter = text.char_indices().map(|(idx, _)| idx);
        let mut end_iter = text.char_indices().map(|(idx, _)| idx);
        let effective_k = k.max(1);
        for _ in 0..effective_k {
            if end_iter.next().is_none() {
                update_mins_from_shingle(text.as_bytes(), &mut mins);
                return Self {
                    values: mins,
                    is_empty_doc: false,
                };
            }
        }

        let raw = text.as_bytes();
        let Some(mut start_b) = start_iter.next() else {
            return Self {
                values: mins,
                is_empty_doc: true,
            };
        };
        for end_b in end_iter {
            update_mins_from_shingle(&raw[start_b..end_b], &mut mins);
            if let Some(next_start) = start_iter.next() {
                start_b = next_start;
            }
        }
        update_mins_from_shingle(&raw[start_b..], &mut mins);

        Self {
            values: mins,
            is_empty_doc: false,
        }
    }

    pub fn jaccard_similarity(&self, other: &Self) -> f64 {
        if self.is_empty_doc || other.is_empty_doc {
            return if self.is_empty_doc && other.is_empty_doc {
                1.0
            } else {
                0.0
            };
        }
        if self.values.is_empty() || self.values.len() != other.values.len() {
            return 0.0;
        }

        let matches = self
            .values
            .iter()
            .zip(other.values.iter())
            .filter(|(a, b)| a == b)
            .count();

        (matches as f64) / (self.values.len() as f64)
    }
}

/// Selects `(num_bands, rows_per_band)` minimizing `|(1/b)^(1/r) - threshold|` so the
/// LSH S-curve inflection point aligns with the requested Jaccard threshold.
pub fn choose_optimal_lsh_params(num_perm: usize, threshold: f64) -> (usize, usize) {
    let t = threshold.clamp(0.05, 0.99);
    let mut best_b = 1usize;
    let mut best_r = num_perm.max(1);
    let mut best_err = f64::INFINITY;

    for b in 1..=num_perm {
        let r = num_perm / b;
        if r == 0 {
            continue;
        }
        let inflection = (1.0 / (b as f64)).powf(1.0 / (r as f64));
        let err = (inflection - t).abs();
        // Prefer slightly higher coverage (b * r closer to num_perm) on ties.
        if err + 1e-9 < best_err || ((err - best_err).abs() <= 1e-9 && b * r > best_b * best_r) {
            best_err = err;
            best_b = b;
            best_r = r;
        }
    }
    (best_b, best_r)
}

pub struct MinHashDeduplicator {
    threshold: f64,
    k_shingle: usize,
    num_perm: usize,
    num_bands: usize,
    rows_per_band: usize,
    signatures: Vec<MinHashSignature>,
    lsh_buckets: HashMap<(usize, u64), Vec<usize>>,
}

impl MinHashDeduplicator {
    pub fn new(threshold: f64, k_shingle: usize, num_perm: usize) -> Self {
        let safe_threshold = if threshold.is_nan() {
            0.85
        } else {
            threshold.clamp(0.0, 1.0)
        };
        let n_perm = num_perm.max(8);
        let (num_bands, rows_per_band) = choose_optimal_lsh_params(n_perm, safe_threshold);

        Self {
            threshold: safe_threshold,
            k_shingle: k_shingle.max(1),
            num_perm: n_perm,
            num_bands,
            rows_per_band,
            signatures: Vec::new(),
            lsh_buckets: HashMap::new(),
        }
    }

    pub fn num_bands(&self) -> usize {
        self.num_bands
    }

    pub fn rows_per_band(&self) -> usize {
        self.rows_per_band
    }

    pub fn compute_signature(&self, text: &str) -> MinHashSignature {
        MinHashSignature::compute_char_shingles(text, self.k_shingle, self.num_perm)
    }

    fn compute_band_hashes(&self, sig: &MinHashSignature) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.num_bands);
        let mut stack_buf = [0u8; 512];
        for b in 0..self.num_bands {
            let start = b * self.rows_per_band;
            let end = (start + self.rows_per_band).min(sig.values.len());
            let slice = &sig.values[start..end];
            let byte_len = slice.len() * 8;
            if byte_len <= stack_buf.len() {
                for (j, &v) in slice.iter().enumerate() {
                    stack_buf[j * 8..(j + 1) * 8].copy_from_slice(&v.to_le_bytes());
                }
                out.push(xxh3_64(&stack_buf[..byte_len]));
            } else {
                let mut heap_buf = Vec::with_capacity(byte_len);
                for &v in slice {
                    heap_buf.extend_from_slice(&v.to_le_bytes());
                }
                out.push(xxh3_64(&heap_buf));
            }
        }
        out
    }

    fn max_similarity_with_bands(&self, sig: &MinHashSignature, band_hashes: &[u64]) -> f64 {
        if self.signatures.is_empty() {
            return 0.0;
        }

        if self.signatures.len() <= 128 {
            let mut max_sim = 0.0f64;
            for existing in &self.signatures {
                let sim = sig.jaccard_similarity(existing);
                if sim > max_sim {
                    max_sim = sim;
                    if max_sim >= self.threshold {
                        return max_sim;
                    }
                }
            }
            return max_sim;
        }

        let mut candidates = HashSet::new();
        for (b, &bh) in band_hashes.iter().enumerate() {
            if let Some(doc_ids) = self.lsh_buckets.get(&(b, bh)) {
                for &id in doc_ids {
                    candidates.insert(id);
                }
            }
        }

        let mut max_sim = 0.0f64;
        for &cand_id in &candidates {
            if let Some(existing) = self.signatures.get(cand_id) {
                let sim = sig.jaccard_similarity(existing);
                if sim > max_sim {
                    max_sim = sim;
                    if max_sim >= self.threshold {
                        return max_sim;
                    }
                }
            }
        }
        max_sim
    }

    pub fn max_similarity(&self, sig: &MinHashSignature) -> f64 {
        let band_hashes = self.compute_band_hashes(sig);
        self.max_similarity_with_bands(sig, &band_hashes)
    }

    fn insert_if_unique(&mut self, sig: MinHashSignature) -> bool {
        let band_hashes = self.compute_band_hashes(&sig);
        if !self.signatures.is_empty() {
            let max_sim = self.max_similarity_with_bands(&sig, &band_hashes);
            if max_sim >= self.threshold {
                return false;
            }
        }

        let doc_id = self.signatures.len();
        for (b, bh) in band_hashes.into_iter().enumerate() {
            self.lsh_buckets.entry((b, bh)).or_default().push(doc_id);
        }
        self.signatures.push(sig);
        true
    }

    pub fn is_unique(&mut self, text: &str) -> bool {
        let sig = self.compute_signature(text);
        self.insert_if_unique(sig)
    }

    /// Computes MinHash signatures in parallel with Rayon, then filters sequentially through LSH.
    pub fn filter_batch(&mut self, texts: &[String]) -> Vec<String> {
        let k = self.k_shingle;
        let num_perm = self.num_perm;
        let sigs: Vec<MinHashSignature> = texts
            .par_iter()
            .map(|t| MinHashSignature::compute_char_shingles(t, k, num_perm))
            .collect();

        let mut out = Vec::with_capacity(texts.len());
        for (text, sig) in texts.iter().zip(sigs) {
            if self.insert_if_unique(sig) {
                out.push(text.clone());
            }
        }
        out
    }

    pub fn count(&self) -> usize {
        self.signatures.len()
    }

    pub fn clear(&mut self) {
        self.signatures.clear();
        self.lsh_buckets.clear();
    }
}
