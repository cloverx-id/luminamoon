use hashbrown::{HashMap, HashSet};
use xxhash_rust::xxh3::{xxh3_64, xxh3_64_with_seed};

pub const DEFAULT_MINHASH_PERMUTATIONS: usize = 64;

#[derive(Debug, Clone)]
pub struct MinHashSignature {
    pub values: Vec<u64>,
}

impl MinHashSignature {
    pub fn compute_char_shingles(text: &str, k: usize, num_perm: usize) -> Self {
        let mut mins = vec![u64::MAX; num_perm];
        if text.is_empty() {
            return Self { values: mins };
        }

        if text.is_ascii() {
            let raw = text.as_bytes();
            let effective_k = k.min(raw.len()).max(1);
            for bytes in raw.windows(effective_k) {
                for (i, min_val) in mins.iter_mut().enumerate() {
                    let seed = (i as u64).wrapping_mul(0x517cc1b727220a95);
                    let h = xxh3_64_with_seed(bytes, seed);
                    if h < *min_val {
                        *min_val = h;
                    }
                }
            }
            return Self { values: mins };
        }

        let char_offsets: Vec<usize> = text.char_indices().map(|(idx, _)| idx).collect();
        let num_chars = char_offsets.len();
        let effective_k = k.min(num_chars).max(1);

        for start_idx in 0..=(num_chars - effective_k) {
            let byte_start = char_offsets[start_idx];
            let byte_end = if start_idx + effective_k < num_chars {
                char_offsets[start_idx + effective_k]
            } else {
                text.len()
            };
            let bytes = &text.as_bytes()[byte_start..byte_end];

            for (i, min_val) in mins.iter_mut().enumerate() {
                let seed = (i as u64).wrapping_mul(0x517cc1b727220a95);
                let h = xxh3_64_with_seed(bytes, seed);
                if h < *min_val {
                    *min_val = h;
                }
            }
        }

        Self { values: mins }
    }

    pub fn jaccard_similarity(&self, other: &Self) -> f64 {
        if self.values.is_empty() || self.values.len() != other.values.len() {
            return 0.0;
        }

        let mut matches = 0usize;
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            if a == b {
                matches += 1;
            }
        }

        (matches as f64) / (self.values.len() as f64)
    }
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
        let n_perm = num_perm.max(8);
        let num_bands = if n_perm >= 64 { 8 } else { 4.min(n_perm) };
        let rows_per_band = n_perm / num_bands;

        Self {
            threshold: threshold.clamp(0.0, 1.0),
            k_shingle: k_shingle.max(1),
            num_perm: n_perm,
            num_bands,
            rows_per_band,
            signatures: Vec::new(),
            lsh_buckets: HashMap::new(),
        }
    }

    pub fn compute_signature(&self, text: &str) -> MinHashSignature {
        MinHashSignature::compute_char_shingles(text, self.k_shingle, self.num_perm)
    }

    fn band_hash(&self, sig: &MinHashSignature, band_idx: usize) -> u64 {
        let start = band_idx * self.rows_per_band;
        let end = (start + self.rows_per_band).min(sig.values.len());
        let slice = &sig.values[start..end];
        let mut byte_buf = Vec::with_capacity(slice.len() * 8);
        for &v in slice {
            byte_buf.extend_from_slice(&v.to_le_bytes());
        }
        xxh3_64(&byte_buf)
    }

    pub fn max_similarity(&self, sig: &MinHashSignature) -> f64 {
        if self.signatures.is_empty() {
            return 0.0;
        }

        // For datasets under 128 items, perform exact scan for 100% precision
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

        // For larger datasets, query LSH candidate buckets in O(1) amortized time
        let mut candidates = HashSet::new();
        for b in 0..self.num_bands {
            let bh = self.band_hash(sig, b);
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

    pub fn is_unique(&mut self, text: &str) -> bool {
        let sig = self.compute_signature(text);
        let max_sim = self.max_similarity(&sig);
        if max_sim >= self.threshold {
            return false;
        }

        let doc_id = self.signatures.len();
        for b in 0..self.num_bands {
            let bh = self.band_hash(&sig, b);
            self.lsh_buckets.entry((b, bh)).or_default().push(doc_id);
        }
        self.signatures.push(sig);
        true
    }

    pub fn count(&self) -> usize {
        self.signatures.len()
    }

    pub fn clear(&mut self) {
        self.signatures.clear();
        self.lsh_buckets.clear();
    }
}
