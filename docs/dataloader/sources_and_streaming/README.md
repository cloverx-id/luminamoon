# Heuristic Quality Filtering & MinHash LSH

[`data/src/quality.rs`](https://github.com/cloverx-id/luminamoon/blob/main/data/src/quality.rs) and [`data/src/minhash.rs`](https://github.com/cloverx-id/luminamoon/blob/main/data/src/minhash.rs) implement parallel text quality evaluation and near-duplicate fuzzy document detection in Rust.

---

## 1. Heuristic Quality Engine (`src/quality.rs`)

`QualityConfig` and `evaluate_quality(text, &cfg)` evaluate document quality across multiple criteria in a single pass, parallelized across batches using Rayon:

| Metric / Field | Type | Inactive Default | Mathematical & Algorithmic Definition |
| :--- | :--- | :--- | :--- |
| `min_chars` / `max_chars` | `usize` | `0` (unbounded) | Unicode scalar value count (`text.chars().count()`). |
| `min_words` | `usize` | `0` (unbounded) | Whitespace-delimited token count (`text.split_whitespace().count()`). |
| `min_entropy` | `f64` | `0.0` | Shannon character entropy $H(X) = -\sum _ c p(c) \log_2 p(c)$ computed over character frequencies. |
| `max_repetition_ratio` | `f64` | `1.0` | Fraction of duplicate word 3-grams (`0.0` when `word_count < 4`) detecting degenerate generation loops. |
| `max_symbol_ratio` | `f64` | `1.0` | Ratio of non-alphanumeric symbol characters to total non-whitespace characters. |

---

## 2. Single-Pass Double-Hashed MinHash LSH (`src/minhash.rs`)

To detect documents that differ only by minor formatting, timestamps, or boilerplate:

1. **Character N-Gram Shingling**:
    - Extracts sliding $k$-character UTF-8 shingles (`k_shingle = 5` by default) with a fast-path for pure ASCII inputs.
2. **Double-Hashed Permutations (`MinHashSignature`, adapted from Kirsch-Mitzenmacher)**:
    - Computes one 128-bit `xxh3_128` hash per shingle, splits it into $(h_1, h_2)$ with odd $h_2$, and derives $P$ permutations via `mix64(h1 + i * h2)` (MurmurHash3 `fmix64` finalizer), retaining the minimum per slot.
3. **Auto-Tuned LSH Banding & Jaccard Estimation**:
    - Partitions signatures into auto-tuned `(num_bands, rows_per_band)` LSH buckets to retrieve candidate matches without an $O(N^2)$ linear scan, and verifies candidate Jaccard similarity:

      $\displaystyle \hat{J}(A, B) = \frac{1}{P} \sum _ {i=1}^{P} \mathbb{I}\left(\text{sig} _ A[i] == \text{sig} _ B[i]\right)$

    - `MinHashDeduplicator::is_unique(text)` and Rayon-parallel `filter_batch(texts)` reject documents with estimated similarity $\ge \text{threshold}$ (`0.85` by default).
