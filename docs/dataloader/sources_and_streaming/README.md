# `quality.rs` & `minhash.rs`: Heuristic Quality Filtering & 64-Permutation MinHash LSH

[`data/src/quality.rs`](https://github.com/cloverx-id/luminamoon/blob/main/data/src/quality.rs) and [`data/src/minhash.rs`](https://github.com/cloverx-id/luminamoon/blob/main/data/src/minhash.rs) implement parallel pre-training corpus quality evaluation and near-duplicate fuzzy document detection in Rust.

---

## 1. Heuristic Quality Engine (`src/quality.rs`)

`QualityConfig` and `evaluate_quality(text, &cfg)` evaluate six orthogonal document quality criteria in a single pass, parallelized across batches via Rayon (`filter_quality_batch_parallel`):

| Metric / Field | Type | Inactive Default | Mathematical & Algorithmic Definition |
| :--- | :--- | :--- | :--- |
| `min_chars` / `max_chars` | `usize` | `0` (unbounded) | Unicode scalar value count (`text.chars().count()`). |
| `min_words` | `usize` | `0` (unbounded) | Whitespace-delimited token count (`text.split_whitespace().count()`). |
| `min_entropy` | `f64` | `0.0` | Shannon character entropy $H(X) = -\sum_{c} p(c) \log_2 p(c)$ computed over character frequencies. |
| `max_repetition_ratio` | `f64` | `1.0` | Fraction of duplicate word 3-grams (or duplicate words for short texts) detecting degenerate generation loops. |
| `max_symbol_ratio` | `f64` | `1.0` | Ratio of non-alphanumeric symbol characters to total non-whitespace characters. |

---

## 2. 64-Permutation MinHash LSH (`src/minhash.rs`)

To detect near-duplicate documents that differ only by minor formatting, timestamps, or boilerplate:

1. **Character N-Gram Shingling**:
   - Normalizes whitespace and extracts sliding $k$-character shingles (`k_shingle = 5` by default).
2. **Permuted `xxh3_64` Signatures (`MinHashSignature`)**:
   - Computes base 64-bit `xxh3_64` hashes for each shingle and applies $P = 64$ deterministic linear congruential permutations ($h_i(x) = (a_i \cdot x + b_i) \bmod 2^{64}$), retaining the minimum hash per permutation slot.
3. **Jaccard Similarity Estimation**:
   - Estimates the Jaccard similarity between two documents $A$ and $B$ in $O(P)$ time:
     $$\hat{J}(A, B) = \frac{1}{P} \sum_{i=1}^{P} \mathbb{I}\left(\text{sig}_A[i] == \text{sig}_B[i]\right)$$
   - `MinHashDeduplicator::is_unique(text)` rejects any document whose estimated similarity against previously accepted signatures is $\ge \text{threshold}$ (`0.85` by default).
