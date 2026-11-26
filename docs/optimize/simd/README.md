# `luminav-simd`: Explicit CPU Vectorization & Rayon Threadpool

`luminav-simd` provides explicit SIMD vectorization implementations targeting x86_64 and ARM64 host processors, combining compiler vector intrinsics with multi-core work stealing via Rayon.

---

## 1. Instruction Set Support & Vector Widths

| Architecture Target | Registers | Floats / Register | Compiler Flags / Extensions | Implementation |
| :--- | :--- | :--- | :--- | :--- |
| **AVX-512** | ZMM (`__m512`) | 16 floats | `avx512f, avx512dq, avx512bw, avx512vl` | [`src/avx512/mod.rs`](https://github.com/cloverx-id/luminamoon/blob/main/optimizer/crates/luminav-simd/src/avx512/mod.rs) |
| **AVX2 + FMA** | YMM (`__m256`) | 8 floats | `avx2, fma` | [`src/avx2/mod.rs`](https://github.com/cloverx-id/luminamoon/blob/main/optimizer/crates/luminav-simd/src/avx2/mod.rs) |
| **ARM64 NEON** | V-reg (`float32x4_t`) | 4 floats | `neon` (`aarch64`) | [`src/neon/mod.rs`](https://github.com/cloverx-id/luminamoon/blob/main/optimizer/crates/luminav-simd/src/neon/mod.rs) |
| **Scalar Fallback** | General Purpose | 1 float | Any portable target | [`luminav-core`](../core/README.md) |

---

## 2. Multi-Pass vs Fused Single-Pass Execution

### 2.1 Pass 1: Local Reductions & Intermediate Steps
1. Computes local sums for cautious masking $\sum \mathcal{M}_i$ and parameter norms $\sum w_i^2$.
2. Uses vector lane tree reductions (`reduce_sum_16`, `reduce_sum_8`, `reduce_sum_4`) to aggregate partial thread accumulations without cross-lane horizontal-add bottlenecks.

### 2.2 Pass 2: Parameter Mutation & Bounding Projection
1. Normalizes cautious scaling $s _ {\text{cautious}}$ and radial ($L_2$-norm) bounding $s _ {\text{radial}}$ from the global reduction.
2. Direct in-place mutation of parameter buffer `param` and state accumulator `exp_avg` with aligned streaming or cached stores.

### 2.3 Fused Single-Pass Mode
When `fused_single_pass = true`, the step avoids a two-pass barrier by projecting bounds with the prior step's Exponential Moving Average (EMA) coefficients ($s _ {\text{cautious, ema}}$, $s _ {\text{radial, ema}}$), fusing update and EMA accumulation into a single memory pass.

---

## 3. Parallel Chunking Architecture (`src/parallel/mod.rs`)

To maximize L1/L2 data cache locality and reduce thread contention:

- **Chunk Partitioning**: Tensors are partitioned into contiguous slices of `ELEM_CHUNK_SIZE = 2,097,152` elements per Rayon task across `src/parallel/dual.rs` and `src/parallel/tdclm.rs`.
- **Cache-Line Alignment**: Work slices are aligned to 64-byte boundaries.
- **Scalar Tail Handling**: Remainder elements ($N \pmod{W} \ne 0$, where $W \in \{16, 8, 4\}$) finish in clean scalar tail loops without out-of-bounds page fault hazards.
