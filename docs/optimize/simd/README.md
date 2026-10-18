# `luminav-simd`: Explicit x86_64 Vectorization & Rayon Threadpool

`luminav-simd` provides explicit SIMD vectorization implementations targeting modern x86_64 host processors, combining compiler vector intrinsics with multi-core work stealing via Rayon.

[Versi Bahasa Indonesia (README.id.md)](README.id.md)

---

## 1. Instruction Set Support & Vector Widths

| Architecture Target | Registers | Floats / Register | Compiler Flags / Extensions | Implementation |
| :--- | :--- | :--- | :--- | :--- |
| **AVX-512** | ZMM (`__m512`) | 16 floats | `avx512f, avx512dq, avx512bw, avx512vl` | [`src/avx512.rs`](file:///c:/Users/rifial/Downloads/PROJECT%20LUMI/luminamoon/optimizer/crates/luminav-simd/src/avx512.rs) |
| **AVX2 + FMA** | YMM (`__m256`) | 8 floats | `avx2, fma` | [`src/avx2.rs`](file:///c:/Users/rifial/Downloads/PROJECT%20LUMI/luminamoon/optimizer/crates/luminav-simd/src/avx2.rs) |
| **Scalar Fallback** | General Purpose | 1 float | Any portable target | [`luminav-core`](../core/README.md) |

---

## 2. Multi-Pass vs Fused Single-Pass Execution

### 2.1 Pass 1: Local Reductions & Intermediate Steps
1. Computes local sums for cautious masking $\sum \mathcal{M}_i$ and parameter norms $\sum w_i^2$.
2. Utilizes vector horizontal adds (`_mm256_hadd_ps` / `_mm512_reduce_add_ps`) to aggregate partial thread accumulations.

### 2.2 Pass 2: Parameter Mutation & Bounding Projection
1. Normalizes cautious scaling $s_{\text{cautious}}$ and manifold bounding $s_{\text{radial}}$ from the global reduction.
2. Direct in-place mutation of parameter buffer `param` and state accumulator `exp_avg` with aligned streaming or cached stores.

### 2.3 Fused Single-Pass Mode
When `fused_single_pass = true`, the step avoids a two-pass barrier by projecting bounds with the prior step's Exponential Moving Average (EMA) coefficients ($s_{\text{cautious, ema}}, s_{\text{radial, ema}}$), fusing update and EMA accumulation into a single memory pass.

---

## 3. Parallel Chunking Architecture (`src/parallel.rs`)

To maximize L1/L2 data cache locality and eliminate thread contention:
- **Chunk Partitioning**: Tensors are partitioned into contiguous slices of 8,192 elements per Rayon task.
- **Cache-Line Alignment**: Work slices are aligned to 64-byte boundaries.
- **Unrolled Tail Masking**: Remainder elements ($N \pmod{16} \ne 0$) are masked using AVX-512 `_mm512_mask_storeu_ps` or scalar tail loops without out-of-bounds page fault hazards.
