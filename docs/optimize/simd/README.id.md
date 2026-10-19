# `luminav-simd`: Vektorisasi SIMD x86_64 & Threadpool Rayon

`luminav-simd` menyediakan implementasi vektorisasi instruksi eksplisit untuk prosesor x86_64 modern, menggabungkan intrinsik vektor tingkat rendah dengan penjadwalan multi-core *work-stealing* berbasis Rayon.

[English Version (README.md)](README.md)

---

## 1. Dukungan Set Instruksi & Lebar Vektor

| Target Arsitektur | Register | Float / Register | Ekstensi CPU Wajib | Berkas Sumber |
| :--- | :--- | :--- | :--- | :--- |
| **AVX-512** | ZMM (`__m512`) | 16 float | `avx512f, avx512dq, avx512bw, avx512vl` | [`src/avx512.rs`](../../../optimizer/crates/luminav-simd/src/avx512.rs) |
| **AVX2 + FMA** | YMM (`__m256`) | 8 float | `avx2, fma` | [`src/avx2.rs`](../../../optimizer/crates/luminav-simd/src/avx2.rs) |
| **Scalar Fallback** | General Purpose | 1 float | Arsitektur apa pun | [`luminav-core`](../core/README.id.md) |

---

## 2. Eksekusi Multi-Pass vs Fused Single-Pass

### 2.1 Pass 1: Reduksi Parsial & Step Perantara
1. Menghitung akumulasi lokal cautious mask $\sum \mathcal{M}_i$ dan norm bobot $\sum w_i^2$.
2. Memanfaatkan instruksi horizontal add (`_mm256_hadd_ps` / `_mm512_reduce_add_ps`) untuk mereduksi parsial register vektor per thread.

### 2.2 Pass 2: Mutasi Bobot & Proyeksi Batas
1. Melakukan normalisasi skala cautious $s_{\text{cautious}}$ dan batas manifold radial $s_{\text{radial}}$ dari hasil reduksi global.
2. Memperbarui buffer parameter `param` dan akumulator `exp_avg` secara langsung di memori (*in-place*).

### 2.3 Mode Fused Single-Pass
Ketika opsi `fused_single_pass = true` diaktifkan, optimizer menghindari barrier sinkronisasi 2-pass dengan memproyeksikan batas menggunakan bobot EMA langkah sebelumnya ($s_{\text{cautious, ema}}, s_{\text{radial, ema}}$), menyatukan mutasi bobot dan pelacakan EMA ke dalam satu siklus pembacaan RAM saja.

---

## 3. Arsitektur Pemecahan Beban Kerja Rayon (`src/parallel.rs`)

Untuk memaksimalkan *cache locality* L1/L2 dan mencegah pertikaian antar-thread:
- **Partisi Chunk**: Tensor dibagi menjadi potongan berukuran 8.192 elemen per task Rayon.
- **Keselarasan Garis Cache**: Setiap potongan dialokasikan pada kelipatan batas 64-byte.
- **Masking Sisa (Tail Remainder)**: Sisa elemen pembagian ($N \pmod{16} \ne 0$) diproses melalui *masked store* `_mm512_mask_storeu_ps` atau loop skalar yang aman dari risiko *page fault*.
