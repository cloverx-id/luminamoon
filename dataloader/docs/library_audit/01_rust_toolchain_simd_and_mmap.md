# 1. Rust Toolchain, `std::arch` SIMD Intrinsics, `memmap2`, dan Crate Inti

## 1.1 Versi Stabil Terbaru
- **Rust Compiler (`rustc` & `cargo`)**: `1.99.1` (`edition = "2024"`, `resolver = "3"`)
- **`memmap2`**: `0.9.11`
- **`rayon`**: `1.12.0`
- **`thiserror`**: `2.0.21`
- **`serde`**: `1.0.229`
- **`serde_json`**: `1.0.154`

---

## 1.2 Riwayat Perubahan pada Rentang Versi (`Rust 1.85.0` - `1.99.1` & Crate Pendukung)

### A. Rust 1.85.0 (`edition = "2024"`)
- **Apa yang berubah**: Lint `unsafe_op_in_unsafe_fn` diaktifkan secara bawaan pada edisi 2024. Mendeklarasikan fungsi sebagai `unsafe fn` tidak lagi secara implisit mengizinkan pemanggilan operasi `unsafe` di dalam badan fungsi tanpa blok `unsafe { ... }` eksplisit.
- **Perubahan Syntax**: Stabilisasi `let-chains` (`if let Some(a) = opt && a > 0`) di dalam kondisi `if` dan `while`, yang kini diwajibkan oleh lint `clippy::collapsible_if`.

### B. Rust 1.86.0 - 1.89.0 (`std::arch` Target Feature & Stabilisasi `avx512bw`)
- **Rust 1.86.0 - 1.87.0**: Intrinsik SIMD berbasis nilai (*value-based intrinsics* seperti `_mm512_set1_epi8`, `_mm512_cmpeq_epi8_mask`, `_mm256_set1_epi8`, `_mm256_cmpeq_epi8`, `_mm256_movemask_epi8`, `vdupq_n_u8`, `vceqq_u8`, `vmaxvq_u8`) tidak lagi memerlukan blok `unsafe` apabila dipanggil di dalam fungsi beratribut `#[target_feature]`. Membungkus intrinsik nilai tersebut dengan `unsafe` akan memicu peringatan kompilator `unused_unsafe`. Hanya intrinsik dereferensi pointer mentah (`_mm512_loadu_si512`, `_mm256_loadu_si256`, `vld1q_u8`) dan aritmetika pointer (`ptr.add(offset)`) yang tetap berstatus `unsafe`.
- **Rust 1.89.0**: Seluruh fitur target `avx512f` dan `avx512bw` beserta intrinsik byte 512-bit (`_mm512_cmpeq_epi8_mask`) distabilkan pada kanal *stable* tanpa memerlukan `#![feature(stdarch_x86_avx512)]`.

### C. Rust 1.88.0 - 1.95.0 (`[T]::as_chunks` & `std::cfg_select!`)
- **Rust 1.88.0**: Metode pustaka standar `slice::as_chunks::<N>()` distabilkan untuk membagi slice menjadi potongan array berukuran konstan `&[[T; N]]` beserta sisa ekornya, menggantikan pola lama `chunks_exact(N)` (diawasi oleh lint `clippy::chunks_exact_to_as_chunks`).
- **Rust 1.95.0**: Penambahan makro standar `std::cfg_select!` untuk percabangan kompilasi bersyarat lintas arsitektur tanpa memerlukan crate eksternal `cfg-if`.

### D. `memmap2` (`0.9.0` - `0.9.11`): Mitigasi `RUSTSEC-2026-0186`
- **Apa yang berubah**: Pada `memmap2 < 0.9.11`, perhitungan batas akhir pemetaan memori internal menggunakan `ptr.add(len)`. Di bawah model *pointer provenance* Rust/LLVM terbaru, memanggil `ptr.add` pada pemetaan kosong atau di tepi ruang alamat melanggar invarian `inbounds` dan memicu *Undefined Behavior* (UB).
- **Perbaikan Upstream (`0.9.11`)**: Mengganti perhitungan batas pointer internal dengan `wrapping_add` serta menstabilkan opsi `Advice::PopulateRead` (`MADV_POPULATE_READ` pada Linux) untuk *page pre-faulting*.

### E. `thiserror` (`1.0` - `2.0.21`), `rayon` (`1.10` - `1.12.0`), dan `serde_json` (`1.0.120` - `1.0.154`)
- **`thiserror 2.0.21`**: Beralih ke trait `core::error::Error` yang telah distabilkan di `core`, menghilangkan ketergantungan wajib pada `std::error::Error` internal dan memperbaiki ekspansi format variabel pada makro `#[error("...")]`.
- **`rayon 1.12.0`**: Peningkatan efisiensi penjadwalan *work-stealing* pada iterator paralel terindeks (`into_par_iter().enumerate()`).
- **`serde_json 1.0.154`**: Optimasi penulisan langsung ke buffer byte (`serde_json::to_writer(&mut Vec<u8>, ...)`) tanpa alokasi `String` perantara.

---

## 1.3 Adaptasi pada Kode Sumber LuminaMoon

1. **Deteksi Arsitektur dengan `std::cfg_select!` dan Pemisahan Blok `unsafe` Intrinsik (`dataloader/src/simd_cpu.rs`)**:
   - Menggunakan `std::cfg_select!` di dalam `detect_cpu_isa()` untuk memilih cabang `x86`/`x86_64`, `aarch64`, atau fallback `Scalar` tanpa dependensi `cfg-if`.
   - Pada `find_byte_avx512`, `rfind_byte_avx512`, `find_byte_avx2`, `rfind_byte_avx2`, `find_byte_neon`, dan `rfind_byte_neon`, blok `unsafe { ... }` hanya diterapkan pada `_mm512_loadu_si512`, `_mm256_loadu_si256`, `vld1q_u8`, dan `ptr.add(...)`:
     ```rust
     while offset + 64 <= len {
         let chunk = unsafe { _mm512_loadu_si512(ptr.add(offset).cast()) };
         let mask = _mm512_cmpeq_epi8_mask(chunk, v_needle);
         if mask != 0 {
             return Some(offset + mask.trailing_zeros() as usize);
         }
         offset += 64;
     }
     ```
2. **Adopsi `slice::as_chunks::<3>()` (`dataloader/src/multimodal.rs`)**:
   - Fungsi `encode_base64` menggunakan `let (chunks, rem) = input.as_chunks::<3>();` untuk memproses blok 3-byte secara langsung tanpa pemeriksaan batas di dalam loop.
3. **Pemetaan Memori Aman dengan `memmap2 0.9.11` (`dataloader/src/source_local.rs`)**:
   - `LocalMmapLineSource::open` memetakan berkas lokal menggunakan `memmap2 0.9.11` yang telah bebas dari `RUSTSEC-2026-0186` dan menjaga pemeriksaan batas `start_coord.byte_offset.min(total_len)` sebelum pengirisan slice.
