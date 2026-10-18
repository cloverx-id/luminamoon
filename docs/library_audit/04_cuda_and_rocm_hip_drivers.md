# 4. Driver GPU Diskrit: NVIDIA CUDA dan AMD ROCm / HIP (TheRock)

## 4.1 Versi Stabil Terbaru
- **NVIDIA CUDA Driver API**: `CUDA 12.x / 13.x` (`nvcuda.dll` / `libcuda.so.1`)
- **AMD ROCm / HIP Runtime**: `ROCm 7.x` / `ROCm Core SDK (TheRock) 10.1.0` (`therock-10.1`)

---

## 4.2 Riwayat Perubahan AMD ROCm & HIP (`ROCm 5.7` - `ROCm 7.x / TheRock 10.1.0`)

### A. Penamaan Berkas Pustaka Dinamis (`.dll` / `.so`) Berversi Mayor
- **Apa yang berubah**:
  - Pada **ROCm 5.x**, pustaka runtime HIP di Windows bernama `amdhip64.dll`.
  - Mulai **ROCm 6.1.2+**, AMD mengubah nama DLL Windows menjadi `amdhip64_6.dll` agar beberapa versi runtime ROCm dapat berdampingan di satu sistem.
  - Pada **ROCm 7.x** dan distribusi modular **TheRock 10.x** (`_rocm_sdk_core`), nama DLL Windows berubah menjadi `amdhip64_7.dll`, sedangkan di Linux menggunakan `libamdhip64.so.7` (di samping symlink `libamdhip64.so` dan `libamdhip64.so.6`).
- **Adaptasi di `optimizer/crates/luminav-triton-aot/src/hip_driver.rs`**:
  Fungsi `candidate_hip_lib_names()` dan `HipDriver::probe()` diperbarui untuk mencoba memuat kandidat pustaka secara berurutan dari versi terbaru ke versi lama:
  - **Windows**: `["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"]`
  - **Linux / Unix**: `["libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so", "libamdhip64.so.5"]`

### B. Perubahan Simbol ABI `hipGetDevicePropertiesR0600`
- **Apa yang berubah**: Mulai ROCm 6.0, struktur C `hipDeviceProp_t` mengalami perubahan tata letak *field*, sehingga simbol fungsi diubah menjadi `hipGetDevicePropertiesR0600` (sementara `hipGetDeviceProperties` dipertahankan sebagai simbol kompatibilitas lama).
- **Adaptasi di `hip_driver.rs`**:
  - `HipDriver::probe()` mencari simbol `hipGetDevicePropertiesR0600` terlebih dahulu, lalu melakukan fallback ke `hipGetDeviceProperties`.
  - Karena posisi byte *field* `gcnArchName` berbeda antara ABI ROCm 5 dan `R0600`, fungsi `extract_gcn_arch_from_prop_bytes` memindai buffer 4096-byte (`HipDevicePropBuffer`) secara langsung untuk mencari pola string `"gfx..."` berakhiran *null-terminator*, sehingga tahan terhadap perubahan *offset* struct di ROCm 5, 6, 7, hingga 10.

### C. Arsitektur GPU AMD Baru & Ukuran Wavefront (`Wave32` vs `Wave64`)
- **Apa yang berubah**:
  - **CDNA 4 (`gfx950` - Instinct MI350X/MI355X)**: Menggunakan ukuran wavefront `64` (*Wave64*), sekeluarga dengan CDNA 3 (`gfx942`).
  - **RDNA 3.5 (`gfx1150`, `gfx1151` - Ryzen AI 300 Strix Point / Ryzen AI Max+ Strix Halo)** dan **RDNA 4 (`gfx1200`, `gfx1201`, `gfx1250` - Radeon RX 9070 XT)**: Menggunakan ukuran wavefront `32` (*Wave32*).
- **Adaptasi di `hip_driver.rs`**:
  - `wavefront_size_for_arch(arch)` diperbarui agar mengembalikan `32` untuk seluruh keluarga `gfx10*`, `gfx11*`, dan `gfx12*`, serta `64` untuk keluarga CDNA (`gfx9*`).
  - `resolve_amd_gfx_arch(gcn_arch_raw, available)` diperbarui agar memetakan varian `gfx940`/`gfx941`/`gfx950` ke biner `gfx942` (Wave64) dan varian `gfx11xx`/`gfx12xx` ke biner `gfx1100` (Wave32) apabila biner spesifik sub-varian belum dipisahkan.

---

## 4.3 Riwayat Perubahan Arsitektur NVIDIA CUDA (`CUDA 12.x` - `13.x` & Triton `3.6` - `3.8`)

### A. Pemisahan Compute Capability Data Center vs Client & Rubin
- **Apa yang berubah**:
  - **Blackwell Data Center**: `sm_100` (B100/B200/GB200) dan `sm_103` (B300/GB300 Blackwell Ultra).
  - **NVIDIA Rubin**: `sm_107` (didukung mulai Triton 3.7/3.8).
  - **Blackwell Client / Workstation & DGX Spark**: `sm_120` (RTX 5090 / RTX PRO 6000 Blackwell) dan `sm_121` (GB10 Superchip).
  - Penting: Biner `sm_100`/`sm_103` menggunakan instruksi *Tensor Memory* (`TMEM`) dan *Cluster Launch Control* yang tidak kompatibel dengan arsitektur *client* `sm_120`/`sm_121`.
- **Adaptasi di `optimizer/crates/luminav-triton-aot/src/cuda_driver.rs`**:
  Fungsi `resolve_cuda_cubin_arch(major, minor, available)` mencocokkan *compute capability* hanya di dalam keluarga `major` yang sama (`c_maj == major && c_min <= minor`). Dengan demikian:
  - `sm_107` terpetakan dengan aman ke `sm_103` (mayor 10).
  - `sm_121` terpetakan dengan aman ke `sm_120` (mayor 12) dan tidak pernah salah memuat biner `sm_100`/`sm_103`.
