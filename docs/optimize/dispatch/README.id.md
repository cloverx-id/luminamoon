# `luminav-dispatch`: Audit Kapabilitas Hardware & Heuristik Perutean

`luminav-dispatch` bertindak sebagai koordinator inspeksi kemampuan perangkat keras runtime dan mengarahkan beban kerja pembaruan parameter ke backend komputasi yang paling optimal berdasarkan ukuran tensor, ketersediaan arsitektur, dan preferensi konfigurasi.

[English Version (README.md)](README.md)

---

## 1. Mesin Deteksi Hardware (`src/detector.rs`)

Metode audit statis `LuminaV.detect_hardware()` mengaudit kapabilitas mesin host tanpa melempar eksepsi sistem:

```mermaid
flowchart TD
    Detect["detect_hardware()"] --> CPU["Audit Fitur x86_64"]
    Detect --> VK["Kueri Instance Vulkan"]
    Detect --> TR["Pemeriksaan Driver Triton AOT"]

    CPU -->|"avx512f, dq, bw, vl, fma"| AVX512["SimdBackend::Avx512"]
    CPU -->|"avx2, fma"| AVX2["SimdBackend::Avx2"]
    CPU -->|"arsitektur lain / ARM"| Scalar["SimdBackend::Scalar"]

    VK -->|"Vulkan 1.3/1.4 + UMA"| VK_OK["vulkan_available = true<br/>wave32_supported = true/false"]
    TR -->|"libloading CUDA / HIP"| TR_OK["triton_aot_available = true/false"]
```

---

## 2. Heuristik Perutean Dinamis (`src/router.rs`)

Ketika parameter `execution = "auto"`, jalur eksekusi dipilih secara dinamis:

```mermaid
graph TD
    Input["Parameter Tensor (numel = N)"] --> CheckExplicit{"Mode Eksekusi Eksplisit?"}
    
    CheckExplicit -->|"Vulkan / SIMD / Scalar"| ExecSelected["Arahkan ke Backend yang Dipilih"]
    CheckExplicit -->|"Auto"| CheckGPU{"N >= 131,072 & Vulkan Siap?"}

    CheckGPU -->|"Ya"| DispatchVK["Arahkan ke luminav-vulkan (Wave32 UMA)"]
    CheckGPU -->|"Tidak"| CheckSIMD{"SIMD x86_64 Tersedia?"}

    CheckSIMD -->|"AVX-512 / AVX2"| DispatchSIMD["Arahkan ke luminav-simd (Rayon Parallel)"]
    CheckSIMD -->|"Fallback"| DispatchScalar["Arahkan ke luminav-core (Scalar Engine)"]
```

### Heuristik Ambang Batas 131.072 Elemen
Untuk tensor berukuran kecil hingga menengah ($N < 131.072$ elemen), latensi pengiriman *command buffer* dan sinkronisasi pipeline GPU lebih besar daripada waktu komputasi murni. Pada rentang ini, SIMD CPU multi-thread mengeksekusi dengan *cache locality* yang jauh lebih cepat. Untuk tensor besar ($N \ge 131.072$), eksekusi dialihkan ke pipeline komputasi GPU.
