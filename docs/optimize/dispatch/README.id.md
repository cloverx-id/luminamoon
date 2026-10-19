# `luminav-dispatch`: Audit Kapabilitas Hardware & Heuristik Perutean

`luminav-dispatch` bertindak sebagai koordinator inspeksi kemampuan perangkat keras runtime dan mengarahkan beban kerja pembaruan parameter ke backend komputasi yang paling optimal berdasarkan ukuran tensor, ketersediaan arsitektur, dan preferensi konfigurasi.

[English Version (README.md)](README.md)

---

## 1. Mesin Deteksi Hardware (`src/detector.rs`)

Metode audit statis `LuminaV.detect_hardware()` mengaudit kapabilitas mesin host tanpa melempar eksepsi sistem:

```mermaid
flowchart TD
    Detect["Probe Kapabilitas Hardware"] --> CPU["Pemindaian Fitur CPU"]
    Detect --> Vulkan["Kueri Instance Vulkan"]
    Detect --> Triton["Pemeriksaan Dinamis Driver GPU"]

    CPU -->|"AVX-512"| AVX512["Backend AVX-512"]
    CPU -->|"AVX2"| AVX2["Backend AVX2"]
    CPU -->|"Lainnya / ARM"| Scalar["Backend Skalar"]

    Vulkan -->|"Tersedia & UMA"| VK_Ready["Vulkan UMA Aktif"]
    Vulkan -->|"Tidak Tersedia"| VK_Off["Vulkan Nonaktif"]

    Triton -->|"CUDA / HIP Ditemukan"| TR_Ready["Driver Triton Aktif"]
    Triton -->|"Tidak Ditemukan"| TR_Off["Triton Nonaktif"]
```

---

## 2. Heuristik Perutean Dinamis (`src/router.rs`)

Ketika parameter `execution = "auto"`, jalur eksekusi dipilih secara dinamis:

```mermaid
flowchart TD
    Tensor["Tensor Input (Panjang N)"] --> ModeCheck{"Mode Eksplisit Disetel?"}

    ModeCheck -->|"Ya"| ManualExec["Arahkan ke Backend Tertentu"]
    ModeCheck -->|"Auto"| SizeCheck{"N &ge; 16.384 & Vulkan Tersedia?"}

    SizeCheck -->|"Ya"| ExecVulkan["Backend Vulkan UMA"]
    SizeCheck -->|"Tidak"| SIMDCheck{"SIMD Didukung?"}

    SIMDCheck -->|"AVX-512 / AVX2"| ExecSIMD["SIMD CPU Paralel"]
    SIMDCheck -->|"Fallback"| ExecScalar["Engine Skalar Referensi"]
```

### Heuristik Ambang Batas 16.384 Elemen (GPU_DISPATCH_THRESHOLD)
Untuk tensor berukuran kecil hingga menengah ($N < 16.384$ elemen, didefinisikan oleh konstanta `GPU_DISPATCH_THRESHOLD = 16_384`), latensi pengiriman *command buffer* dan sinkronisasi pipeline GPU lebih besar daripada waktu komputasi murni. Pada rentang ini, SIMD CPU multi-thread mengeksekusi dengan *cache locality* yang efisien. Untuk tensor besar ($N \ge 16.384$), eksekusi dialihkan ke pipeline komputasi GPU jika backend tersedia.
