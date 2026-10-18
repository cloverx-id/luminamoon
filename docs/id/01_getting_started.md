# Panduan Memulai (Getting Started)

Panduan instalasi, kebutuhan lingkungan, kompilasi dari sumber, dan contoh kode kilat untuk menjalankan optimizer **LuminaV v1.0.0**.

## Kebutuhan Sistem

- **Python**: Versi 3.10 atau lebih baru (kompatibel penuh dengan Python 3.10 hingga 3.15+, termasuk runtime *free-threaded* `abi3t` / PEP 703).
- **Sistem Operasi**: Linux (x86_64, aarch64), Windows (x86_64), atau macOS (Apple Silicon & Intel).
- **Kompilasi Sumber (Opsional)**:
  - Rust Toolchain: `rustc` dan `cargo` versi 1.75+.
  - Python Build Tool: `maturin >= 1.15, < 2.0`.
- **Akselerator Perangkat Keras (Opsional)**:
  - CPU SIMD: Prosesor x86_64 dengan dukungan instruksi AVX2 atau AVX-512 (F, DQ, BW, VL).
  - Vulkan Compute: Driver Vulkan 1.3 / 1.4 berkemampuan compute shader (SPIR-V).
  - Triton JIT: Driver NVIDIA CUDA (SM75–SM120) atau AMD ROCm (RDNA3/CDNA) dengan paket `triton >= 3.0.0` (atau `triton-windows`).

## Instalasi

### 1. Instalasi Paket Standar (Tanpa Dependensi Berat)

Paket inti `luminamoon` dirancang sangat ringan dengan ukuran binary Rust di bawah 1 MB dan tanpa dependensi eksternal wajib:

```bash
pip install luminamoon
```

### 2. Instalasi dengan Ekstensi Opsional

Jika ingin mengintegrasikan LuminaMoon dengan framework machine learning tertentu, gunakan opsi *extras*:

```bash
# Untuk integrasi PyTorch
pip install "luminamoon[torch]"

# Untuk akselerasi Triton JIT pada GPU (CUDA / ROCm)
pip install "luminamoon[gpu]"

# Untuk NumPy 2.x
pip install "luminamoon[numpy]"

# Menginstal seluruh dependensi eksternal
pip install "luminamoon[all]"
```

### 3. Kompilasi dari Sumber (Source Build)

Untuk mendapatkan performa optimal yang dikompilasi langsung sesuai arsitektur mesin host:

```bash
git clone https://github.com/cloverx-id/luminamoon.git
cd luminamoon

# Buat dan aktifkan virtual environment (opsional)
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Instal maturin build tool
pip install "maturin>=1.15,<2.0"

# Kompilasi native extension dalam mode release
maturin develop --release
```

Hasil kompilasi akan menaruh pustaka native `_luminamoon_optimizer_rs` langsung ke dalam folder Python `luminamoon/optimizer/`.

## Deteksi Kemampuan Perangkat Keras

LuminaMoon menyediakan metode statis untuk mengaudit kapabilitas perangkat keras saat runtime:

```python
from luminamoon import LuminaV

hw = LuminaV.detect_hardware()
print("Hardware Capabilities:", hw)
# Contoh Output:
# {
#     'simd_backend': 'x86_64 AVX-512 (avx512f+dq+bw+vl+fma)',
#     'triton_aot_available': False,
#     'triton_arch': None,
#     'vulkan_available': True,
#     'gpu_device_name': 'AMD Radeon (TM) 740M Graphics',
#     'wave32_supported': True,
#     'gpu_dispatch_threshold': 131072
# }
```

## Quickstart: Standalone NumPy (Tanpa PyTorch)

LuminaMoon dapat digunakan secara mandiri untuk proyek sains data, komputasi numerik, atau custom neural network berbasis NumPy tanpa memerlukan instalasi PyTorch sama sekali:

```python
import numpy as np
from luminamoon import LuminaV

# 1. Inisialisasi parameter model sebagai array NumPy contiguous berpresisi FP32
rng = np.random.default_rng(42)
weights = rng.standard_normal((64, 10), dtype=np.float32)
bias = np.zeros((10,), dtype=np.float32)

# 2. Inisialisasi optimizer LuminaV
optimizer = LuminaV(
    [weights, bias],
    lr=0.01,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    buffer=2,  # 2: Dual-Buffer CLM, 1: Single-Buffer TD-CLM
    bound=True,
    cautious=True,
)

# 3. Training loop dengan gradien eksplisit
for step in range(1, 11):
    # Hitung gradien sintetis (misal dari loss function Anda)
    grad_w = rng.standard_normal((64, 10), dtype=np.float32) * 0.1
    grad_b = rng.standard_normal((10,), dtype=np.float32) * 0.1

    # Eksekusi step optimasi dengan mengoper daftar gradien
    optimizer.step(grads=[grad_w, grad_b])
    print(f"Step {step:02d} selesai dieksekusi via Rust SIMD / Vulkan UMA.")
```

## Quickstart: PyTorch Integration

Integrasi langsung dengan PyTorch `nn.Module` mengikuti antarmuka standar `torch.optim.Optimizer`:

```python
import torch
import torch.nn as nn
from luminamoon import LuminaV

# 1. Definisikan model PyTorch
class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)

model = Regressor()

# 2. Inisialisasi optimizer LuminaV
optimizer = LuminaV(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    buffer=2,             # 2 = Dual-Buffer CLM
    master_weights="fp32", # Menjaga bobot master FP32 saat training FP16/BF16
    stochastic_rounding=True,
    cautious=True,
)

criterion = nn.MSELoss()
x_batch = torch.randn(32, 64)
y_target = torch.randn(32, 10)

# 3. Standar Training Loop
for step in range(1, 11):
    optimizer.zero_grad(set_to_none=True)
    out = model(x_batch)
    loss = criterion(out, y_target)
    loss.backward()
    optimizer.step()

    print(f"Step {step:02d} | Loss: {loss.item():.6f}")
```
