# Panduan Memulai (Getting Started)

Panduan instalasi, kebutuhan lingkungan, kompilasi dari sumber, verifikasi hardware, dan contoh kode kilat untuk menjalankan **LuminaV Optimizer v1.0.0** serta **LuminaMoon DataLoader v1.0.0**.

## Kebutuhan Sistem

- **Python**: Versi 3.10 atau lebih baru (kompatibel penuh dengan Python 3.10 hingga 3.15+, termasuk runtime *free-threaded* `abi3t` / PEP 703).
- **Sistem Operasi**: Linux (x86_64, aarch64), Windows (x86_64), atau macOS (Apple Silicon & Intel).
- **Kompilasi Sumber (Opsional)**:
  - Rust Toolchain: `rustc` dan `cargo` versi 1.85+ (Edition 2024).
  - Python Build Tool: `maturin >= 1.15, < 2.0`.
- **Akselerator Perangkat Keras (Opsional)**:
  - CPU SIMD: Prosesor x86_64 dengan dukungan instruksi AVX2 atau AVX-512 (F, DQ, BW, VL), atau ARM64 NEON.
  - Vulkan Compute: Driver Vulkan 1.3 / 1.4 berkemampuan compute shader (SPIR-V).
  - Triton JIT: Driver NVIDIA CUDA (SM75–SM120) atau AMD ROCm (RDNA3/CDNA) dengan paket `triton >= 3.0.0` (atau `triton-windows`).

## Instalasi

### 1. Instalasi Paket Standar (Tanpa Dependensi Berat)

Paket inti `luminamoon` dirancang sangat ringan tanpa dependensi Python eksternal wajib (`dependencies = []`) dan menyediakan `LuminaV` sekaligus pipeline `luminamoon.dataloader`:

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

Untuk mengompilasi workspace native Rust (`optimizer` + `dataloader`) secara lokal sesuai arsitektur mesin host:

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

Hasil kompilasi akan menaruh pustaka native `_luminamoon_optimizer_rs` (yang memuat `luminamoon-optimizer` dan `luminamoon-dataloader`) langsung ke dalam folder `python/luminamoon/optimizer/`.

## Deteksi Kemampuan Perangkat Keras

LuminaMoon menyediakan metode statis untuk mengaudit kapabilitas perangkat keras pada subsistem Optimizer maupun DataLoader:

```python
import luminamoon as lm
from luminamoon import LuminaV

# 1. Periksa kapabilitas SIMD, Vulkan UMA, dan Triton AOT pada Optimizer
hw = LuminaV.detect_hardware()
print("Optimizer Hardware:", hw)

# 2. Periksa set instruksi SIMD pemindaian byte pada DataLoader ('AVX-512', 'AVX2', 'NEON', atau 'SCALAR')
print("DataLoader CPU ISA:", lm.dataloader_cpu_isa())
```

## Quickstart: NumPy Mandiri (Tanpa PyTorch)

LuminaV dapat digunakan secara mandiri untuk proyek sains data, komputasi numerik, atau custom neural network berbasis NumPy tanpa memerlukan instalasi PyTorch:

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
    grad_w = rng.standard_normal((64, 10), dtype=np.float32) * 0.1
    grad_b = rng.standard_normal((10,), dtype=np.float32) * 0.1

    optimizer.step(grads=[grad_w, grad_b])
    print(f"Step {step:02d} selesai dieksekusi via Rust SIMD / Vulkan UMA.")
```

## Quickstart: Integrasi PyTorch

Integrasi langsung dengan PyTorch `nn.Module` mengikuti antarmuka standar `torch.optim.Optimizer`:

```python
import torch
import torch.nn as nn
from luminamoon import LuminaV

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

for step in range(1, 11):
    optimizer.zero_grad(set_to_none=True)
    out = model(x_batch)
    loss = criterion(out, y_target)
    loss.backward()
    optimizer.step()

    print(f"Step {step:02d} | Loss: {loss.item():.6f}")
```

## Quickstart: LuminaMoon DataLoader & Pipeline HuggingFace

Alirkan dataset lokal maupun remote dari HuggingFace Hub (`.parquet`, `.jsonl`, `.txt`) dengan pemulihan crash $O(1)$, *prefetching* watermark Pre-RAM 100-batch, deduplikasi dua tahap `xxh3_64`, serta unggahan tabel multimodal:

```python
import luminamoon as lm

# 1. Alirkan dataset lokal atau hf://datasets/... dengan penemuan shard otomatis dan hook tokenizer
for batch in lm.stream(
    "hf://datasets/org/corpus@main",
    split="train",
    batch_size=64,
    preram_batches=100,
    refill_trigger=10,
    download_workers=4,
    deduplicate=True,
    tokenizer=lambda texts, **kw: {"input_ids": [[len(t)] for t in texts]},
):
    records = batch.records
    input_ids = batch.tokens["input_ids"]
    # Deferred auto_commit otomatis menyimpan progres GPU ke slot Ping-Pong 128B .meta

# 2. Bungkus datasets.Dataset / IterableDataset dengan normalisasi SIMD + Rayon + deduplikasi XXH3-64
for batch in lm.from_hf_dataset(
    [{"text": "  Sample A \r\n"}, {"text": "Sample A"}, {"text": "Sample B"}],
    column="text",
    batch_size=32,
    deduplicate=True,
):
    print(batch.records)  # ['Sample A', 'Sample B']

# 3. Serialisasi & unggah tabel multimodal (Text, Image Bytes, JSON, Int64, Float64, Bool) ke HuggingFace Hub
uri = lm.push_to_hub(
    repo="org/cleaned-multimodal-ds",
    data=[
        {"prompt": "hello", "image": b"\x89PNG...", "score": 0.98, "verified": True},
    ],
    path_in_repo="train/part-00000.parquet",
)
```
