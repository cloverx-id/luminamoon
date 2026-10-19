# Panduan Penggunaan (User Guide)

Panduan mendalam mengenai fitur-fitur teknis **LuminaV Optimizer** (pemilihan mode buffer, mixed-precision, boundary dynamics, checkpointing, distributed training FSDP) serta **LuminaMoon DataLoader** (streaming lokal/HTTP-206 $O(1)$, ring buffer watermark Pre-RAM, pemulihan crash Ping-Pong, deduplikasi dua tahap XXH3-64, dan pipeline multimodal HuggingFace).

## 1. Pemilihan Mode Buffer Optimizer: Dual-Buffer vs Single-Buffer

LuminaV menyediakan dua paradigma algoritma optimasi utama yang dikontrol melalui argumen `buffer`:

```python
# Mode 1: Dual-Buffer CLM (Default)
optimizer = LuminaV(model.parameters(), buffer=2)

# Mode 2: Single-Buffer TD-CLM (Hemat Memori)
optimizer = LuminaV(model.parameters(), buffer=1)
```

### Perbandingan Teknis

| Karakteristik | Dual-Buffer CLM (`buffer=2`) | Single-Buffer TD-CLM (`buffer=1`) |
| :--- | :--- | :--- |
| **State Buffer** | Menyimpan `exp_avg` ($m$) dan `exp_avg_sq` ($v$). | Menyimpan hanya `exp_avg` ($m$). |
| **Footprint Memori** | Identik dengan AdamW ($2 \times$ ukuran parameter). | **50% lebih hemat** ($1 \times$ ukuran parameter). |
| **Mekanisme Penskalaan** | Menghitung $\sigma_t = \sqrt{v_t} \cdot c_1 + c_2$ dengan koreksi bias Nesterov dinamis. | Menghitung Temporal-Difference Envelope $\text{env} = \beta_1^2 |m| + (1-\beta_1^2)|g| + \lambda |\Delta| + \epsilon$. |
| **Kasus Penggunaan Optimal** | Pre-training Transformer, LLM besar, model Vision (ViT/CNN), di mana stabilitas ragam gradien adalah prioritas utama. | Fine-tuning LLM (LoRA/Full), memory-constrained training, edge devices, dan high-throughput training. |

---

## 2. Mixed-Precision & Stochastic Rounding

LuminaV mendukung training berpresisi rendah (`torch.float16` dan `torch.bfloat16`) dengan perlindungan kehilangan gradien (*gradient underflow*).

### Konfigurasi Master Weights

Argumen `master_weights` mengontrol apakah optimizer memelihara bobot parameter dalam format FP32:

- `"none"` atau `0`: Tanpa bobot master. Parameter berpresisi rendah diupdate langsung menggunakan *Stochastic Rounding*.
- `"fp32"` atau `1`: Memelihara master weights FP32 untuk parameter berformat FP16 atau BF16. State akumulator momentum mengikuti presisi asli parameter untuk menghemat VRAM.
- `"pure_fp32"` atau `2`: Seluruh bobot master dan buffer state internal (`exp_avg`, `exp_avg_sq`) disimpan dalam FP32 penuh.

```python
optimizer = LuminaV(
    model.parameters(),
    lr=1e-3,
    master_weights="fp32",
    stochastic_rounding=True,
)
```

### Stochastic Rounding (SR)

Jika parameter atau momentum disimpan dalam presisi rendah (FP16/BF16), pembulatan deterministik standar (Round-to-Nearest-Even / RNE) dapat menyebabkan pembaruan gradien kecil mandek (*gradient stagnation*).

LuminaV mengintegrasikan generator PRNG berbasis XORShift/PCG yang mengeksekusi *Stochastic Rounding* secara deterministik per-langkah:
- Probabilitas bit dibulatkan ke atas proporsional terhadap sisa fraksi numerik.
- Mencegah stagnasi parameter tanpa overhead latensi sinkronisasi GPU/CPU.

---

## 3. Boundary Dynamics & Bounding

Untuk mencegah ledakan gradien (*gradient explosion*) saat lonjakan *loss spike*, LuminaV menyematkan pembatas dinamis:

```python
optimizer = LuminaV(
    model.parameters(),
    bound=True,
    bound_type="coordinate",  # Pilihan: 'coordinate' atau 'radial'
    bound_ratio=0.05,         # Maksimal pembaruan 5% dari norma parameter
)
```

### Mode Bounding: Coordinate vs Radial

1. **Coordinate Bounding (`bound_type="coordinate"` atau `2`)**:
   - Membatasi langkah per-elemen secara independen menggunakan formula hiperbolik tangen:
     $$B_i = \text{bound}_{\text{ratio}} \cdot \sqrt{\theta_i^2 + 10^{-4}}$$
     $$\Delta\theta_i = B_i \cdot \tanh\left(\frac{\Delta\theta_i}{B_i}\right)$$
   - Menjaga stabilitas numerik terhadap *sparse outliers* atau gradien ekstrem pada bobot embedding tertentu.

2. **Radial Bounding (`bound_type="radial"` atau `1`)**:
   - Membatasi langkah secara global pada ruang vektor bola (*spherical hyper-ball*):
     $$R = \text{bound}_{\text{ratio}} \cdot \max(\|\theta\|_2, 1.0)$$
     $$r = \frac{\eta \|\Delta\theta\|_2}{R}, \quad \text{scale} = \frac{\tanh(r)}{\max(r, 10^{-8})}$$
   - Mempertahankan arah vektor pembaruan asli tanpa distorsi sudut per-komponen.

---

## 4. Cautious Learning Momentum

Fitur `cautious=True` memastikan optimizer hanya memperbarui bobot jika arah vektor pembaruan $u$ selaras dengan tanda gradien instan $g$:

$$\text{mask}_i = \mathbb{I}(u_i \cdot g_i > 0)$$
$$\bar{m} = \max\left(\text{clamp}_{\text{min}}, \frac{1}{N}\sum_{i=1}^N \text{mask}_i\right)$$
$$\Delta\theta_i = u_i \cdot \text{mask}_i \cdot \frac{1}{\bar{m}}$$

Skalasi dinamis $\frac{1}{\bar{m}}$ memastikan magnitudo efektif pembaruan tetap konsisten meskipun sebagian parameter dinonaktifkan oleh mask.

---

## 5. Centered Defect Compensation

Dengan mengaktifkan `centered_defect=True`, LuminaV menghitung rata-rata deviasi temporal $\Delta_t = g_t - m_t$ di seluruh tensor atau shard, lalu memusatkan nilai defek tersebut:

$$\text{diff} = (g_t - m_t) - \mathbb{E}[g_t - m_t]$$

Hal ini mengeliminasi *bias drift* sistematik pada model distributed atau dataset yang memiliki fluktuasi statistik batch tinggi.

---

## 6. Fused Single-Pass Execution

Secara default, `fused_single_pass=True` diaktifkan:
- Pada langkah kedua dan seterusnya, pereduksian mask cautious dan skala radial diintegrasikan ke dalam nilai *Exponential Moving Average* (EMA) yang diperbarui secara asinkronus.
- Mengeliminasi kebutuhan *two-pass reduction kernel* pada GPU/CPU, memangkas *memory bandwidth* hingga 50% dan mempercepat *throughput step*.

---

## 7. State Checkpointing (`state_dict` & `load_state_dict`)

LuminaV mendukung serialisasi state penuh yang kompatibel dengan PyTorch dan NumPy:

```python
checkpoint = {
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
}
torch.save(checkpoint, "checkpoint.pt")

loaded_checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(loaded_checkpoint["model_state"])
optimizer.load_state_dict(loaded_checkpoint["optimizer_state"])
```

> [!NOTE]
> Buffer sementara seperti `p_contig`, `_plan`, dan `_contig_step` secara otomatis dibersihkan saat serialisasi `state_dict()` untuk menghasilkan ukuran berkas checkpoint yang minimal.

---

## 8. Distributed Training & FSDP / DDP

Pada arsitektur distributed (Fully Sharded Data Parallel / FSDP atau Distributed Data Parallel / DDP), shard parameter terbagi di beberapa GPU:

```python
optimizer = LuminaV(
    model.parameters(),
    lr=1e-3,
    fsdp_sync=True,  # Mengaktifkan sinkronisasi kolektif lintas rank GPU
    cautious=True,
    bound=True,
    bound_type="radial",
)
```

Saat `fsdp_sync=True` aktif:
- Reduksi global `mask.sum()`, $\|u\|_2^2$, dan $\|\theta\|_2^2$ disinkronkan lintas rank menggunakan operasi batched collective `all_reduce`.
- Shard kosong (*empty shards*) ditangani secara deterministik tanpa menyebabkan deadlock sinkronisasi FSDP.

---

## 9. Streaming Dataset Lokal Zero-Copy & HTTP 206 (`lm.stream`)

`LuminaMoon DataLoader` mengalirkan berkas lokal (`.parquet`, `.pq`, `.jsonl`, `.ndjson`, `.txt`) melalui pemetaan memori OS `mmap` (`memmap2`) zero-copy dan lompatan halaman `RowSelection` Arrow, atau mengalirkan dataset remote langsung dari `hf://datasets/...` dan `https://...` melalui worker HTTP 206 multi-range konkuren (`download_workers`) tanpa mengunduh berkas penuh ke disk:

```python
import luminamoon as lm

# Otomatis menjabarkan akar repositori ("hf://datasets/org/repo@main") menjadi daftar shard .parquet / .jsonl
with lm.stream(
    "hf://datasets/org/corpus@main",
    split="train",
    batch_size=64,
    preram_batches=100,
    refill_trigger=10,
    download_workers=4,
    deduplicate=True,
    auto_commit=True,
) as loader:
    for batch in loader:
        texts = batch.records
```

---

## 10. Ring Buffer Watermark Pre-RAM & Pemulihan Crash Dual-Metadata

Untuk menjaga utilitas loop pelatihan GPU tetap jenuh 100% sekaligus tahan terhadap kegagalan daya atau *spot preemption*:
1. **Ring Buffer Watermark Pre-RAM 100-Batch**: Thread latar belakang Rust khusus (`luminamoon-preram-worker`) menyiapkan hingga `preram_batches=100` batch di RAM dan otomatis mengisi ulang antrean ketika jumlah batch turun mencapai atau di bawah `preram_batches - refill_trigger` (misal $\le 90$ batch).
2. **Checkpoint Ping-Pong 128-Byte (`Slot A` / `Slot B`)**: Berkas `.meta` menulis secara bergantian di antara dua slot 64-byte berproteksi IEEE CRC32 (`Slot A` untuk epoch ganjil, `Slot B` untuk epoch genap) bersama *write-ahead log* `.meta.dedup`.
3. **Deferred `auto_commit` vs `batch.commit()` Eksplisit**:
   - Saat `auto_commit=True` (bawaan pada `lm.stream`), batch $N$ hanya di-commit ke disk setelah tubuh loop selesai memproses batch $N$ dan meminta batch $N+1$.
   - Untuk sinkronisasi eksplisit bersama `optimizer.step()`, setel `auto_commit=False` dan panggil `batch.commit()` (atau `loader.mark_in_gpu(batch.batch_id)`) tepat setelah step GPU selesai:
     ```python
     with lm.stream(shards, checkpoint_path="checkpoints/train.meta", auto_commit=False) as loader:
         for batch in loader:
             optimizer.step(grads=...)
             batch.commit()  # Memperbarui Slot A/B Ping-Pong dan WAL .dedup secara atomik
     ```

---

## 11. Deduplikasi Dua Tahap XXH3-64 & Integrasi Resmi `datasets`

`TwoStageDeduplicator` menghitung hash setiap baris menggunakan `xxh3_64` di atas SwissTable `IdentityBuildHasher` (`hashbrown`), memisahkan himpunan batch yang masih berada di Pre-RAM (`staged_lookup`) dari batch yang telah di-commit ke GPU (`committed_hashes`).

Pipeline normalisasi SIMD + Rayon + `xxh3_64` bebas-GIL ini juga dapat dipasang langsung di atas objek `datasets.Dataset`, `IterableDataset`, atau iterable Python melalui `lm.from_hf_dataset`:

```python
import luminamoon as lm

for batch in lm.from_hf_dataset(
    hf_dataset,
    column="text",
    batch_size=64,
    deduplicate=True,
    trim_whitespace=True,
    normalize_newlines=True,
    min_byte_len=8,
    max_byte_len=4096,
    truncate_over_max=True,
    tokenizer=my_tokenizer,
):
    tokens = batch.tokens
```

---

## 12. Streaming Parquet Multimodal & Unggahan Hibrida HuggingFace Hub

Memberikan lebih dari satu nama kolom (`columns=["prompt", "image", "score", "meta"]`) atau `multimodal=True` pada `lm.stream` akan menghasilkan objek `MultimodalBatch` yang mendukung tipe `Text`, `Bytes` (gambar/audio), `Json`, `Int64`, `Float64`, dan `Bool`.

Untuk melakukan serialisasi bebas-GIL di Rust dan mengunggah tabel multimodal kembali ke HuggingFace Hub:

```python
import luminamoon as lm

# Otomatis membaca HF_TOKEN, HF_ORG, dan HF_USERNAME dari os.environ, Google Colab userdata, atau Kaggle Secrets.
# Kolom byte gambar ditulis sebagai Arrow Struct<bytes: binary, path: utf8> agar langsung tampil di HF Dataset Viewer.
uri = lm.push_to_hub(
    repo="org/multimodal-cleaned",
    data=batch,  # Menerima DataBatch, MultimodalBatch, dict[str, list], atau list[dict]
    path_in_repo="train/part-00000.parquet",
    private=True,
)
```
