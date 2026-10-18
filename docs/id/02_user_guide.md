# Panduan Penggunaan (User Guide)

Panduan mendalam mengenai fitur-fitur teknis LuminaV, strategi pemilihan mode buffer, mixed-precision, boundary dynamics, checkpointing, dan distributed training.

## 1. Pemilihan Mode Buffer: Dual-Buffer vs Single-Buffer

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
# Konfigurasi optimal untuk model FP16/BF16
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
     $$B_i = \text{bound\_ratio} \cdot \sqrt{\theta_i^2 + 10^{-4}}$$
     $$\Delta\theta_i = B_i \cdot \tanh\left(\frac{\Delta\theta_i}{B_i}\right)$$
   - Sangat tangguh terhadap *sparse outliers* atau gradien ekstrem pada bobot embedding tertentu.

2. **Radial Bounding (`bound_type="radial"` atau `1`)**:
   - Membatasi langkah secara global pada ruang vektor bola (*spherical hyper-ball*):
     $$R = \text{bound\_ratio} \cdot \max(\|\theta\|_2, 1.0)$$
     $$r = \frac{\eta \|\Delta\theta\|_2}{R}, \quad \text{scale} = \frac{\tanh(r)}{\max(r, 10^{-8})}$$
   - Mempertahankan arah vektor pembaruan asli tanpa distorsi sudut per-komponen.

---

## 4. Cautious Learning Momentum

Fitur `cautious=True` memastikan optimizer hanya memperbarui bobot jika arah vektor pembaruan $u$ selaras dengan tanda gradien instan $g$:

$$\text{mask}_i = \mathbb{I}(u_i \cdot g_i > 0)$$
$$\bar{m} = \max\left(\text{clamp\_min}, \frac{1}{N}\sum_{i=1}^N \text{mask}_i\right)$$
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
# Menyimpan checkpoint
checkpoint = {
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
}
torch.save(checkpoint, "checkpoint.pt")

# Memuat checkpoint
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
