# Referensi Konfigurasi & API (Configuration & API)

Dokumentasi lengkap tanda tangan kelas, tabel hyperparameter, metode publik, aturan validasi, dan penanganan galat pada `LuminaV`.

## Tanda Tangan Konstruktor

```python
from luminamoon import LuminaV

optimizer = LuminaV(
    params,
    lr: float = 8.0e-4,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1.0e-8,
    weight_decay: float = 8.0e-2,
    tau: float = 0.8,
    lam: float = 0.3,
    cautious: bool = True,
    cautious_clamp_min: float = 0.5,
    buffer: Union[int, str] = 2,
    centered_defect: bool = False,
    stochastic_rounding: bool = True,
    bound: bool = True,
    bound_type: Union[int, str] = "coordinate",
    bound_ratio: float = 0.05,
    master_weights: Union[bool, int, str] = "none",
    execution: Union[int, str] = "auto",
    fused_single_pass: bool = True,
    ema_decay: Optional[Union[float, str]] = "auto",
    fsdp_sync: bool = False,
    seed: int = 1337,
)
```

---

## Tabel Parameter Lengkap

| Parameter | Tipe | Nilai Default | Pilihan Valid | Deskripsi Teknis |
| :--- | :--- | :--- | :--- | :--- |
| `params` | `Iterable` | *(Wajib)* | Tensor list, NumPy array list, atau list dict parameter groups | Daftar bobot parameter model yang akan dioptimasi. |
| `lr` | `float` | `8.0e-4` | $> 0.0$ | Learning rate utama. |
| `betas` | `Tuple[float, float]` | `(0.9, 0.999)` | $0.0 \le \beta < 1.0$ | Koefisien peluruhan momentum tingkat pertama ($\beta_1$) dan tingkat kedua ($\beta_2$). |
| `eps` | `float` | `1.0e-8` | $> 0.0$ | Suku stabilitas pembagi numerik untuk menghindari pembagian dengan nol. |
| `weight_decay` | `float` | `8.0e-2` | $\ge 0.0$ | Koefisien pembusukan bobot independen (*decoupled weight decay* ala AdamW). |
| `tau` | `float` | `0.8` | $> 0.0$ | Parameter temperatur penskalaan CLM (Dual-Buffer). |
| `lam` | `float` | `0.3` | $\ge 0.0$ | Koefisien penalti Temporal-Difference Defect Envelope pada Single-Buffer TD-CLM. |
| `cautious` | `bool` | `True` | `True`, `False` | Mengaktifkan cautious masking (pembaruan hanya jika $u \cdot g > 0$). |
| `cautious_clamp_min` | `float` | `0.5` | $0.0 < x \le 1.0$ | Batas bawah rasio mask cautious agar pembaruan tidak mengalami *over-amplification*. |
| `buffer` | `int \| str` | `2` | `1`, `2`, `"single"`, `"dual"`, `"clm"`, `"tdclm"` | Jumlah buffer momentum: `2` (Dual-Buffer CLM) atau `1` (Single-Buffer TD-CLM). |
| `centered_defect` | `bool` | `False` | `True`, `False` | Memusatkan deviasi temporal $(g - m)$ dengan mengurangi nilai rata-ratanya. |
| `stochastic_rounding` | `bool` | `True` | `True`, `False` | Mengaktifkan pembulatan stokastik deterministik berbasis PRNG untuk FP16/BF16. |
| `bound` | `bool` | `True` | `True`, `False` | Mengaktifkan pembatas pembaruan gradien (*gradient step bounding*). |
| `bound_type` | `int \| str` | `"coordinate"` | `1`, `2`, `"coordinate"`, `"radial"` | Mode pembatasan pembaruan: `"coordinate"` (per-komponen) atau `"radial"` (vektor norma bola). |
| `bound_ratio` | `float` | `0.05` | $\ge 0.0$ | Rasio batas maksimum pembaruan terhadap norma parameter. `0.0` menonaktifkan bounding. |
| `master_weights` | `bool \| int \| str` | `"none"` | `"none"` / `0`, `"fp32"` / `1`, `"pure_fp32"` / `2` | Strategi penyimpanan bobot master FP32 saat melatih parameter FP16/BF16. |
| `execution` | `int \| str` | `"auto"` | `"auto"`, `"vulkan"`, `"simd"`, `"scalar"`, `"triton"`, `"single"` | Backend eksekusi perangkat keras yang diprioritaskan. |
| `fused_single_pass` | `bool` | `True` | `True`, `False` | Menggabungkan pembaruan bobot dan pelacakan EMA dalam satu lintasan kernel memori. |
| `ema_decay` | `float \| str` | `"auto"` | `"auto"` atau float $[0.0, 1.0)$ | Koefisien EMA untuk pelacakan mask/bound. `"auto"` memilih `0.95` (buffer 1) atau `0.80` (buffer 2). |
| `fsdp_sync` | `bool` | `False` | `True`, `False` | Mengaktifkan sinkronisasi kolektif global lintas rank GPU untuk FSDP/DDP. |
| `seed` | `int` | `1337` | Integer | Seed dasar untuk generator Stochastic Rounding per-parameter. |

---

## Metode Publik

### 1. `step(closure=None, grads=None)`
Mengeksekusi satu langkah pembaruan parameter.

- **Argumen**:
  - `closure` (*Optional[Callable]*): Fungsi closure opsional untuk menghitung ulang evaluasi model dan mengembalikan nilai loss (mendukung algoritma L-BFGS style).
  - `grads` (*Optional[List[Any] | Dict[Any, Any]]*): Daftar atau kamus gradien eksplisit. Wajib disediakan jika menggunakan standalone NumPy tanpa objek autograd PyTorch.
- **Return**: `Optional[float]` (nilai loss jika closure disertakan).

### 2. `zero_grad(set_to_none=True)`
Mereset gradien seluruh parameter yang terdaftar.

- **Argumen**:
  - `set_to_none` (*bool*): Jika `True`, menyetel `p.grad = None` (menghemat memori dan mempercepat alokasi PyTorch). Jika `False`, mengeksekusi `p.grad.zero_()`.

### 3. `state_dict()`
Mengembalikan kamus berisi state serialisasi lengkap optimizer (`state` dan `param_groups`). Secara otomatis membersihkan buffer memori sementara (`_plan`, `p_contig`, dll).

### 4. `load_state_dict(state_dict)`
Memuat state checkpoint ke dalam optimizer. Mendukung pemulihan status PyTorch Tensor maupun NumPy array, serta secara otomatis memulihkan parameter groups dan buffer status.

### 5. `detect_hardware()`
Metode statis untuk mendeteksi kapabilitas instruksi CPU SIMD, Vulkan UMA, dan threshold akselerasi GPU host saat ini.

---

## Validasi & Auto-Koreksi Typo

LuminaV dilengkapi validasi parameter ketat yang mencegah kesalahan pengetikan (*typo*) saat runtime. Jika pengguna memasukkan nilai string yang tidak dikenal, optimizer akan menghitung jarak Levenshtein dan memberikan saran perbaikan:

```python
# Contoh kesalahan pengetikan
LuminaV(model.parameters(), bound_type="radal")
# Menghasilkan:
# ValueError: Invalid bound_type: 'radal'. Expected 'coordinate' or 'radial'. Did you mean 'radial'?

LuminaV(model.parameters(), execution="vulkn")
# Menghasilkan:
# ValueError: Invalid execution mode: 'vulkn'. Expected 'auto', 'vulkan', 'simd', or 'scalar'. Did you mean 'vulkan'?
```
