# Referensi Konfigurasi & API (Configuration & API)

Dokumentasi lengkap tanda tangan kelas dan fungsi, tabel parameter, metode publik, aturan validasi, dan penanganan galat untuk **LuminaV Optimizer** maupun **LuminaMoon DataLoader**.

---

## Bagian I: API LuminaV Optimizer

### 1. Tanda Tangan Konstruktor (`LuminaV`)

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

### 2. Tabel Parameter Lengkap Optimizer

| Parameter | Tipe | Nilai Default | Pilihan Valid | Deskripsi Teknis |
| :--- | :--- | :--- | :--- | :--- |
| `params` | `Iterable` | *(Wajib)* | Tensor list, NumPy array list, atau list dict parameter groups | Daftar bobot parameter model yang akan dioptimasi. |
| `lr` | `float` | `8.0e-4` | $> 0.0$ | Learning rate utama. |
| `betas` | `Tuple[float, float]` | `(0.9, 0.999)` | $0.0 \le \beta < 1.0$ | Koefisien peluruhan momentum tingkat pertama ($\beta_1$) dan tingkat kedua ($\beta_2$). |
| `eps` | `float` | `1.0e-8` | $> 0.0$ | Suku stabilitas pembagi numerik untuk menghindari pembagian dengan nol. |
| `weight_decay` | `float` | `8.0e-2` | $\ge 0.0$ | Koefisien peluruhan bobot independen (*decoupled weight decay* ala AdamW). |
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

### 3. Metode Publik Optimizer

1. **`step(closure=None, grads=None)`**: Mengeksekusi satu langkah pembaruan parameter. `grads` wajib diisi saat menggunakan NumPy mandiri.
2. **`zero_grad(set_to_none=True)`**: Mereset gradien seluruh parameter yang terdaftar.
3. **`state_dict()`**: Mengembalikan kamus state serialisasi bersih tanpa buffer sementara.
4. **`load_state_dict(state_dict)`**: Memulihkan state checkpoint ke dalam optimizer.
5. **`detect_hardware()`**: Metode statis untuk mendeteksi kapabilitas CPU SIMD, Vulkan UMA, dan Triton AOT host.

### 4. Validasi & Auto-Koreksi Typo

Jika pengguna memasukkan opsi string yang salah ketik, `LuminaV` menghitung jarak Levenshtein dan memberikan saran perbaikan langsung:

```python
LuminaV(model.parameters(), bound_type="radal")
# ValueError: Invalid bound_type: 'radal'. Expected 'coordinate' or 'radial'. Did you mean 'radial'?
```

---

## Bagian II: API LuminaMoon DataLoader

### 1. `lm.stream(...)` & `DataStreamLoader`

```python
import luminamoon as lm

loader = lm.stream(
    sources: str | Path | Sequence[str | Path],
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = 64,
    preram_batches: int = 100,
    refill_trigger: int = 10,
    download_workers: int = 4,
    deduplicate: bool = True,
    auto_commit: bool = True,
    checkpoint_path: str | Path | None = None,
    json_extract_key: str | None = None,
    auth_token: str | None = None,
    tokenizer: Any = None,
    tokenizer_kwargs: Mapping[str, Any] | None = None,
    postprocess_fn: Callable[[Any], Any] | None = None,
    split: str | None = None,
    subset: str | None = None,
    **kwargs: Any,
)
```

| Parameter | Tipe | Nilai Default | Deskripsi Teknis |
| :--- | :--- | :--- | :--- |
| `sources` | `str \| Path \| Sequence` | *(Wajib)* | Jalur berkas lokal, URL `https://...`, atau URI `hf://datasets/<org>/<repo>[@rev]`. |
| `columns` | `Sequence[str] \| None` | `None` | Daftar proyeksi kolom. Lebih dari satu kolom otomatis diarahkan ke `MultimodalParquetLoader`. |
| `batch_size` | `int` | `64` | Jumlah rekaman bersih atau baris multimodal per batch. |
| `preram_batches` | `int` | `100` | Kapasitas maksimum batch di dalam ring buffer watermark Pre-RAM latar belakang. |
| `refill_trigger` | `int` | `10` | Selisih pemicu isi ulang ($\Delta$): worker bangun saat sisa antrean $\le \text{preram}_{\text{batches}} - \text{refill}_{\text{trigger}}$. |
| `download_workers` | `int` | `4` | Jumlah koneksi worker HTTP 206 multi-range Tokio yang berjalan secara paralel. |
| `deduplicate` | `bool` | `True` | Mengaktifkan deduplikasi dua tahap `xxh3_64` lintas batch Pre-RAM dan GPU. |
| `auto_commit` | `bool` | `True` (`stream`) / `False` (`DataStreamLoader`) | Auto-commit tertunda: menyimpan komit batch $N$ ke slot Ping-Pong 128B `.meta` saat batch $N+1$ diminta. |
| `checkpoint_path` | `str \| Path \| None` | `None` | Jalur berkas checkpoint `.meta` 128B (bawaan: `.luminamoon_cache/stream_<sha256>.meta`). |
| `parquet_column` | `str` | `"text"` | Nama kolom target saat mengalirkan berkas Parquet satu kolom. |
| `json_extract_key` | `str \| None` | `None` | Mengekstrak kunci string tertentu dari baris `.jsonl` di sisi Rust sebelum normalisasi. |
| `trim_whitespace` | `bool` | `True` | Memangkas spasi awal/akhir secara paralel melalui Rayon. |
| `normalize_newlines` | `bool` | `True` | Mengubah `\r\n` dan `\r` menjadi `\n` dengan deteksi awal SIMD `contains_byte_fast`. |
| `min_byte_len` | `int` | `1` | Panjang byte UTF-8 minimum agar rekaman tidak dibuang. |
| `max_byte_len` | `int` | `0` | Panjang byte UTF-8 maksimum (`0` = tanpa batas). |
| `truncate_over_max` | `bool` | `False` | Memotong rekaman yang melebihi `max_byte_len` tepat pada batas karakter UTF-8 (`is_char_boundary`). |
| `remote_chunk_bytes` | `int` | `262144` | Ukuran jendela byte HTTP 206 per worker (256 KB) untuk aliran `.jsonl`/`.txt` remote. |
| `auth_token` | `str \| None` | `None` | Token Bearer HuggingFace (otomatis dicari dari `os.environ`, Google Colab `userdata`, atau Kaggle Secrets). |
| `tokenizer` | `Callable \| None` | `None` | Fungsi/objek tokenizer (misal `AutoTokenizer`) untuk mengisi `batch.tokens`. |
| `tokenizer_kwargs` | `Mapping \| None` | `None` | Argumen kata kunci untuk `tokenizer` (bawaan: `{"padding": True, "truncation": True}`). |
| `postprocess_fn` | `Callable \| None` | `None` | Hook Python kustom `fn(batch) -> batch` pasca-tokenisasi. |
| `split` / `subset` | `str \| None` | `None` | Penyaring nama jalur shard saat menjabarkan akar repositori HuggingFace secara otomatis. |

### 2. `lm.from_hf_dataset(...)`

Membungkus `datasets.Dataset`, `IterableDataset`, atau iterable Python dengan normalisasi SIMD + Rayon dan deduplikasi `xxh3_64` bebas-GIL (`postprocess_and_dedup_records`):

```python
iterator = lm.from_hf_dataset(
    hf_dataset: Iterable[Any],
    *,
    column: str = "text",
    batch_size: int = 64,
    deduplicate: bool = True,
    trim_whitespace: bool = True,
    normalize_newlines: bool = True,
    min_byte_len: int = 1,
    max_byte_len: int = 0,
    truncate_over_max: bool = False,
    json_extract_key: str | None = None,
    tokenizer: Any = None,
    tokenizer_kwargs: Mapping[str, Any] | None = None,
    postprocess_fn: Callable[[Any], Any] | None = None,
) -> Iterator[DataBatch]
```

### 3. `lm.push_to_hub(...)` & `lm.serialize_table_to_bytes(...)`

Menyerialisasikan `DataBatch`, `MultimodalBatch`, `dict[str, list]`, atau `list[dict]` di Rust (`py.detach()`) ke format `.parquet` (kompatibel penuh dengan pratinjau gambar `struct<bytes: binary, path: utf8>` HuggingFace Dataset Viewer), `.jsonl`, atau `.json`, lalu mengunggahnya ke HuggingFace Hub:

```python
uri = lm.push_to_hub(
    repo: str,
    data: Any,
    path_in_repo: str = "data/part-00000.parquet",
    *,
    columns: Sequence[str] | None = None,
    rename: Mapping[str, str] | None = None,
    private: bool = True,
    revision: str = "main",
    commit_message: str = "Upload dataset shard via LuminaMoon",
    hf_image_struct: bool = True,
    token: str | None = None,
    username: str | None = None,
    org: str | None = None,
    endpoint: str | None = None,
) -> str
```

### 4. Objek Batch, Koordinat & Fungsi Utilitas

- **`DataBatch`**:
  - Properti: `batch_id: int`, `records: list[str]`, `tokens: Any`, `next_coord: StreamCoordinate`, `is_eof: bool`.
  - Metode: `batch.commit() -> StreamCoordinate`, `batch.to_dict() -> dict[str, list[str]]`, `batch.push_to_hub(repo, path_in_repo=...) -> str`, `len(batch)`, `batch[idx]`, `batch["text"]`, `batch["tokens"]`.
- **`MultimodalBatch`**:
  - Properti: `batch_id: int`, `num_rows: int`, `tokens: Any`, `next_coord: StreamCoordinate`, `is_eof: bool`.
  - Metode: `batch.column_names() -> list[str]`, `batch.column(name) -> list[Any]`, `batch.to_dict() -> dict[str, list[Any]]`, `batch.push_to_hub(repo, path_in_repo=...) -> str`, `batch[col_name]`, `batch[row_idx]`, `batch[slice]`.
- **`StreamCoordinate`**:
  - Field: `shard_id`, `byte_offset`, `row_group_idx`, `page_idx`, `row_offset_in_page`, `global_row_idx`.
  - Metode: `coord.to_bytes() -> bytes` (32-byte *little-endian*), `StreamCoordinate.from_bytes(raw) -> StreamCoordinate`.
- **`lm.expand_hf_sources(sources, *, split=None, subset=None, token=None, endpoint=None) -> list[str]`**:
  - Menemukan otomatis seluruh berkas shard `.parquet` dan `.jsonl` di dalam repositori `hf://datasets/...` (mendukung paginasi `Link` RFC 5988).
- **`lm.resolve_hf_secrets(token=None, username=None, org=None, endpoint=None) -> dict[str, str | None]`**:
  - Membaca kredensial HuggingFace secara berurutan dari argumen fungsi, `os.environ`, Google Colab `userdata`, atau Kaggle `UserSecretsClient`.
- **`lm.dataloader_cpu_isa() -> str`**:
  - Mengembalikan set instruksi SIMD pemindaian byte CPU yang aktif (`"AVX-512"`, `"AVX2"`, `"NEON"`, atau `"SCALAR"`).
