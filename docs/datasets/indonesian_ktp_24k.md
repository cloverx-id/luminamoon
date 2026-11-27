# Indonesian KTP Dataset 24K (Synthetic Flat & 3D Augmented)

- **Hugging Face Repository**: [`cloverx-id/indonesian-id-card-dummy`](https://huggingface.co/datasets/cloverx-id/indonesian-id-card-dummy)
- **Pretty Name**: *Indonesian KTP Dataset 24K (Flat & 3D Perspective Augmented)*
- **Language**: Indonesian (`id`)
- **Tasks**: Object Detection (`object-detection`), Text Retrieval (`text-retrieval`), Document OCR / Key Information Extraction (DIE / KIE)
- **Format & License**: Apache Parquet · `CC-BY-4.0` (commercially safe synthetic data)

---

## Overview

The **Indonesian KTP Dataset 24K** is a synthetic, privacy-safe dataset designed for Document Information Extraction (DIE), Key Information Extraction (KIE), card localization, and OCR on Indonesian national ID card (*KTP-el*) layouts. All identities and fields are synthetically generated with dynamic anonymization—zero real citizen PII is included.

---

## Configurations & Verified Splits

| Config | Parquet Path Pattern | Samples (`train`) | Size (Uncompressed) | Composition |
| :--- | :--- | :--- | :--- | :--- |
| **`flat`** *(default)* | `data/flat/*.parquet` | `24,000` | `~12.49 GB` | Straight-on, axis-aligned card PNGs with rounded transparent corners plus full field and JSON annotations. |
| **`augmented`** | `data/augmented/*.parquet` | `108,804` | `~22.90 GB` | `72,000` 3D perspective-warped cards over realistic desk/office backgrounds, `12,804` certified clean hard negatives, and `24,000` flat cards on white backgrounds. |

!!! note "Spatial Positioning vs. Online Training Augmentation"
    The `augmented` subset models **physical 3D spatial positioning** (camera pitch, yaw, roll, and real background scenes) rather than pixel-level photometric noise. Keep standard photometric/sensor augmentations (blur, JPEG compression, exposure shifts) inside your training loop as needed.

---

## Column Schema (24 Fields)

| Group | Columns |
| :--- | :--- |
| **Image & Geometry** | `image`, `image_name`, `image_width`, `image_height`, `card_bounding_box`, `annotations_json` |
| **Identity Fields** | `nik`, `nama`, `tempat_tanggal_lahir`, `jenis_kelamin`, `golongan_darah`, `agama`, `status_perkawinan`, `pekerjaan`, `kewarganegaraan`, `berlaku_hingga` |
| **Address & Issuance** | `alamat`, `rt_rw`, `kel_desa`, `kecamatan`, `provinsi`, `kota`, `penerbitan_kota`, `penerbitan_tanggal` |

---

## Quickstart

```python
from datasets import load_dataset

# Load the default clean flat subset (24,000 samples)
flat_ds = load_dataset(
    "cloverx-id/indonesian-id-card-dummy",
    name="flat",
    split="train",
    streaming=True,
)
sample = next(iter(flat_ds))
print(sample["image_name"], sample["nik"], sample["card_bounding_box"])

# Load the 3D perspective-augmented + hard-negative subset (108,804 samples)
aug_ds = load_dataset(
    "cloverx-id/indonesian-id-card-dummy",
    name="augmented",
    split="train",
    streaming=True,
)
```
