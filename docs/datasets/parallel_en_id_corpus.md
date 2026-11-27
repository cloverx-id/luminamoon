# Parallel English–Indonesian Corpus

- **Hugging Face Repository**: [`cloverx-id/lumi-repository-parallel-en-id-corpus`](https://huggingface.co/datasets/cloverx-id/lumi-repository-parallel-en-id-corpus)
- **Pretty Name**: *Xone Parallel English-Indonesian Corpus*
- **Languages**: English (`en`), Indonesian (`id`)
- **Task**: Machine Translation (`translation`)
- **Format & License**: Apache Parquet · `Apache-2.0`

---

## Overview

The **Lumi Parallel English-Indonesian Corpus** aggregates parallel sentence pairs compiled from 11 translation sources (including `opus100`, `opus_ccmatrix_en_id`, `paracrawl`, `opus_opensubtitles_en_id`, `quran_translation`, `flores`, and `nusax_mt_parallel`). It publishes sanitized training pairs alongside unfiltered progress shards and rejected rows with explicit filter reasons for diagnostic reproducibility.

---

## Configurations & Verified Splits

| Config | Parquet Path | Scale (`train`) | Purpose |
| :--- | :--- | :--- | :--- |
| **`default`** | `data/train-*.parquet` | `64,612,845` rows (`~13.91 GB`) | Sanitized, deduplicated parallel pairs with `LMSE 1.0 Q-SCORE`. |
| **`no_score`** | `progress/train-*.parquet` | `54,767,993` rows (`~241.63 GB`) | Unscored intermediate sentence pairs prior to LMSE scoring. |
| **`drop_by_filter`** | `drop_by_filter/dropped-*.parquet` | `37,870,529` rows (`~99.19 GB`) | Filtered-out sentence pairs retained with explicit rejection `reason`. |

!!! tip "Quality Score Guidance"
    Under the v1 scoring model (`LMSE 1.0 Q-SCORE`), pairs scoring **`55%` and above** indicate high-quality English–Indonesian translations.

---

## Column Schema

| Column | Type | Configs | Description |
| :--- | :--- | :--- | :--- |
| `no` | `int64` | All | Unique global row index. |
| `en` | `string` | All | English source or target sentence. |
| `id` | `string` | All | Indonesian target or source sentence. |
| `source` | `string` | All | Source dataset identifier (e.g., `opus100`, `flores`, `paracrawl`). |
| `domain` | `string` | All | Domain tag associated with the source corpus. |
| `LMSE 1.0 Q-SCORE` | `float` | `default`, `drop_by_filter` | Translation quality score produced by LMSE v1. |
| `reason` | `string` | `drop_by_filter` | Explicit filter rule explaining why the pair was excluded from `default`. |

---

## Quickstart: Streaming & `luminamoon.data`

```python
from datasets import load_dataset
from luminamoon.data import DataPipeline

# 1. Stream directly from Hugging Face Hub without downloading all shards upfront
ds = load_dataset(
    "cloverx-id/lumi-repository-parallel-en-id-corpus",
    name="default",
    split="train",
    streaming=True,
)

# 2. Filter high-scoring pairs and feed into DataPipeline for Best-Fit packing
def high_quality_records():
    for row in ds:
        if row["LMSE 1.0 Q-SCORE"] >= 55.0:
            yield {"text": f"{row['en']}\n{row['id']}"}

pipe = DataPipeline(
    source=high_quality_records(),
    tokenizer=my_tokenizer,
    batch_size=8,
    seq_len=2048,
    packing="best_fit",
    dedup="exact",
    checkpoint="./checkpoints/en_id_stream",
)
```
