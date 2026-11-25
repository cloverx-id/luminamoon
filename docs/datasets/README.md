# Datasets (`cloverx-id` on Hugging Face Hub)

Public datasets maintained under the **`cloverx-id`** Hugging Face organization covering English-Indonesian machine translation, synthetic Indonesian ID card OCR/detection, and academic visual content moderation research.

---

## Dataset Catalog

| Dataset | Domain & Task | Scale / Configs | Format | License | Guide |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`lumi-repository-parallel-en-id-corpus`](https://huggingface.co/datasets/cloverx-id/lumi-repository-parallel-en-id-corpus) | Language · EN–ID Translation | `64.6M` clean (`default`), `54.8M` (`no_score`), `37.9M` (`drop_by_filter`) | Apache Parquet | `Apache-2.0` | [Parallel EN–ID Corpus](parallel_en_id_corpus.md) |
| [`indonesian-id-card-dummy`](https://huggingface.co/datasets/cloverx-id/indonesian-id-card-dummy) | Vision · Synthetic KTP OCR & Detection | `24,000` (`flat`) · `108,804` (`augmented`) | Apache Parquet | `CC-BY-4.0` | [Indonesian KTP 24K](indonesian_ktp_24k.md) |
| [`image-moderation-dataset`](https://huggingface.co/datasets/cloverx-id/image-moderation-dataset) | Vision · Binary Safety Gatekeeper | `30,000+` (`15k+` Safe / `15k+` Unsafe) | ZIP Archive | Custom Research | [Visual Safety & Moderation](image_moderation_safety.md) |
| [`unsafe-image-dataset`](https://huggingface.co/datasets/cloverx-id/unsafe-image-dataset) | Vision · Granular Unsafe Classes | Hazard-only multi-class archive | ZIP Archive | Custom Research | [Visual Safety & Moderation](image_moderation_safety.md) |

---

## Using Parquet Datasets with `luminamoon.data`

The Parquet-backed corpora (`lumi-repository-parallel-en-id-corpus` and `indonesian-id-card-dummy`) stream directly through Hugging Face `datasets` or local Parquet files and connect cleanly into `luminamoon.data.DataPipeline` and `FusedPipeline`:

```python
from datasets import load_dataset
from luminamoon.data import FusedPipeline

stream = load_dataset(
    "cloverx-id/lumi-repository-parallel-en-id-corpus",
    name="default",
    split="train",
    streaming=True,
)

batch_id = [row["id"] for _, row in zip(range(256), stream)]
cleaned = FusedPipeline(trim_whitespace=True, min_chars=8)(batch_id)
arrow_arr = cleaned.to_arrow()
```

!!! note "Scope & Research Notices"
    - `indonesian-id-card-dummy` contains **100% synthetic dummy identities** and is commercially safe under `CC-BY-4.0`.
    - `image-moderation-dataset` and `unsafe-image-dataset` contain unfiltered hazardous visual content restricted strictly to **AI safety engineering and institutional academic research** (zero CSAM or illegal exploitation material). Review the full terms on the [Visual Safety & Moderation](image_moderation_safety.md) page before downloading.
