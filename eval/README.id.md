# LSO-Eval: Rangkaian Verifikasi Formal & Pengujian Tolok Ukur

**LSO-Eval Suite** menyediakan verifikasi matematika formal, pembuktian invarian solver Z3 SMT, pengujian konvergensi lintasan 2D, dan audit keselarasan numerik IEEE untuk **LuminaV Optimizer** dan **LuminaMoon DataLoader** di bawah spesifikasi **PROJECT LUMI**, yang dikembangkan oleh **Lumina Moon ([@cloverx-id](https://github.com/cloverx-id))** dan **Silver Moon ([@cloverxion](https://github.com/cloverxion))**.

[English Version (README.md)](README.md)

---

## Komponen Evaluasi

| Komponen | Jalur | Deskripsi |
| :--- | :--- | :--- |
| **Spesifikasi Formal** | [`lso_eval_specification.tex`](lso_eval_specification.tex) | Formulasi matematika lengkap, lemma pembuktian, dan manuskrip LaTeX (standar tata letak IEEE dua kolom). |
| **Manuskrip PDF** | [`LSO-Eval.pdf`](LSO-Eval.pdf) | Berkas terkompilasi dari manuskrip spesifikasi formal dan catatan pembuktian. |
| **Tahap 1 (SMT & Presisi)** | [`stage1_smt_and_precision.py`](stage1_smt_and_precision.py) | Pembuktian solver Z3 SMT yang memverifikasi invarian batasan langkah pembaruan, non-divergensi CLM, serta jaminan pembulatan stokastik FP16/BF16. |
| **Orkestrator Eksekusi** | [`run_all.py`](run_all.py) | Skrip orkestrator utama untuk menjalankan seluruh tahapan evaluasi formal secara berurutan. |

---

## Menjalankan Rangkaian Verifikasi

```bash
# Jalankan seluruh rangkaian verifikasi formal
python -m eval

# Atau jalankan pembuktian SMT Tahap 1 secara langsung
python eval/stage1_smt_and_precision.py
```
